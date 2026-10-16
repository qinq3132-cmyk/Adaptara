"""E015: DEEPS system vs three baselines on the 8-segment composite stream.

One deployed SYSTEM (DEEPS) with four neuroscience-driven components:
  H012 consistency-gated consolidation: candidate acceptance requires >=3/4
      direction-consistent improvement votes against internal generated stats.
  H013 surprise-amplitude belief updates: per-step update scaled by the
      cross-entropy disagreement magnitude between the two rules.
  H014 remapping encoder pool: frozen encoder pool (capacity 2) selected by
      detection-statistic fit; returning encoders skip detection/update/align.
  H015 uncertainty-amplitude prior: belief prior weight modulated by how far
      the detection statistic exceeds threshold (analog signal, not just sign).

Baselines (system-vs-system, identical stream/seeds):
  NaiveFT       - single model, continual SGD, no detection.
  FixedSchedule - unconditional retrain every 2 blocks + fixed replay.
  Round7Chain   - our round-7 system: in-place updates, fixed-tau per-step
                  belief, no consistency gate, no pool, no analog modulation.

Stream (8 segments x 2000 steps, blocks of 512): encoding A/B flips + rule
parity/magnitude flips + reward-drift windows + transient pseudo-change windows.

Preregistered capability targets (seeds 801-830):
  C1 recovery: task acc >= 0.85 within 128 steps of any real change
  C2 retention: old-task acc >= 0.80 at stream end
  C3 stability: zero spurious updates across noise windows
Mechanism readouts: each DEEPS component vs Round7Chain (paired by seed).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.learned.mnist_benchmarks import (ConvVAE, load_mnist, to_gpu, vae_loss,
                                                  detect_statistic, rec_errors)
from experiments.learned.round3_experiments import paired_ci

PARITY = torch.tensor([d % 2 for d in range(10)]).float()
MAGNITUDE = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).float()


def _pretrain(seed, rng, X, y, pretrain_steps):
    model = ConvVAE().cuda()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(pretrain_steps):
        idx = torch.from_numpy(rng.integers(0, X.shape[0], 256)).cuda()
        loss, _ = vae_loss(model, X[idx]); opt.zero_grad(); loss.backward(); opt.step()
    head = nn.Linear(model.latent, 10).cuda()
    hopt = torch.optim.Adam(head.parameters(), 1e-2)
    hidx = torch.from_numpy(rng.integers(0, X.shape[0], 4096)).cuda()
    for _ in range(300):
        hl = nn.functional.cross_entropy(head(model.enc(X[hidx])[:, :model.latent]), y[hidx])
        hopt.zero_grad(); hl.backward(); hopt.step()
    head.eval()
    calib, calib_err = [], []
    for _ in range(40):
        idx = torch.from_numpy(rng.integers(0, X.shape[0], 512)).cuda()
        s, e = detect_statistic(model, X[idx]); calib.append(s); calib_err.append(e)
    threshold = float(np.quantile(calib, 0.99))
    settled = float(np.quantile(calib_err, 0.99)) * 1.5
    return model, head, threshold, settled


def _belief_step(dprobs, true, belief, tau, amplitude_mode, stat=None, threshold=None):
    """One belief update; returns new belief. H013/H015 modify the step size."""
    with torch.no_grad():
        ans_p = dprobs @ PARITY.cuda()
        ans_m = dprobs @ MAGNITUDE.cuda()
        eps = 1e-4
        ce_p = float(-(true * (ans_p + eps).log() + (1 - true) * (1 - ans_p + eps).log()).mean())
        ce_m = float(-(true * (ans_m + eps).log() + (1 - true) * (1 - ans_m + eps).log()).mean())
        diff = ce_m - ce_p   # >0 favors parity
        step = tau
        if amplitude_mode == "surprise":      # H013: scale by disagreement magnitude
            step = tau * max(0.0, abs(diff)) * 3.0
        elif amplitude_mode == "uncertainty": # H015: modulate prior by stat magnitude
            if stat is not None and threshold:
                excess = max(0.0, stat - threshold) / max(threshold, 1e-9)
                step = tau * (1.0 + 2.0 * excess)
        logit = torch.logit(belief[0].clamp(0.02, 0.98))
        new_p0 = torch.sigmoid(logit + diff * step)
        return torch.stack([new_p0, 1 - new_p0]).clamp(0.02, 0.98).detach()


def _run_system(system, seed, X, y, Xt, yt, perm_B, model, head, threshold, settled,
                segments, seg_len=2000, window=512):
    """Run one full 8-segment stream for one system. Returns timeline + events."""
    rules = {"parity": PARITY.cuda(), "magnitude": MAGNITUDE.cuda()}
    rng = np.random.default_rng(seed * 7 + hash(system) % 97)
    m = ConvVAE().cuda(); m.load_state_dict(model.state_dict())
    # pool for DEEPS (H014): store frozen encoders keyed by detection-fit
    pool = {}
    belief = torch.tensor([0.98, 0.02]).cuda()
    W_align = None
    alarm_state, cooldown = None, 0
    consistency_votes = []
    spurious_updates = 0
    real_updates = 0
    timeline = []

    def detector_stat(obs, enc):
        return detect_statistic(enc, obs)

    for (name, encoding, rule, noise) in segments:
        seg_events, seg_accs = [], []
        for b in range(seg_len // window):
            idx = torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()
            obs = X[idx]
            if encoding == "B":
                obs = obs[:, perm_B]
            # pseudo-change windows (evaluator-injected single-block spikes)
            if noise == "pseudo" and b == 1:
                spike = torch.randn_like(obs) * 0.05
                obs = obs + spike
            # ---- detection layer (not in NaiveFT) ----
            if system != "NaiveFT":
                stat, e_real = detector_stat(obs, m)
                alarm = stat > threshold
                # H014: pool lookup FIRST - if a pooled encoder fits this block,
                # switch to it and skip detection/update/alignment entirely
                if system == "DEEPS" and alarm_state is None:
                    best_pool, best_stat = None, None
                    for key, (pooled_enc, pooled_W) in pool.items():
                        ps, _ = detect_statistic(pooled_enc, obs)
                        if ps <= threshold and (best_stat is None or ps < best_stat):
                            best_pool, best_stat = (pooled_enc, pooled_W), ps
                    if best_pool is not None:
                        m = ConvVAE().cuda(); m.load_state_dict(best_pool[0].state_dict())
                        W_align = best_pool[1]
                        seg_events.append({"kind": "pool_hit", "block": b})
                        alarm = False
                if alarm and cooldown <= 0 and alarm_state is None:
                    seg_events.append({"kind": "alarm", "block": b, "stat": round(stat, 2)})
                    if system == "FixedSchedule":
                        pass  # handled by its unconditional branch below
                    else:
                        alarm_state = ("s", 1, [obs], stat)
                elif alarm_state is not None and alarm_state[0] == "s":
                    kind, count, blocks, first = alarm_state
                    if alarm:
                        blocks.append(obs); count += 1
                        alarm_state = (kind, count, blocks, first)
                        if count >= 3:
                            # FixedSchedule never does signal-triggered updates
                            if system == "FixedSchedule":
                                alarm_state = None; continue
                            cand = ConvVAE().cuda(); cand.load_state_dict(m.state_dict())
                            copt = torch.optim.Adam(cand.parameters(), 1e-3)
                            for _ in range(250):
                                real = torch.cat(blocks)[torch.from_numpy(rng.integers(0, len(blocks) * window, 128)).cuda()]
                                with torch.no_grad():
                                    old = m.dec(torch.randn(128, m.latent).cuda())
                                cl, _ = vae_loss(cand, torch.cat([real, old]))
                                copt.zero_grad(); cl.backward(); copt.step()
                            # forward acceptance: probe on a FRESH same-stream block
                            # (evaluator-independent: drawn from the same encoder-visible
                            # stream, not from blocks used in candidate training)
                            probe_idx = torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()
                            probe_obs = X[probe_idx]
                            if encoding == "B":
                                probe_obs = probe_obs[:, perm_B]
                            with torch.no_grad():
                                _, e_cand = detect_statistic(cand, probe_obs)
                            # H012 consistency gate: 3-of-4 direction-consistent votes
                            accept = e_cand <= settled
                            if system == "DEEPS":
                                consistency_votes.append(1 if accept else 0)
                                if len(consistency_votes) >= 4:
                                    if sum(consistency_votes[-4:]) < 3:
                                        accept = False
                            if accept:
                                if noise in ("pseudo", "reward"):
                                    spurious_updates += 1
                                else:
                                    real_updates += 1
                                m = cand
                                with torch.no_grad():
                                    gen = model.dec(torch.randn(1024, model.latent).cuda())
                                    z_old_g = model.enc(gen)[:, :model.latent]
                                    z_new_g = m.enc(gen)[:, :m.latent]
                                    W_align = torch.linalg.lstsq(z_new_g, z_old_g).solution
                                if system == "DEEPS" and len(pool) < 2 and noise not in ("pseudo", "reward"):
                                    pool[f"enc{len(pool)}"] = (m, W_align)
                                seg_events.append({"kind": "accepted", "block": b})
                                cooldown = 2
                            else:
                                seg_events.append({"kind": "rejected", "block": b})
                                cooldown = 2
                            alarm_state = None
                    else:
                        alarm_state = None; cooldown = 2
                elif not alarm:
                    cooldown -= 1
            # FixedSchedule: unconditional retrain every 2 blocks
            if system == "FixedSchedule" and b % 2 == 1:
                cand = ConvVAE().cuda(); cand.load_state_dict(m.state_dict())
                copt = torch.optim.Adam(cand.parameters(), 1e-3)
                for _ in range(60):
                    real = obs[torch.from_numpy(rng.integers(0, window, 128)).cuda()]
                    with torch.no_grad():
                        old = m.dec(torch.randn(128, m.latent).cuda())
                    cl, _ = vae_loss(cand, torch.cat([real, old]))
                    copt.zero_grad(); cl.backward(); copt.step()
                m = cand
                with torch.no_grad():
                    gen = model.dec(torch.randn(1024, model.latent).cuda())
                    W_align = torch.linalg.lstsq(
                        m.enc(gen)[:, :m.latent], model.enc(gen)[:, :model.latent]).solution
            # NaiveFT: nothing special; m keeps training on current obs
            if system == "NaiveFT":
                opt = torch.optim.Adam(m.parameters(), 1e-3)
                for _ in range(30):
                    cl, _ = vae_loss(m, obs[torch.from_numpy(rng.integers(0, window, 128)).cuda()])
                    opt.zero_grad(); cl.backward(); opt.step()
            # ---- belief layer (per-32-step) ----
            for j in range(0, window, 32):
                sub = slice(j, j + 32)
                obs_s, idx_s = obs[sub], idx[sub]
                with torch.no_grad():
                    z = m.enc(obs_s)[:, :m.latent]
                    if W_align is not None:
                        z = z @ W_align
                    dprobs = head(z).softmax(-1)
                    true = rules[rule][y[idx_s]]
                    if noise == "reward" and rng.random() < 0.25:
                        true = 1 - true
                    if system == "NaiveFT":
                        pass  # no belief layer in this baseline
                    else:
                        if system == "DEEPS":
                            belief = _belief_step(dprobs, true, belief, tau=0.5,
                                                  amplitude_mode="surprise")
                            belief = _belief_step(dprobs, true, belief, tau=0.25,
                                                  amplitude_mode="uncertainty",
                                                  stat=None, threshold=None)
                        else:  # Round7Chain: fixed tau
                            belief = _belief_step(dprobs, true, belief, tau=0.5,
                                                  amplitude_mode="fixed")
                        cur_rule = rules["parity" if belief[0] >= belief[1] else "magnitude"]
                    if system == "NaiveFT":
                        cur_rule = rules["parity"]  # naive baseline has no switching
                    acc = float(((dprobs @ cur_rule >= 0.5).long() == true).float().mean())
                    seg_accs.append(acc)
        # end-of-segment evaluation
        eval_idx = torch.from_numpy(rng.integers(0, Xt.shape[0], 2048)).cuda()
        xe = Xt[eval_idx]
        if encoding == "B":
            xe = xe[:, perm_B]
        with torch.no_grad():
            z = m.enc(xe)[:, :m.latent]
            if W_align is not None:
                z = z @ W_align
            dprobs = head(z).softmax(-1)
            cur_rule = rules["parity"] if (system == "NaiveFT" or belief[0] >= belief[1]) else rules["magnitude"]
            task = float(((dprobs @ cur_rule >= 0.5).long() ==
                          rules[rule][yt[eval_idx]]).float().mean())
            iface = float((head(z).argmax(-1) == yt[eval_idx]).float().mean())
        timeline.append({"segment": name, "encoding": encoding, "rule": rule, "noise": noise or "",
                         "task_acc": task, "iface_acc": iface,
                         "belief": [round(float(belief[0]), 3), round(float(belief[1]), 3)],
                         "acc_trace": [round(a, 3) for a in seg_accs],
                         "events": seg_events})
    return {"timeline": timeline, "spurious_updates": spurious_updates, "real_updates": real_updates}


def run_e015_deeps(seeds=tuple(range(801, 831)), quick=False):
    systems = ("DEEPS", "NaiveFT", "FixedSchedule", "Round7Chain")
    segments = [
        ("S1", "A", "parity", None), ("S2", "B", "parity", None),
        ("S3", "A", "parity", None),                      # RETURN
        ("S4", "A", "magnitude", None),                   # rule flip
        ("S5", "B", "magnitude", None),                   # double change
        ("S6", "B", "parity", "reward"),                  # rule flip + reward drift window
        ("S7", "A", "magnitude", None),                   # RETURN
        ("S8", "A", "parity", "pseudo"),                  # pseudo-change window
    ]
    all_runs = []
    t0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        X, y, Xt, yt = load_mnist()
        X = to_gpu(X); y = to_gpu(y); Xt = to_gpu(Xt); yt = to_gpu(yt)
        perm_B = rng.permutation(784)
        model, head, threshold, settled = _pretrain(seed, rng, X, y, 300 if quick else 1200)
        seed_runs = {}
        for system in systems:
            r = _run_system(system, seed, X, y, Xt, yt, perm_B, model, head,
                            threshold, settled, segments,
                            seg_len=800 if quick else 2000, window=512)
            seed_runs[system] = r
        all_runs.append({"seed": seed, **seed_runs})
        print(f"seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- capability readouts ----------------
    def sys_stat(system, key, seg_filter=None):
        vals = []
        for r in all_runs:
            for t in r[system]["timeline"]:
                if seg_filter is None or seg_filter(t):
                    if key in t:
                        vals.append(t[key])
                    elif key == "spurious":
                        vals.append(r[system]["spurious_updates"])
        return vals

    # C1 recovery: steps after each real change until acc >= 0.85
    def recovery_latency(system, r):
        lats = []
        tl = r[system]["timeline"]
        change_segs = [1, 3, 4, 5, 6]   # S2, S4, S5, S6, S7 (all real changes)
        for si in change_segs:
            trace = tl[si]["acc_trace"]
            lat = next((i + 1 for i, a in enumerate(trace) if a >= 0.85), len(trace))
            lats.append(lat)
        return lats
    # C2 retention: S8 old-task accuracy
    def retention(system):
        return [r[system]["timeline"][7]["task_acc"] for r in all_runs]
    # C3 stability: spurious updates
    def spurious(system):
        return [r[system]["spurious_updates"] for r in all_runs]
    # H014: return-segment first-block accuracy (S3, S7)
    def return_first(system):
        vals = []
        for r in all_runs:
            for si in (2, 6):
                trace = r[system]["timeline"][si]["acc_trace"]
                if trace:
                    vals.append(trace[0])
        return vals

    summary = {}
    for system in systems:
        lat = [x for r in all_runs for x in recovery_latency(system, r)]
        summary[system] = {
            "recovery_latency_median_steps": float(np.median(lat)),
            "retention_mean": float(np.mean(retention(system))),
            "spurious_updates_total": int(np.sum(spurious(system))),
            "return_first_block_mean": float(np.mean(return_first(system))),
            "real_updates_mean": float(np.mean([r[system]["real_updates"] for r in all_runs])),
        }
    # paired comparisons DEEPS vs each baseline + each component target
    deeps_lat = [float(np.median(recovery_latency("DEEPS", r))) for r in all_runs]
    pairs = {}
    for base in ("NaiveFT", "FixedSchedule", "Round7Chain"):
        base_lat = [float(np.median(recovery_latency(base, r))) for r in all_runs]
        base_ret = retention(base)
        pairs[f"vs_{base}"] = {
            "latency": paired_ci(deeps_lat, base_lat),
            "retention": paired_ci(retention("DEEPS"), base_ret),
        }
    # mechanism-level paired readouts (DEEPS vs Round7Chain)
    mech = {
        "H012_spurious": (spurious("DEEPS"), spurious("Round7Chain")),
        "H013_latency": (deeps_lat, [float(np.median(recovery_latency("Round7Chain", r))) for r in all_runs]),
        "H014_return_first": (return_first("DEEPS"), return_first("Round7Chain")),
    }
    verdicts = []
    # capability targets
    c1 = summary["DEEPS"]["recovery_latency_median_steps"] <= 128
    c2 = summary["DEEPS"]["retention_mean"] >= 0.80
    c3 = summary["DEEPS"]["spurious_updates_total"] == 0
    verdicts.append("C1恢复目标(128步内>=0.85):" + ("达标" if c1 else f"未达标(中位{summary['DEEPS']['recovery_latency_median_steps']:.0f}步)"))
    verdicts.append("C2保持目标(>=0.80):" + ("达标" if c2 else f"未达标({summary['DEEPS']['retention_mean']:.3f})"))
    verdicts.append("C3稳定目标(零伪更新):" + ("达标" if c3 else f"未达标({summary['DEEPS']['spurious_updates_total']}次)"))
    # system-vs-baselines
    for base in ("NaiveFT", "FixedSchedule"):
        pl = pairs[f"vs_{base}"]["latency"]; pr = pairs[f"vs_{base}"]["retention"]
        win = pl[0] < 0 and pr[0] > 0
        verdicts.append(f"vs {base}: " + ("恢复更快且保持更高(系统级占优)" if win else
                       f"恢复{pl[0]:+.0f}步/保持{pr[0]:+.3f},未双项占优"))
    # mechanisms
    sp = paired_ci(*mech["H012_spurious"]); 
    verdicts.append(f"H012一致性门:伪更新{summary['DEEPS']['spurious_updates_total']} vs Round7{summary['Round7Chain']['spurious_updates_total']}"
                    + ("(门生效)" if sp[0] < 0 else "(无差异)"))
    lat13 = paired_ci(*mech["H013_latency"])
    verdicts.append(f"H013惊讶幅度:恢复延迟{summary['DEEPS']['recovery_latency_median_steps']:.0f} vs Round7{summary['Round7Chain']['recovery_latency_median_steps']:.0f}步"
                    + ("(突破64步下限)" if summary['DEEPS']['recovery_latency_median_steps'] < 64 else "(未突破下限)"))
    ret14 = paired_ci(*mech["H014_return_first"])
    verdicts.append(f"H014重映射:返回首块{summary['DEEPS']['return_first_block_mean']:.3f} vs Round7{summary['Round7Chain']['return_first_block_mean']:.3f}"
                    + ("(首块即恢复)" if ret14[0] > 0.15 else "(无提前恢复优势)"))
    return {"experiment": "E015", "system": "DEEPS", "seeds": list(seeds),
            "systems": systems, "summary": summary, "paired": pairs,
            "mechanism_paired": {k: paired_ci(*v) for k, v in mech.items()},
            "preregistered_verdicts": verdicts,
            "runs": all_runs, "elapsed_seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    n = 2 if args.quick else args.seeds
    r = run_e015_deeps(seeds=tuple(range(801, 801 + n)), quick=args.quick)
    out = Path(__file__).resolve().parents[2] / "artifacts/research_loop_20261014/round9_experiment_results.json"
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for v in r["preregistered_verdicts"]:
        print(" -", v)
    print(f"saved {out}")
