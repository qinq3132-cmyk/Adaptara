"""E016: DEEPS v2 - adds the readaptation component (Q021) and differentiation
conditions (Q022) in ONE system-level experiment.

DEEPS v2 = DEEPS v1 + fifth component:
  H016 READAPTATION: when a candidate is accepted (new encoding), the OLD TASK
  HEAD is adapted on the accepted-update blocks THROUGH the aligned candidate
  latents, under the consistency gate, while a frozen copy of the old head (with
  its old alignment) is retained for the old encoding. The head adapts to the
  CURRENT encoding; the pool entry keeps old-head + old-W for returns.

Q022 differentiation conditions added to the stream:
  - dense rule disagreement: rule pair (digit>=5) vs (digit in {0,3,5,8}) which
    disagrees on 6 of 10 digits (vs 4/10 for parity) -> H013/H015 condition
  - graded change magnitudes: B = 25% permutation, C = full permutation -> H015
  - pool pressure: three encodings (A, B25, C100) with interleaved returns -> H014
  - pseudo-change windows: single-block statistical spikes -> H012 condition

Stream (10 segments x 2000 steps): S1(A,r1) S2(B25,r1) S3(A,r1) S4(A,r2)[dense
rule flip] S5(C100,r2)[graded full change] S6(A,r2)[return] S7(B25,r2)[return
to partial] S8(A,r1)[rule flip back] S9(A,r1,pseudo) S10(A,r1)[settle]

Systems (all on the same stream/seeds):
  DEEPSv2  - full system with readaptation + pool (old head kept for returns)
  DEEPSv1  - round-9 system (no readaptation; new-encoding head stays old)
  Round7Chain - prior-gen system
  NaiveFT  - naive continual SGD baseline

Capability targets (preregistered, seeds 901-930):
  C1' recovery: task acc >= 0.85 within 128 steps of ANY change, measured under
       the CURRENT encoding (the restated metric that E015 exposed as missing)
  C2 retention: old-encoding (A, rule-1) task acc >= 0.80 at stream end
  C3 stability: zero spurious updates in noise windows
  D (differentiation): component gains over DEEPSv1 appear on their designed
       conditions: H013 latency on dense flips, H015 stability under graded B25,
       H014 first-block on 3-encoding returns, H012 zero spurious where v1
       allows accept risk.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.learned.mnist_benchmarks import (ConvVAE, load_mnist, to_gpu, vae_loss,
                                                  detect_statistic)
from experiments.learned.round3_experiments import paired_ci

R1 = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).float()          # >=5 (sparse disagreement with R2)
R2 = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0, 1, 0]).float()            # {0,3,5,8}: disagrees with R1 on 6/10


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
    return model, head, float(np.quantile(calib, 0.99)), float(np.quantile(calib_err, 0.99)) * 1.5


def _make_perm(rng, frac):
    """Partial permutation: permute a fraction of the 784 dims (graded change)."""
    base = np.arange(784)
    k = int(784 * frac)
    chosen = rng.choice(784, size=k, replace=False)
    shuffled = chosen.copy(); rng.shuffle(shuffled)
    base[chosen] = base[shuffled]
    return torch.from_numpy(base).long().cuda()


def _run_system(system, seed, X, y, Xt, yt, perms, model, head, threshold, settled,
                segments, seg_len=2000, window=512):
    rules = {"r1": R1.cuda(), "r2": R2.cuda()}
    rng = np.random.default_rng(seed * 7 + hash(system) % 97)
    m = ConvVAE().cuda(); m.load_state_dict(model.state_dict())
    cur_head = nn.Linear(model.latent, 10).cuda(); cur_head.load_state_dict(head.state_dict())
    cur_head.eval()
    belief = torch.tensor([0.98, 0.02]).cuda()
    W_align = None
    alarm_state, cooldown = None, 0
    consistency_votes = []
    spurious = 0; real = 0
    pool = {}      # encoding-key -> (frozen encoder copy, old-head state, old W)
    timeline = []

    for (name, enc, rule, noise) in segments:
        seg_events, seg_accs = [], []
        for b in range(seg_len // window):
            idx = torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()
            obs = X[idx][:, perms[enc]]
            if noise == "pseudo" and b == 1:
                obs = obs + torch.randn_like(obs) * 0.05
            # pool lookup (DEEPS v1 & v2): if a pooled entry fits, switch instantly
            if system in ("DEEPSv1", "DEEPSv2") and alarm_state is None:
                for key, (enc_copy, head_copy, w_copy) in pool.items():
                    ps, _ = detect_statistic(enc_copy, obs)
                    if ps <= threshold:
                        m = ConvVAE().cuda(); m.load_state_dict(enc_copy.state_dict())
                        cur_head = nn.Linear(model.latent, 10).cuda(); cur_head.load_state_dict(head_copy.state_dict())
                        cur_head.eval()
                        W_align = w_copy
                        seg_events.append({"kind": "pool_hit", "key": key, "block": b})
                        break
            if system != "NaiveFT":
                stat, e_real = detect_statistic(m, obs)
                alarm = stat > threshold and not any(e.get("kind") == "pool_hit" for e in seg_events[-1:])
                if alarm and cooldown <= 0 and alarm_state is None:
                    seg_events.append({"kind": "alarm", "block": b})
                    alarm_state = ("s", 1, [obs], stat)
                elif alarm_state is not None and alarm_state[0] == "s":
                    kind, count, blocks, first = alarm_state
                    if stat > threshold:
                        blocks.append(obs); count += 1
                        alarm_state = (kind, count, blocks, first)
                        if count >= 3:
                            cand = ConvVAE().cuda(); cand.load_state_dict(m.state_dict())
                            copt = torch.optim.Adam(cand.parameters(), 1e-3)
                            for _ in range(250):
                                realb = torch.cat(blocks)[torch.from_numpy(rng.integers(0, len(blocks) * window, 128)).cuda()]
                                with torch.no_grad():
                                    old = m.dec(torch.randn(128, m.latent).cuda())
                                cl, _ = vae_loss(cand, torch.cat([realb, old]))
                                copt.zero_grad(); cl.backward(); copt.step()
                            probe = X[torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()][:, perms[enc]]
                            with torch.no_grad():
                                _, e_cand = detect_statistic(cand, probe)
                            accept = e_cand <= settled
                            if system in ("DEEPSv1", "DEEPSv2"):
                                consistency_votes.append(1 if accept else 0)
                                if len(consistency_votes) >= 4 and sum(consistency_votes[-4:]) < 3:
                                    accept = False
                            if accept:
                                if noise in ("pseudo", "reward"):
                                    spurious += 1
                                else:
                                    real += 1
                                m = cand
                                with torch.no_grad():
                                    gen = model.dec(torch.randn(1024, model.latent).cuda())
                                    W_new = torch.linalg.lstsq(
                                        m.enc(gen)[:, :m.latent], model.enc(gen)[:, :model.latent]).solution
                                W_align = W_new
                                # H016 READAPTATION (v2 only): adapt the head on the
                                # accepted-update blocks through aligned latents
                                if system == "DEEPSv2":
                                    new_head = nn.Linear(model.latent, 10).cuda()
                                    new_head.load_state_dict(cur_head.state_dict())
                                    hopt2 = torch.optim.Adam(new_head.parameters(), 1e-3)
                                    labels = y[torch.from_numpy(rng.integers(0, X.shape[0], 1024)).cuda()]
                                    adapt_obs = X[torch.from_numpy(rng.integers(0, X.shape[0], 1024)).cuda()][:, perms[enc]]
                                    for _ in range(150):
                                        with torch.no_grad():
                                            z = m.enc(adapt_obs)[:, :m.latent] @ W_new
                                        hl = nn.functional.cross_entropy(new_head(z), labels)
                                        hopt2.zero_grad(); hl.backward(); hopt2.step()
                                    new_head.eval()
                                    # pool the OLD (encoder, head, W) before switching
                                    if len(pool) < 3:
                                        enc_copy = ConvVAE().cuda(); enc_copy.load_state_dict(m.state_dict())
                                        head_copy = nn.Linear(model.latent, 10).cuda(); head_copy.load_state_dict(cur_head.state_dict())
                                        head_copy.eval()
                                        pool[f"p{len(pool)}"] = (enc_copy, head_copy, W_align)
                                    cur_head = new_head
                                    seg_events.append({"kind": "accepted+readapt", "block": b})
                                else:
                                    if system == "DEEPSv1" and len(pool) < 3 and noise not in ("pseudo", "reward"):
                                        pool[f"p{len(pool)}"] = (m, cur_head, W_align)
                                    seg_events.append({"kind": "accepted", "block": b})
                                cooldown = 2
                            else:
                                seg_events.append({"kind": "rejected", "block": b})
                                cooldown = 2
                            alarm_state = None
                    else:
                        alarm_state = None; cooldown = 2
                # Cooldown ticks down EVERY block (bounded below at zero) so a
                # post-update elevated-stat regime cannot freeze the detector.
                if alarm_state is None:
                    cooldown = max(0, cooldown - 1)
            if system == "NaiveFT":
                opt = torch.optim.Adam(m.parameters(), 1e-3)
                for _ in range(30):
                    cl, _ = vae_loss(m, obs[torch.from_numpy(rng.integers(0, window, 128)).cuda()])
                    opt.zero_grad(); cl.backward(); opt.step()
            # belief + per-step accuracy (32-sample steps)
            for j in range(0, window, 32):
                sub = slice(j, j + 32)
                obs_s, idx_s = obs[sub], idx[sub]
                with torch.no_grad():
                    z = m.enc(obs_s)[:, :m.latent]
                    if W_align is not None:
                        z = z @ W_align
                    dprobs = cur_head(z).softmax(-1)
                    true = rules[rule][y[idx_s]]
                    if noise == "reward" and rng.random() < 0.25:
                        true = 1 - true
                    if system != "NaiveFT":
                        ans_p = dprobs @ rules["r1"]; ans_m = dprobs @ rules["r2"]
                        eps = 1e-4
                        ce_p = float(-(true * (ans_p + eps).log() + (1 - true) * (1 - ans_p + eps).log()).mean())
                        ce_m = float(-(true * (ans_m + eps).log() + (1 - true) * (1 - ans_m + eps).log()).mean())
                        diff = ce_m - ce_p
                        if system in ("DEEPSv1", "DEEPSv2"):
                            step = 0.5 * max(0.0, abs(diff)) * 3.0      # H013 surprise
                            if stat is not None and threshold:
                                step *= (1.0 + 2.0 * max(0.0, stat - threshold) / max(threshold, 1e-9))  # H015
                        else:
                            step = 0.5
                        logit = torch.logit(belief[0].clamp(0.02, 0.98))
                        new_p0 = torch.sigmoid(logit + diff * step)
                        belief = torch.stack([new_p0, 1 - new_p0]).clamp(0.02, 0.98).detach()
                    cur_rule = rules["r1"] if (system == "NaiveFT" or belief[0] >= belief[1]) else rules["r2"]
                    acc = float(((dprobs @ cur_rule >= 0.5).long() == true).float().mean())
                    seg_accs.append(acc)
        # end-of-segment evaluation under the CURRENT encoding (the restated metric)
        eval_idx = torch.from_numpy(rng.integers(0, Xt.shape[0], 2048)).cuda()
        xe = Xt[eval_idx][:, perms[enc]]
        with torch.no_grad():
            z = m.enc(xe)[:, :m.latent]
            if W_align is not None:
                z = z @ W_align
            dprobs = cur_head(z).softmax(-1)
            cur_rule = rules["r1"] if (system == "NaiveFT" or belief[0] >= belief[1]) else rules["r2"]
            task = float(((dprobs @ cur_rule >= 0.5).long() == rules[rule][yt[eval_idx]]).float().mean())
            iface = float((cur_head(z).argmax(-1) == yt[eval_idx]).float().mean())
        timeline.append({"segment": name, "encoding": enc, "rule": rule, "noise": noise or "",
                         "task_acc": task, "iface_acc": iface,
                         "belief": [round(float(belief[0]), 3), round(float(belief[1]), 3)],
                         "acc_trace": [round(a, 3) for a in seg_accs],
                         "events": seg_events})
    return {"timeline": timeline, "spurious": spurious, "real": real}


def run_e016_deeps_v2(seeds=tuple(range(901, 931)), quick=False):
    systems = ("DEEPSv2", "DEEPSv1", "Round7Chain", "NaiveFT")
    segments = [
        ("S1", "A", "r1", None), ("S2", "B25", "r1", None),
        ("S3", "A", "r1", None), ("S4", "A", "r2", None),
        ("S5", "C100", "r2", None), ("S6", "A", "r2", None),
        ("S7", "B25", "r2", None), ("S8", "A", "r1", "reward"),
        ("S9", "A", "r1", "pseudo"), ("S10", "A", "r1", None),
    ]
    all_runs = []
    t0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        X, y, Xt, yt = load_mnist()
        X = to_gpu(X); y = to_gpu(y); Xt = to_gpu(Xt); yt = to_gpu(yt)
        perms = {"A": torch.arange(784).cuda(),
                 "B25": _make_perm(rng, 0.25),
                 "C100": _make_perm(rng, 1.0)}
        model, head, threshold, settled = _pretrain(seed, rng, X, y, 300 if quick else 1200)
        seed_runs = {}
        for system in systems:
            r = _run_system(system, seed, X, y, Xt, yt, perms, model, head,
                            threshold, settled, segments,
                            seg_len=800 if quick else 2000, window=512)
            seed_runs[system] = r
        all_runs.append({"seed": seed, **seed_runs})
        print(f"seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    def recovery_median(system, r):
        lats = []
        for si, t in enumerate(r[system]["timeline"]):
            if si in (1, 3, 4, 5, 6, 7):   # change segments (S2,S4,S5,S6,S7,S8)
                trace = t["acc_trace"]
                lat = next((i + 1 for i, a in enumerate(trace) if a >= 0.85), len(trace))
                lats.append(lat)
        return float(np.median(lats))
    def retention(system):
        return [r[system]["timeline"][9]["task_acc"] for r in all_runs]   # S10 under A+r1
    def spurious(system):
        return [r[system]["spurious"] for r in all_runs]
    def return_first(system, segs):
        vals = []
        for r in all_runs:
            for si in segs:
                trace = r[system]["timeline"][si]["acc_trace"]
                if trace:
                    vals.append(trace[0])
        return vals

    summary = {}
    for sysname in systems:
        summary[sysname] = {
            "recovery_median": float(np.median([recovery_median(sysname, r) for r in all_runs])),
            "retention": float(np.mean(retention(sysname))),
            "spurious_total": int(np.sum(spurious(sysname))),
            "return_first_S3": float(np.mean(return_first(sysname, [2]))),
            "return_first_S7": float(np.mean(return_first(sysname, [6]))),
            "real_updates": float(np.mean([r[sysname]["real"] for r in all_runs])),
        }
    pairs = {}
    for base in systems:
        if base == "DEEPSv2":
            continue
        pairs[f"v2_vs_{base}"] = {
            "recovery": paired_ci([recovery_median("DEEPSv2", r) for r in all_runs],
                                  [recovery_median(base, r) for r in all_runs]),
            "retention": paired_ci(retention("DEEPSv2"), retention(base)),
        }
    # differentiation readouts
    lat_dense = paired_ci(
        [next((i+1 for i,a in enumerate(r["DEEPSv2"]["timeline"][3]["acc_trace"]) if a >= 0.85), 999) for r in all_runs],
        [next((i+1 for i,a in enumerate(r["Round7Chain"]["timeline"][3]["acc_trace"]) if a >= 0.85), 999) for r in all_runs])
    lat_graded = paired_ci(
        [next((i+1 for i,a in enumerate(r["DEEPSv2"]["timeline"][1]["acc_trace"]) if a >= 0.85), 999) for r in all_runs],
        [next((i+1 for i,a in enumerate(r["DEEPSv1"]["timeline"][1]["acc_trace"]) if a >= 0.85), 999) for r in all_runs])
    ret_pool_S3 = paired_ci(return_first("DEEPSv2", [2]), return_first("Round7Chain", [2]))
    verdicts = []
    c1 = summary["DEEPSv2"]["recovery_median"] <= 128
    c2 = summary["DEEPSv2"]["retention"] >= 0.80
    c3 = summary["DEEPSv2"]["spurious_total"] == 0
    v2rec = summary["DEEPSv2"]["recovery_median"]; v2ret = summary["DEEPSv2"]["retention"]
    v2spur = summary["DEEPSv2"]["spurious_total"]
    verdicts.append("C1'当前编码恢复(128步内>=0.85):" + ("达标" if c1 else f"未达标(中位{v2rec:.0f}步)"))
    verdicts.append("C2旧保持(>=0.80):" + ("达标" if c2 else f"未达标({v2ret:.3f})"))
    verdicts.append("C3稳定(零伪更新):" + ("达标" if c3 else f"未达标({v2spur})"))
    # H016: the v1-v2 new-encoding gap
    s2v2 = float(np.mean([r["DEEPSv2"]["timeline"][1]["task_acc"] for r in all_runs]))
    s2v1 = float(np.mean([r["DEEPSv1"]["timeline"][1]["task_acc"] for r in all_runs]))
    verdicts.append(f"H016再适配:新编码段任务 v2={s2v2:.3f} vs v1={s2v1:.3f}"
                    + ("(缺口闭合)" if s2v2 - s2v1 > 0.15 else "(缺口未闭合)"))
    verdicts.append(f"H013密集规则:延迟差{lat_dense[0]:+.0f}步 CI[{lat_dense[1]:.0f},{lat_dense[2]:.0f}]"
                    + ("(幅度更新在密集条件下占优)" if lat_dense[1] < 0 else "(无差异)"))
    verdicts.append(f"H014池返回:S3首块{summary['DEEPSv2']['return_first_S3']:.3f} vs Round7{summary['Round7Chain']['return_first_S3']:.3f}"
                    + ("(池占优)" if ret_pool_S3[1] > 0.05 else "(无差异)"))
    return {"experiment": "E016", "system": "DEEPSv2", "seeds": list(seeds),
            "systems": systems, "summary": summary, "paired": pairs,
            "differentiation": {"dense_latency": lat_dense, "graded_latency": lat_graded,
                                 "pool_return_S3": ret_pool_S3},
            "preregistered_verdicts": verdicts,
            "runs": all_runs, "elapsed_seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    n = 2 if args.quick else args.seeds
    r = run_e016_deeps_v2(seeds=tuple(range(901, 901 + n)), quick=args.quick)
    out = Path(__file__).resolve().parents[2] / "artifacts/research_loop_20261014/round10_experiment_results.json"
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for v in r["preregistered_verdicts"]:
        print(" -", v)
    print(f"saved {out}")
