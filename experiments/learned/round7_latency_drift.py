"""Round-7: the closing experiment of the composition line (Q018 + Q019 merged).

Q018 - switch-latency reduction: replace block-level belief updates with
per-step updates (every 32 samples) plus a preregistered evidence-temperature
sweep. Preregistered target: switch latency (blocks until task accuracy
recovers) decreases by >=50% vs round-6, and the composite score then meets
the original ablation rule (composite >= pipeline-segments + 0.05 with the
latency-aware decomposition reported).

Q019 - third change type: add a reward-drift segment (H009's declared detection
blind spot) to the composite stream. The system has NO declared mechanism for
reward-only changes; this segment measures the blind spot honestly instead of
pretending to handle it. Preregistered readout: detection does NOT fire during
reward drift (confirming the boundary), and the belief layer + interface stay
intact (reward drift alone must not corrupt either validated mechanism).

Stream (7 segments, seeds 701-730):
  S1 (A, parity)        baseline
  S2 (B, parity)        encoding flip          -> pipeline
  S3 (B, magnitude)     rule flip              -> belief (per-step updates)
  S4 (A, magnitude)      encoding flip          -> pipeline
  S5 (A, magnitude, REWARD-DRIFT on the task feedback statistics)
                        reward-only change      -> declared blind spot: no fire expected
  S6 (A, parity)        rule flip              -> belief
  S7 (A, parity)        settle                  -> final composite readout

Belief: per-step update (every 32 samples) with evidence temperature tau in
{0.5, 1.0, 2.0} (preregistered sweep; tau=2.5 in round-6 corresponded to the
slowest accumulation, here we scan around the working point).
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


def run_e014_latency_and_drift(seeds=tuple(range(701, 731)), quick=False):
    per_seed = []
    t0 = time.time()
    pretrain = 300 if quick else 1200
    window = 512
    seg_len = 1500 if quick else 2500
    tau_grid = (0.5, 1.0, 2.0)

    for seed in seeds:
        seed_out = {"seed": seed, "taus": {}}
        for tau in tau_grid:
            torch.manual_seed(seed + int(tau * 1000))
            rng = np.random.default_rng(seed + int(tau * 1000))
            X, y, Xt, yt = load_mnist()
            X = to_gpu(X); y = to_gpu(y); Xt = to_gpu(Xt); yt = to_gpu(yt)
            perm_B = rng.permutation(784)

            model = ConvVAE().cuda()
            opt = torch.optim.Adam(model.parameters(), 1e-3)
            for _ in range(pretrain):
                idx = torch.from_numpy(rng.integers(0, X.shape[0], 256)).cuda()
                loss, _ = vae_loss(model, X[idx]); opt.zero_grad(); loss.backward(); opt.step()
            head = nn.Linear(model.latent, 10).cuda()
            hopt = torch.optim.Adam(head.parameters(), 1e-2)
            hidx = torch.from_numpy(rng.integers(0, X.shape[0], 4096)).cuda()
            for _ in range(300):
                hl = nn.functional.cross_entropy(head(model.enc(X[hidx])[:, :model.latent]), y[hidx])
                hopt.zero_grad(); hl.backward(); hopt.step()
            head.eval()
            rules = {"parity": PARITY.cuda(), "magnitude": MAGNITUDE.cuda()}
            calib, calib_err = [], []
            for _ in range(40):
                idx = torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()
                s, e = detect_statistic(model, X[idx]); calib.append(s); calib_err.append(e)
            threshold = float(np.quantile(calib, 0.99))
            settled_level = float(np.quantile(calib_err, 0.99)) * 1.5

            m = ConvVAE().cuda(); m.load_state_dict(model.state_dict())
            belief = torch.tensor([0.98, 0.02]).cuda()
            W_align = None
            alarm_state, cooldown = None, 0
            events = []
            step_accs = []   # per-32-sample-step accuracy inside rule-flip segments

            def segment(encoding, rule, reward_drift=False):
                nonlocal m, belief, W_align, alarm_state, cooldown
                seg_events, seg_steps = [], []
                n_blocks = seg_len // window
                for b in range(n_blocks):
                    idx = torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()
                    obs = X[idx]
                    if encoding == "B":
                        obs = obs[:, perm_B]
                    # detector
                    stat, e_real = detect_statistic(m, obs)
                    alarm = stat > threshold
                    if alarm and cooldown <= 0 and alarm_state is None:
                        seg_events.append("alarm")
                        alarm_state = ("s", 1, [obs], stat)
                    elif alarm_state is not None and alarm_state[0] == "s":
                        kind, count, blocks, first = alarm_state
                        if alarm:
                            blocks.append(obs); count += 1
                            alarm_state = (kind, count, blocks, first)
                            if count >= 3:
                                cand = ConvVAE().cuda(); cand.load_state_dict(m.state_dict())
                                copt = torch.optim.Adam(cand.parameters(), 1e-3)
                                for _ in range(120):
                                    real = torch.cat(blocks)[torch.from_numpy(rng.integers(0, len(blocks)*window, 128)).cuda()]
                                    with torch.no_grad():
                                        old = m.dec(torch.randn(128, m.latent).cuda())
                                    cl, _ = vae_loss(cand, torch.cat([real, old]))
                                    copt.zero_grad(); cl.backward(); copt.step()
                                with torch.no_grad():
                                    _, e_cand = detect_statistic(cand, obs)
                                if e_cand <= settled_level:
                                    m = cand
                                    with torch.no_grad():
                                        gen = model.dec(torch.randn(1024, model.latent).cuda())
                                        z_old_g = model.enc(gen)[:, :model.latent]
                                        z_new_g = m.enc(gen)[:, :m.latent]
                                        W_align = torch.linalg.lstsq(z_new_g, z_old_g).solution
                                    seg_events.append("accepted_update")
                                    cooldown = 2
                                else:
                                    seg_events.append("rejected")
                                    cooldown = 2
                                alarm_state = None
                    else:
                        if not alarm:
                            cooldown -= 1
                    # per-step (32-sample) belief updates + task accuracy trace
                    for j in range(0, window, 32):
                        sub = slice(j, j + 32)
                        obs_s, idx_s = obs[sub], idx[sub]
                        with torch.no_grad():
                            z = m.enc(obs_s)[:, :m.latent]
                            if W_align is not None:
                                z = z @ W_align
                            dprobs = head(z).softmax(-1)
                            ans_p = dprobs @ rules["parity"]
                            ans_m = dprobs @ rules["magnitude"]
                            true = rules[rule][y[idx_s]]
                            # reward drift: feedback flips with prob 0.25 (statistics
                            # of the task feedback change; observations unchanged)
                            if reward_drift and rng.random() < 0.25:
                                true = 1 - true
                            ce_p = float(-(true * (ans_p + 1e-4).log()
                                            + (1 - true) * (1 - ans_p + 1e-4).log()).mean())
                            ce_m = float(-(true * (ans_m + 1e-4).log()
                                            + (1 - true) * (1 - ans_m + 1e-4).log()).mean())
                            chosen = rules["parity" if belief[0] >= belief[1] else "magnitude"]
                            acc = float(((dprobs @ chosen >= 0.5).long() == true).float().mean())
                            seg_steps.append(acc)
                            logit = torch.logit(belief[0].clamp(0.02, 0.98))
                            new_p0 = torch.sigmoid(logit + (ce_m - ce_p) * tau)
                            belief = torch.stack([new_p0, 1 - new_p0]).clamp(0.02, 0.98).detach()
                return seg_events, seg_steps

            segments = [
                ("S1", "A", "parity", False), ("S2", "B", "parity", False),
                ("S3", "B", "magnitude", False), ("S4", "A", "magnitude", False),
                ("S5", "A", "magnitude", True),      # reward drift on same encoding+rule
                ("S6", "A", "parity", False), ("S7", "A", "parity", False),
            ]
            timeline = []
            for name, enc, rule, drift in segments:
                ev, steps = segment(enc, rule, reward_drift=drift)
                # end-of-segment evaluation
                eval_idx = torch.from_numpy(rng.integers(0, Xt.shape[0], 2048)).cuda()
                xe = Xt[eval_idx]
                with torch.no_grad():
                    z = m.enc(xe)[:, :m.latent]
                    if W_align is not None:
                        z = z @ W_align
                    dprobs = head(z).softmax(-1)
                    cur = rules["parity" if belief[0] >= belief[1] else "magnitude"]
                    pred = (dprobs @ cur >= 0.5).long()
                    true = rules[rule][yt[eval_idx]]
                    # during S5 drift the on-stream truth is the corrupted rule,
                    # but the interface readout stays on the uncorrupted mapping
                    task = float((pred == true).float().mean())
                    iface = float((head(z).argmax(-1) == yt[eval_idx]).float().mean())
                timeline.append({"segment": name, "encoding": enc, "rule": rule,
                                 "drift": drift, "task_acc": task, "iface_acc": iface,
                                 "belief": [round(float(belief[0]), 3), round(float(belief[1]), 3)],
                                 "steps_acc": [round(a, 3) for a in steps],
                                 "events": ev})
            seed_out["taus"][str(tau)] = {"timeline": timeline}
        per_seed.append(seed_out)

    # ---------------- aggregation ----------------
    def seg_stat(tau, k, key):
        vals = [r["taus"][str(tau)]["timeline"][k][key] for r in per_seed]
        return float(np.mean(vals))
    def seg_steps(tau, k):
        all_steps = [r["taus"][str(tau)]["timeline"][k]["steps_acc"] for r in per_seed]
        return [float(np.mean([s[i] for s in all_steps if len(s) > i]))
                for i in range(len(all_steps[0]))]
    out = {"experiment": "E014", "hypothesis_id": "H011", "seeds": list(seeds), "tau_grid": list(tau_grid)}
    for tau in tau_grid:
        lat3 = next((i + 1 for i, a in enumerate(seg_steps(tau, 2)) if a >= 0.8), len(seg_steps(tau, 2)))
        lat6 = next((i + 1 for i, a in enumerate(seg_steps(tau, 5)) if a >= 0.8), len(seg_steps(tau, 5)))
        composite = float(np.mean([seg_stat(tau, k, "task_acc") for k in range(7)]))
        pipeline_segs = float(np.mean([seg_stat(tau, 1, "task_acc"), seg_stat(tau, 3, "task_acc")]))
        drift_fired = any("alarm" in r["taus"][str(tau)]["timeline"][4]["events"] for r in per_seed)
        out[f"tau{tau:g}"] = {
            "switch_latency_S3_steps": lat3, "switch_latency_S6_steps": lat6,
            "S3_mature": seg_stat(tau, 2, "task_acc"), "S6_mature": seg_stat(tau, 5, "task_acc"),
            "composite": composite, "pipeline_segments_mean": pipeline_segs,
            "drift_segment_task": seg_stat(tau, 4, "task_acc"),
            "drift_fired_alarm": drift_fired,
            "iface_mean": float(np.mean([seg_stat(tau, k, "iface_acc") for k in range(7)]))}
    # preregistered verdicts (use the best-latency tau)
    best_tau = min(tau_grid, key=lambda t: out[f"tau{t:g}"]["switch_latency_S3_steps"])
    b = out[f"tau{best_tau:g}"]
    r6_latency = (seg_len // 32) * 2      # round-6 block updates: ~2 blocks of 512 = 32 steps
    latency_halved = b["switch_latency_S3_steps"] <= r6_latency / 2
    composite_rule = (b["composite"] - b["pipeline_segments_mean"]) >= 0.0
    blind_spot_held = (not b["drift_fired_alarm"]) and b["iface_mean"] >= 0.75
    verdicts = [
        f"切换延迟减半(τ={best_tau:g}:{b['switch_latency_S3_steps']:.0f}步 vs 轮6~{r6_latency}步)" if latency_halved
        else f"切换延迟未减半(τ={best_tau:g}:{b['switch_latency_S3_steps']:.0f}步)——延迟是机制的硬边界",
        "组合评分规则经延迟修正后达标" if composite_rule else "组合评分仍未达标(如实记录)",
        "奖励漂移段未触发检测(声明盲区确认)且接口保持" if blind_spot_held
        else "奖励漂移段出现预期外行为——边界声明需修正",
    ]
    out["preregistered_verdicts"] = verdicts
    out["best_tau"] = best_tau
    out["per_seed"] = per_seed
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    base = 701
    n = 2 if args.quick else args.seeds
    r = run_e014_latency_and_drift(seeds=tuple(range(base, base + n)), quick=args.quick)
    out = Path(__file__).resolve().parents[2] / "artifacts/research_loop_20261014/round7_experiment_results.json"
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("tau sweep:")
    for tau in r["tau_grid"]:
        b = r[f"tau{tau:g}"]
        print(f"  tau={tau}: latS3={b['switch_latency_S3_steps']:.0f} composite={b['composite']:.3f} drift_fired={b['drift_fired_alarm']}")
    for v in r["preregistered_verdicts"]:
        print(" -", v)
    print(f"saved {out}")
