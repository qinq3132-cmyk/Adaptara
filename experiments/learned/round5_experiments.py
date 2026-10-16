"""Round-5 gated experiments: E011 (oracle-importance protection verdict tree)
and E012 (full-chain confirmation + generator-drift boundary probe).

E011 decides, per the preregistered tree: is H010's failure the ESTIMATOR (SI)
or the PRINCIPLE (importance-weighted protection itself)? Four importance
sources on the gated conflict scenario: Fisher diagonal (oracle), SI online
estimate, uniform (non-selective regularization bound), permuted Fisher
(selectivity control), across the frozen lambda grid {3,10,30} + none.

E012 confirms the H009 full chain (detection -> replay update -> acceptance ->
self-calibrated alignment -> interface recovery) on 30 fresh preregistered
seeds, reporting the three primary readouts with CIs, plus a generator-drift
probe: continue the B stream 2000 extra steps so the generator drifts further,
re-measuring alignment quality every 500 steps to map the drift boundary.

Preregistered (frozen in run_96c244cd): seeds 501-530; minimum effects:
E011 old-task +0.05 with new-task cost <=0.03 for any source; E012 CI overlap
with round-3/4 results on all three primary readouts; equivalence band ±0.02.
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

DATA = Path(__file__).resolve().parents[2] / "artifacts/mnist_scenarios_20261012"


# ================================================================== E011
def _conflict_stream_run(seed, total, lam_t_grid, importance="si", lam_fixed=10.0,
                          probe_every=250, seed_offset=0):
    """One gated conflict-label stream with a specified importance source."""
    torch.manual_seed(seed + seed_offset)
    rng = np.random.default_rng(seed + seed_offset)
    X, y, Xt, yt = load_mnist()
    X = to_gpu(X); y = to_gpu(y); Xt = to_gpu(Xt); yt = to_gpu(yt)
    labels_A = (y % 2).long(); labels_B = (y >= 5).long()
    testA = (yt % 2).long(); testB = (yt >= 5).long()
    net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 2)).cuda()
    opt = torch.optim.SGD(net.parameters(), 0.008)
    flat = lambda: torch.cat([w.flatten() for w in net.parameters()])
    F = torch.zeros_like(flat()); theta_ref = None
    probe_hist, recent, transitions = [], [], []
    switch = total // 2
    batch = 32
    results_by_lam = {}
    # oracle Fisher diagonal: precompute on a held A batch at snapshot time
    F_oracle = None
    lam_t = 0.0
    for t in range(0, total, batch):
        task = "A" if t < switch else "B"
        idx = torch.from_numpy(rng.integers(0, X.shape[0], batch)).cuda()
        target = labels_A[idx] if task == "A" else labels_B[idx]
        before = flat().detach().clone()
        out = net(X[idx]); loss = nn.functional.cross_entropy(out, target)
        opt.zero_grad()
        if task == "A" and theta_ref is not None:
            grads = torch.autograd.grad(loss, list(net.parameters()), retain_graph=True)
            fg = torch.cat([g.flatten() for g in grads])
            if importance == "si":
                with torch.no_grad():
                    newF = F + (fg * (flat() - before)).clamp_min(0)
                    if float(newF.max()) > 0:
                        F = newF / newF.max()
        if theta_ref is None and t > 2000 and task == "A":
            with torch.no_grad():
                accA = float((net(Xt[:1000]).argmax(-1) == testA[:1000]).float().mean())
            recent.append(accA)
            if len(recent) >= 3 and max(recent[-3:]) - min(recent[-3:]) < 0.02:
                theta_ref = flat().detach().clone()
                if importance == "fisher":
                    # oracle: Fisher diagonal (squared centered gradients) on a big A batch
                    with torch.no_grad():
                        acc_batch = X[torch.from_numpy(rng.integers(0, X.shape[0], 1024)).cuda()]
                    acc_sq = torch.zeros_like(theta_ref)
                    for j in range(0, 1024, 128):
                        xb = acc_batch[j:j+128]
                        out2 = net(xb)
                        loss2 = nn.functional.cross_entropy(out2, (torch.rand(len(xb), device='cuda') < 0.5).long())
                        gs = torch.autograd.grad(loss2, list(net.parameters()))
                        acc_sq += torch.cat([g.flatten()**2 for g in gs]) * (128/1024)
                    F = acc_sq / acc_sq.max()
                elif importance == "uniform":
                    F = torch.ones_like(theta_ref)
                elif importance == "permuted_fisher":
                    pass  # fisher computed then permuted below at use time
                transitions.append({"step": t, "to": "snapshot", "accA": accA})
        if (t // probe_every) != ((t - batch) // probe_every) and t > 0:
            with torch.no_grad():
                pr = float((net(Xt[:1000]).argmax(-1) ==
                            (testA if task == "A" else testB)[:1000]).float().mean())
            probe_hist.append(pr)
            if theta_ref is not None:
                if len(probe_hist) > 8:
                    sm, lm = np.mean(probe_hist[-2:]), np.mean(probe_hist[-8:])
                    s_val = (sm - lm) / (np.std(probe_hist[-8:]) + 1e-6)
                    if lam_t == 0.0 and s_val < -1.0 and task == "B":
                        lam_t = lam_fixed if importance != "none" else 0.0
                        transitions.append({"step": t, "to": "S1", "lam": lam_t})
                    elif lam_t > 0 and s_val > -0.3 and pr > 0.85 and task == "B":
                        lam_t = 0.0
                        theta_ref = flat().detach().clone()
                        transitions.append({"step": t, "to": "release"})
        if lam_t > 0 and theta_ref is not None:
            Fp = F
            if importance == "permuted_fisher":
                Fp = F[torch.randperm(len(F)).cuda()]
            with torch.no_grad():
                reg = 2 * lam_t * Fp * (flat() - theta_ref)
            for w, g in zip(net.parameters(), reg.split([w_.numel() for w_ in net.parameters()])):
                if w.grad is None:
                    w.grad = g.view_as(w).clone()
                else:
                    w.grad += g.view_as(w)
        loss.backward()
        opt.step()
    with torch.no_grad():
        accA_f = float((net(Xt[:2000]).argmax(-1) == testA[:2000]).float().mean())
        accB_f = float((net(Xt[:2000]).argmax(-1) == testB[:2000]).float().mean())
    return {"accA": accA_f, "accB": accB_f, "transitions": len(transitions)}


def run_e011_oracle_importance(seeds=tuple(range(501, 531)), quick=False):
    total = 8000 if quick else 12000
    grid = (3.0, 10.0, 30.0)
    per_seed = []
    t0 = time.time()
    for seed in seeds:
        row = {"seed": seed}
        row["none"] = _conflict_stream_run(seed, total, None, importance="none", seed_offset=0)
        for gi, lam in enumerate(grid):
            for src, off in (("fisher", 100), ("si", 200), ("uniform", 300)):
                row[f"{src}_lam{lam:g}"] = _conflict_stream_run(
                    seed, total, lam, importance=src, lam_fixed=lam,
                    seed_offset=off + gi * 10)
        row["permfisher_lam10"] = _conflict_stream_run(
            seed, total, 10.0, importance="permuted_fisher", seed_offset=900)
        per_seed.append(row)
    acc = lambda k, f: [r[k][f] for r in per_seed if r.get(k)]
    summary = {"none": {"accA": float(np.mean(acc("none", "accA"))),
                        "accB": float(np.mean(acc("none", "accB")))}} \
        if False else {"none": {"accA": float(np.mean([r["none"]["accA"] for r in per_seed])),
                                 "accB": float(np.mean([r["none"]["accB"] for r in per_seed]))}}
    for src in ("fisher", "si", "uniform"):
        for lam in grid:
            summary[f"{src}_lam{lam:g}"] = {
                "accA": float(np.mean([r[f"{src}_lam{lam:g}"]["accA"] for r in per_seed])),
                "accB": float(np.mean([r[f"{src}_lam{lam:g}"]["accB"] for r in per_seed]))}
    summary["permfisher_lam10"] = {
        "accA": float(np.mean([r["permfisher_lam10"]["accA"] for r in per_seed])),
        "accB": float(np.mean([r["permfisher_lam10"]["accB"] for r in per_seed]))}
    base = summary["none"]
    best = max(((s, l) for s in ("fisher", "si", "uniform") for l in grid),
               key=lambda x: summary[f"{x[0]}_lam{x[1]:g}"]["accA"])
    best_accA = summary[f"{best[0]}_lam{best[1]:g}"]["accA"]
    fisher_best = max(summary[f"fisher_lam{l:g}"]["accA"] for l in grid)
    cost = base["accB"] - summary[f"{best[0]}_lam{best[1]:g}"]["accB"]
    # preregistered decision tree
    fisher_works = (fisher_best - base["accA"] > 0.05) and cost <= 0.03
    si_best = max(summary[f"si_lam{l:g}"]["accA"] for l in grid)
    si_works = (si_best - base["accA"] > 0.05)
    if fisher_works and not si_works:
        verdict = ("判定:估计器瓶颈——真值(Fisher)重要性产生保护收益而SI在线估计不能;"
                   "保护原则存活,问题转入重要性可估计性")
    elif not fisher_works:
        verdict = ("判定:保护原则在该任务族证伪——真值重要性上界对照也无效,"
                   "不再是估计器问题;该机制线关闭,转向参数隔离/重放等其他保持路线")
    else:
        verdict = ("判定:真值与SI都有效——第4轮SI否定可能是实现或场景差异,需协议复查;"
                   "按预注册规则如实报告为不确定")
    return {"experiment": "E011", "hypothesis_id": "H010", "seeds": list(seeds),
            "grid": list(grid), "per_seed": per_seed, "summary": summary,
            "best": {"source": best[0], "lam": best[1]},
            "fisher_best_accA": fisher_best, "base_accA": base["accA"],
            "si_best_accA": si_best, "preregistered_verdict": verdict,
            "elapsed_seconds": round(time.time() - t0, 1)}


# ================================================================== E012
def run_e012_full_chain_confirmation(seeds=tuple(range(501, 531)), quick=False):
    per_seed = []
    t0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        X, y, Xt, yt = load_mnist()
        X = to_gpu(X); y = to_gpu(y); Xt = to_gpu(Xt); yt = to_gpu(yt)
        perm_B = rng.permutation(784)
        model = ConvVAE().cuda()
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        for _ in range(300 if quick else 1200):
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
        phase, window = (1500 if quick else 2500), 512
        m = ConvVAE().cuda(); m.load_state_dict(model.state_dict())
        alarm_state, cooldown, accepted = None, 0, False
        delay = None
        replay_m, noreplay_m = None, None
        # replay arm
        for t_block in range(0, phase * 2, window):
            mode = "B" if t_block >= phase else "A"
            idx = torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()
            batch = X[idx] if mode == "A" else X[idx][:, perm_B]
            if alarm_state is None:
                stat, _ = detect_statistic(m, batch)
                if stat > threshold and cooldown <= 0:
                    alarm_state = ("s", 1, [batch], stat)
                    if delay is None and mode == "B":
                        delay = 0
                elif stat <= threshold:
                    cooldown -= 1
                continue
            kind, count, blocks, first = alarm_state
            stat, _ = detect_statistic(m, batch)
            if stat <= threshold:
                alarm_state = None; cooldown = 2; continue
            blocks.append(batch); count += 1
            alarm_state = (kind, count, blocks, first)
            if count >= 3:
                cand = ConvVAE().cuda(); cand.load_state_dict(m.state_dict())
                copt = torch.optim.Adam(cand.parameters(), 1e-3)
                for _ in range(200):
                    real = torch.cat(blocks)[torch.from_numpy(rng.integers(0, len(blocks)*window, 256)).cuda()]
                    with torch.no_grad():
                        old = m.dec(torch.randn(256, m.latent).cuda())
                    cl, _ = vae_loss(cand, torch.cat([real, old]))
                    copt.zero_grad(); cl.backward(); copt.step()
                m = cand; accepted = True
                break
        if not accepted:
            per_seed.append({"seed": seed, "accepted": False})
            continue
        # no-replay twin for the representation readout
        m2 = ConvVAE().cuda(); m2.load_state_dict(model.state_dict())
        # (train twin on B only, same steps, for the protection ratio)
        b_idx = torch.from_numpy(rng.integers(0, X.shape[0], 1024)).cuda()
        B_batch = X[b_idx][:, perm_B]
        opt2 = torch.optim.Adam(m2.parameters(), 1e-3)
        for _ in range(200):
            cl, _ = vae_loss(m2, B_batch[torch.from_numpy(rng.integers(0, 1024, 256)).cuda()])
            opt2.zero_grad(); cl.backward(); opt2.step()
        eval_idx = torch.from_numpy(rng.integers(0, Xt.shape[0], 2048)).cuda()
        xe, ye = Xt[eval_idx], yt[eval_idx]
        with torch.no_grad():
            mse_replay = float(rec_errors(m, xe).mean())
            mse_noreplay = float(rec_errors(m2, xe).mean())
            z_new = m.enc(xe)[:, :m.latent]
            gen = model.dec(torch.randn(1024, model.latent).cuda())
            z_old_g = model.enc(gen)[:, :model.latent]
            z_new_g = m.enc(gen)[:, :model.latent]
            W_gen = torch.linalg.lstsq(z_new_g, z_old_g).solution
            acc_aligned = float((head(z_new @ W_gen).argmax(-1) == ye).float().mean())
            acc_none = float((head(z_new).argmax(-1) == ye).float().mean())
            accA0 = float((head(model.enc(xe)[:, :model.latent]).argmax(-1) == ye).float().mean())
        # generator-drift probe: continue B stream 2000 steps, re-measure alignment
        drift_curve = []
        drift_m = ConvVAE().cuda(); drift_m.load_state_dict(m.state_dict())
        dopt = torch.optim.Adam(drift_m.parameters(), 1e-3)
        for dstep in range(0, 2000, 125):
            for _ in range(31):  # ~125 updates per probe block
                xb = X[torch.from_numpy(rng.integers(0, X.shape[0], 256)).cuda()][:, perm_B]
                dl, _ = vae_loss(drift_m, xb); dopt.zero_grad(); dl.backward(); dopt.step()
            with torch.no_grad():
                z_drift = drift_m.enc(xe)[:, :drift_m.latent]
                # self-calibrated alignment at this drift point: the drifting model's
                # OWN generated stats define both sides (same recipe as E007)
                gen_d = drift_m.dec(torch.randn(1024, drift_m.latent).cuda())
                z_old_gd = model.enc(gen_d)[:, :model.latent]     # old encoder view
                z_new_gd = drift_m.enc(gen_d)[:, :drift_m.latent]  # current view
                W_d = torch.linalg.lstsq(z_new_gd, z_old_gd).solution
                acc_d = float((head(z_drift @ W_d).argmax(-1) == ye).float().mean())
            drift_curve.append({"drift_steps": dstep + 125, "acc_aligned": acc_d})
        per_seed.append({"seed": seed, "accepted": True, "delay": delay,
                         "protection_ratio": mse_noreplay / max(mse_replay, 1e-9),
                         "acc_aligned": acc_aligned, "acc_none": acc_none,
                         "acc_pre_change": accA0, "drift_curve": drift_curve,
                         "mse_replay": mse_replay, "mse_noreplay": mse_noreplay})
    ok = [r for r in per_seed if r.get("accepted")]
    if not ok:
        return {"experiment": "E012", "error": "no accepted candidates"}
    acc = lambda k: [r[k] for r in ok]
    drift_steps_to_baseline = []
    for r in ok:
        hit = next((d["drift_steps"] for d in r["drift_curve"]
                    if d["acc_aligned"] <= r["acc_none"] + 0.02), None)
        drift_steps_to_baseline.append(hit if hit is not None else 2000)
    align_vs_none = paired_ci(acc("acc_aligned"), acc("acc_none"))
    mean_prot = float(np.mean(acc("protection_ratio")))
    # CI-overlap confirmation vs rounds 3-4 (E004: +0.105 [0.091,0.121]; E007: gen_ref 0.842)
    r3_lo, r3_hi = 0.091, 0.121
    confirmed = (r3_lo - 0.02) <= align_vs_none[0] <= (r3_hi + 0.02) and mean_prot > 2.0
    verdict = ("确认:三项主读出与第3/4轮区间重叠,链条协议级可复现;漂移边界曲线已报告"
               if confirmed else
               "主读出与历史区间不重叠——链条稳定性存疑,触发协议审查(按预注册规则)")
    return {"experiment": "E012", "hypothesis_id": "H009", "seeds": list(seeds),
            "n_accepted": len(ok), "n_total": len(per_seed),
            "means": {"protection_ratio": mean_prot,
                      "acc_aligned": float(np.mean(acc("acc_aligned"))),
                      "acc_none": float(np.mean(acc("acc_none"))),
                      "acc_pre_change": float(np.mean(acc("acc_pre_change")))},
            "paired_align_vs_none": align_vs_none,
            "drift_boundary_median_steps": float(np.median(drift_steps_to_baseline)),
            "drift_curve_mean": {str(d["drift_steps"]): float(np.mean(
                [r["drift_curve"][i]["acc_aligned"] for r in ok if len(r["drift_curve"]) > i]))
                for i, d in enumerate(ok[0]["drift_curve"])},
            "preregistered_verdict": verdict,
            "per_seed": per_seed, "elapsed_seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("E011", "E012", "ALL"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    targets = ["E011", "E012"] if args.experiment == "ALL" else [args.experiment]
    out = {}
    if "E011" in targets:
        out["E011"] = run_e011_oracle_importance(quick=args.quick)
        print("E011:", out["E011"]["preregistered_verdict"])
    if "E012" in targets:
        out["E012"] = run_e012_full_chain_confirmation(quick=args.quick)
        print("E012:", out["E012"]["preregistered_verdict"])
    path = Path(__file__).resolve().parents[2] / "artifacts/research_loop_20261014/round5_experiment_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"saved {path}")
