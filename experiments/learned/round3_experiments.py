"""E004/E005 implementations for the round-3 gated experiments.

E004 (H009, latent alignment): after the detection->update->accept flow produces a
frozen candidate encoder, evaluate the OLD-task interface under four conditions:
latent alignment (least-squares map to the pre-change latent reference), random
orthogonal alignment (matched compute), head-retraining (linear probe), and no
alignment (round-1 baseline). Separates distribution-misalignment from
head-degradation explanations for the observed interface loss.

E005 (H010, lambda sweep): run the full protection state machine across the
pre-registered grid lambda in {0,1,3,10,30} plus a permuted-importance control at
lambda=10, each an independent full stream, and report the retention-recovery
tradeoff frontier.

Preregistered parameters (frozen in run_ee166a17…):
  seeds 301-320; minimum effects: interface +0.05 (E004), old-perm +0.05 with
  current-perm cost <=0.03 (E005); equivalence bands ±0.02; paired bootstrap CIs.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.learned.mnist_benchmarks import (ConvVAE, load_mnist, to_gpu, vae_loss,
                                                  detect_statistic, rec_errors, run_h009,
                                                  run_h010)
import experiments.learned.mnist_benchmarks as MB

DATA = Path(__file__).resolve().parents[2] / "artifacts/mnist_scenarios_20261012"


def paired_ci(a, b, confidence=0.95, n_boot=20000, seed=917):
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return float(d.mean()), float(lo), float(hi)


# ================================================================== E004
def run_h009_alignment(seeds=tuple(range(301, 321)), quick=False):
    """Four-condition interface evaluation on frozen candidate encoders.

    Reuses the round-1 H009 flow (detection -> sustained alarm -> update blocks ->
    forward acceptance) to produce the frozen candidate, then measures the old-task
    interface under the four preregistered conditions.
    """
    results = []
    t0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        X, y, Xt, yt = load_mnist()
        X = to_gpu(X); y = to_gpu(y); Xt = to_gpu(Xt); yt = to_gpu(yt)
        perm_B = rng.permutation(784)

        # --- reproduce the round-1 protocol to obtain the frozen candidate encoder
        model = ConvVAE().cuda()
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        pretrain = 300 if quick else 1200
        for _ in range(pretrain):
            idx = torch.from_numpy(rng.integers(0, X.shape[0], 256)).cuda()
            loss, _ = vae_loss(model, X[idx]); opt.zero_grad(); loss.backward(); opt.step()
        # frozen old head on ORIGINAL encoder latents
        head = nn.Linear(model.latent, 10).cuda()
        hopt = torch.optim.Adam(head.parameters(), 1e-2)
        hidx = torch.from_numpy(rng.integers(0, X.shape[0], 4096)).cuda()
        for _ in range(300):
            hloss = nn.functional.cross_entropy(head(model.enc(X[hidx])[:, :model.latent]), y[hidx])
            hopt.zero_grad(); hloss.backward(); hopt.step()
        head.eval()
        # calibration on A
        calib, calib_err = [], []
        for _ in range(40):
            idx = torch.from_numpy(rng.integers(0, X.shape[0], 512)).cuda()
            s, e = detect_statistic(model, X[idx]); calib.append(s); calib_err.append(e)
        threshold = float(np.quantile(calib, 0.99))
        settled_level = float(np.quantile(calib_err, 0.99)) * 1.5
        # old-reference latent statistics (pre-change, permitted: deployment-time)
        with torch.no_grad():
            ref = model.enc(X[hidx])[:, :model.latent]
            ref_mean = ref.mean(0); ref_cov = torch.linalg.vecdot(ref.t(), ref.t()) / len(ref)

        # --- A->B stream with the round-1 update flow
        phase, window = (1500 if quick else 2500), 512
        m = ConvVAE().cuda(); m.load_state_dict(model.state_dict())
        alarm_state, cooldown = None, 0
        accepted = False
        for t_block in range(0, phase * 2, window):
            mode = "B" if t_block >= phase else "A"
            idx = torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()
            batch = X[idx] if mode == "A" else X[idx][:, perm_B]
            if alarm_state is None:
                stat, _ = detect_statistic(m, batch)
                if stat > threshold and cooldown <= 0:
                    alarm_state = ("sustain", 1, [batch], stat)
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
                    real = torch.cat(blocks)[torch.from_numpy(rng.integers(0, len(blocks) * window, 256)).cuda()]
                    with torch.no_grad():
                        old = m.dec(torch.randn(256, m.latent).cuda())
                    closs, _ = vae_loss(cand, torch.cat([real, old]))
                    copt.zero_grad(); closs.backward(); copt.step()
                alarm_state = None; m = cand; accepted = True
                break
        if not accepted:
            continue  # seed produced no accepted candidate; excluded and reported

        # --- four-condition interface evaluation on the SAME frozen candidate
        eval_idx = torch.from_numpy(rng.integers(0, Xt.shape[0], 2048)).cuda()
        xe, ye = Xt[eval_idx], yt[eval_idx]
        with torch.no_grad():
            z_old_ref = model.enc(xe)[:, :model.latent]      # pre-change encoder
            z_new = m.enc(xe)[:, :model.latent]              # candidate encoder
            # 1) no alignment: candidate latents straight into the old head
            acc_none = float((head(z_new).argmax(-1) == ye).float().mean())
            # 2) latent alignment: least-squares map z_new -> z_old_ref on a fitting split
            fit_n = 1024
            A_fit, b_fit = z_new[:fit_n], z_old_ref[:fit_n]
            W = torch.linalg.lstsq(A_fit, b_fit).solution
            z_aligned = z_new @ W
            acc_align = float((head(z_aligned).argmax(-1) == ye).float().mean())
            # 3) random orthogonal alignment: same dims, same matrix estimation cost
            g = torch.randn_like(torch.empty(model.latent, model.latent)).cuda()
            Q, _ = torch.linalg.qr(g)
            acc_ortho = float((head(z_new @ Q).argmax(-1) == ye).float().mean())
        # 4) retrained head (linear probe) on the candidate latents (outside no_grad)
        probe_head = nn.Linear(m.latent, 10).cuda()
        popt = torch.optim.Adam(probe_head.parameters(), 1e-2)
        pidx = torch.from_numpy(rng.integers(0, X.shape[0], 2048)).cuda()
        for _ in range(200):
            ploss = nn.functional.cross_entropy(
                probe_head(m.enc(X[pidx])[:, :m.latent]), y[pidx])
            popt.zero_grad(); ploss.backward(); popt.step()
        with torch.no_grad():
            acc_probe = float((probe_head(z_new).argmax(-1) == ye).float().mean())
            mse_old = float(rec_errors(m, xe).mean())
            alignment_residual = float((z_aligned - z_old_ref).norm(dim=1).mean())
        results.append({"seed": seed, "acc_no_alignment": acc_none, "acc_latent_alignment": acc_align,
                        "acc_random_orthogonal": acc_ortho, "acc_retrained_head": acc_probe,
                        "old_rec_mse": mse_old, "alignment_residual": alignment_residual})

    if not results:
        return {"hypothesis_id": "H009", "experiment": "E004", "seeds": list(seeds),
                "error": "no seed produced an accepted candidate encoder"}
    acc = lambda k: [r[k] for r in results]
    align_vs_none = paired_ci(acc("acc_latent_alignment"), acc("acc_no_alignment"))
    align_vs_ortho = paired_ci(acc("acc_latent_alignment"), acc("acc_random_orthogonal"))
    verdict = ("支持潜在分布错位解释" if align_vs_none[1] > 0.05 and align_vs_ortho[1] > 0.05 else
               "否定错位解释:对齐与对照等效或差异不足" if align_vs_none[2] <= 0.02 or align_vs_ortho[2] <= 0.02
               else "不确定:区间跨越预注册边界")
    return {"hypothesis_id": "H009", "experiment": "E004", "seeds": list(seeds),
            "n_evaluated": len(results), "per_seed": results,
            "means": {k: float(np.mean(acc(k))) for k in
                      ("acc_no_alignment", "acc_latent_alignment", "acc_random_orthogonal", "acc_retrained_head")},
            "paired": {"align_minus_none": align_vs_none, "align_minus_ortho": align_vs_ortho},
            "preregistered_verdict": verdict,
            "scope": "gated round-3 experiment; seeds 301-320 preregistered",
            "elapsed_seconds": round(time.time() - t0, 1)}


# ================================================================== E005
def run_h010_lambda(seeds=tuple(range(301, 321)), quick=False):
    """Pre-registered lambda sweep: each grid point is an independent full stream.

    Grid {0,1,3,10,30} + permuted-importance control at lambda=10. Reads the
    retention-recovery tradeoff frontier: old-perm accuracy, current-perm
    accuracy, recovery latency, and state-machine legality per lambda.
    """
    grid = (0.0, 1.0, 3.0, 10.0, 30.0)
    total = 8000 if quick else 12000   # quick still needs >=3 permutations for separation
    per_seed = []
    t0 = time.time()
    for seed in seeds:
        row = {"seed": seed}
        for i, lam in enumerate(grid):
            r = MB.run_h010(seed, total=total, lambda_override=lam, seed_offset=i * 1000)
            full = r["arms"]["full"]
            row[f"lam{lam:g}"] = {"old": full["final_old_perm_accuracy"],
                                  "current": full["final_current_perm_accuracy"],
                                  "mean_current": full["mean_current_acc"],
                                  "transitions": len(full["state_transitions"])}
        # permuted-importance control at lambda=10 (independent draw)
        r = MB.run_h010(seed, total=total, lambda_override=10.0, seed_offset=9000)
        pf = r["arms"]["permuted_F"]
        row["permF_lam10"] = {"old": pf["final_old_perm_accuracy"],
                              "current": pf["final_current_perm_accuracy"]}
        per_seed.append(row)
    acc = lambda key, field: [row[key][field] for row in per_seed if row.get(key)]
    summary = {}
    for lam in grid:
        key = f"lam{lam:g}"
        summary[key] = {"old_mean": float(np.mean(acc(key, "old"))),
                        "current_mean": float(np.mean(acc(key, "current")))}
    summary["permF_lam10"] = {"old_mean": float(np.mean(acc("permF_lam10", "old"))),
                              "current_mean": float(np.mean(acc("permF_lam10", "current")))}
    # preregistered verdicts
    best = max((lam for lam in grid if lam > 0),
               key=lambda lam: summary[f"lam{lam:g}"]["old_mean"])
    best_old, best_cur = summary[f"lam{best:g}"]["old_mean"], summary[f"lam{best:g}"]["current_mean"]
    base_old, base_cur = summary["lam0"]["old_mean"], summary["lam0"]["current_mean"]
    frontier_exists = (best_old - base_old > 0.05) and (base_cur - best_cur <= 0.03)
    selectivity = paired_ci(acc(f"lam{best:g}", "old"), acc("permF_lam10", "old"))
    verdict = ("存在可判定的权衡前沿,机制在该场景条件性成立" if frontier_exists
               else "全部 λ 与无保护等效或代价超限,机制在该场景被否定")
    if frontier_exists and selectivity[1] > 0.03:
        verdict += ";正常重要性优于置换重要性,选择性保护成立"
    elif frontier_exists and selectivity[2] <= 0.02:
        verdict += ";正常与置换重要性等效,选择性解释被否定(保护强度本身仍有效)"
    return {"hypothesis_id": "H010", "experiment": "E005", "seeds": list(seeds),
            "grid": list(grid), "per_seed": per_seed, "summary": summary,
            "best_lambda": best, "frontier_exists": frontier_exists,
            "selectivity_paired": selectivity,
            "preregistered_verdict": verdict,
            "scope": "gated round-3 experiment; seeds 301-320 preregistered",
            "elapsed_seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    print("import from the research loop driver; E004/E005 are invoked there")
