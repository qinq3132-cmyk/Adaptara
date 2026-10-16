"""Round-4 gated experiments: E007 (generated-reference alignment) and
E008 (premise-gated lambda sweep in a conflict-task scenario).

E007 tests H009's autonomy: can the latent alignment be computed from the
generator's OWN samples (self-calibrated, no privileged old-observation
channel) and still restore the interface? Three reference conditions on the
same frozen candidate encoder: real-reference alignment (round-3 upper bound),
generated-reference alignment (self-calibrated), random orthogonal (control).

E008 tests H010's conditional claim: construct a conflict-label task pair on a
shared feature family (parity vs magnitude on the same digits) where
recovering task B provably requires moving the parameters that carry task A's
responses; verify the premise gate (parameter-movement<->recovery correlation);
then run the frozen lambda grid + permuted-importance control inside the gate.

Preregistered (frozen in run_617edbae): seeds 401-420; minimum effects:
E007 generated-vs-random > +0.05 and generated-vs-real within ±0.02 for the
autonomy verdict; E008 old-task +0.05 with current-task cost <=0.03;
equivalence band ±0.02.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import experiments.learned.mnist_benchmarks as MB
from experiments.learned.mnist_benchmarks import (ConvVAE, load_mnist, to_gpu, vae_loss,
                                                  detect_statistic, rec_errors)
from experiments.learned.round3_experiments import paired_ci

DATA = Path(__file__).resolve().parents[2] / "artifacts/mnist_scenarios_20261012"


# ================================================================== E007
def run_e007_generated_reference(seeds=tuple(range(401, 421)), quick=False):
    per_seed = []
    t0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        X, y, Xt, yt = load_mnist()
        X = to_gpu(X); y = to_gpu(y); Xt = to_gpu(Xt); yt = to_gpu(yt)
        perm_B = rng.permutation(784)

        # frozen A-model, old head, calibration (same as round 3)
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

        # real-reference latent statistics (privileged channel, deployment-time)
        with torch.no_grad():
            ref_latents = model.enc(X[hidx])[:, :model.latent]
            ref_mean = ref_latents.mean(0)
            ref_Z = (ref_latents - ref_mean)
            ref_second = ref_Z.t() @ ref_Z / len(ref_Z)

        # A->B stream with the round-1 update flow (replay arm semantics)
        phase, window = (1500 if quick else 2500), 512
        m = ConvVAE().cuda(); m.load_state_dict(model.state_dict())
        alarm_state, cooldown, accepted = None, 0, False
        for t_block in range(0, phase * 2, window):
            mode = "B" if t_block >= phase else "A"
            idx = torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()
            batch = X[idx] if mode == "A" else X[idx][:, perm_B]
            if alarm_state is None:
                stat, _ = detect_statistic(m, batch)
                if stat > threshold and cooldown <= 0:
                    alarm_state = ("s", 1, [batch], stat)
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
                    cl, _ = vae_loss(cand, torch.cat([real, old]))
                    copt.zero_grad(); cl.backward(); copt.step()
                m = cand; accepted = True
                break
        if not accepted:
            continue

        # ---- three alignment conditions on the SAME frozen candidate
        eval_idx = torch.from_numpy(rng.integers(0, Xt.shape[0], 2048)).cuda()
        xe, ye = Xt[eval_idx], yt[eval_idx]
        with torch.no_grad():
            z_old = model.enc(xe)[:, :model.latent]
            z_new = m.enc(xe)[:, :model.latent]
            # real-reference least squares (privileged)
            fit = 1024
            W_real = torch.linalg.lstsq(z_new[:fit], z_old[:fit]).solution
            acc_real = float((head(z_new @ W_real).argmax(-1) == ye).float().mean())
            # generated-reference alignment: fit the map on the INCUMBENT generator's
            # own samples — old latents via the OLD model on generated pixels, new
            # latents via the candidate on the SAME generated pixels. No real old
            # observations are touched by this condition.
            with torch.no_grad():
                gen = model.dec(torch.randn(1024, model.latent).cuda())  # old-model samples
                z_from_old = model.enc(gen)[:, :model.latent]            # self-consistent old stats
                z_from_new = m.enc(gen)[:, :model.latent]                # candidate's view
            W_gen = torch.linalg.lstsq(z_from_new, z_from_old).solution
            acc_gen = float((head(z_new @ W_gen).argmax(-1) == ye).float().mean())
            # random orthogonal control (matched compute)
            g = torch.randn(model.latent, model.latent).cuda()
            Q, _ = torch.linalg.qr(g)
            acc_ortho = float((head(z_new @ Q).argmax(-1) == ye).float().mean())
            acc_none = float((head(z_new).argmax(-1) == ye).float().mean())
        per_seed.append({"seed": seed, "acc_real_ref": acc_real, "acc_gen_ref": acc_gen,
                         "acc_random_orthogonal": acc_ortho, "acc_none": acc_none})

    if not per_seed:
        return {"experiment": "E007", "error": "no accepted candidates"}
    acc = lambda k: [r[k] for r in per_seed]
    gen_vs_real = paired_ci(acc("acc_gen_ref"), acc("acc_real_ref"))
    gen_vs_ortho = paired_ci(acc("acc_gen_ref"), acc("acc_random_orthogonal"))
    gen_vs_none = paired_ci(acc("acc_gen_ref"), acc("acc_none"))
    autonomous = gen_vs_ortho[1] > 0.05 and abs(gen_vs_real[0]) <= 0.02
    needs_channel = gen_vs_ortho[2] <= 0.02 and gen_vs_real[0] < -0.02
    verdict = ("生成参照对齐恢复接口且与真实参照等效 → 对齐自主性成立,不需要特权旧观测通道"
               if autonomous else
               "生成参照与随机正交等效且低于真实参照 → 机制限定为需要旧参照通道"
               if needs_channel else
               "不确定:区间跨越预注册边界")
    return {"experiment": "E007", "hypothesis_id": "H009", "seeds": list(seeds),
            "n_evaluated": len(per_seed), "per_seed": per_seed,
            "means": {k: float(np.mean(acc(k))) for k in
                      ("acc_real_ref", "acc_gen_ref", "acc_random_orthogonal", "acc_none")},
            "paired": {"gen_vs_real": gen_vs_real, "gen_vs_ortho": gen_vs_ortho,
                       "gen_vs_none": gen_vs_none},
            "preregistered_verdict": verdict,
            "elapsed_seconds": round(time.time() - t0, 1)}


# ================================================================== E008
def run_e008_premise_gated_lambda(seeds=tuple(range(401, 421)), quick=False):
    """Conflict-label premise gate + lambda sweep inside the gate.

    Task A: parity labeling (digit % 2). Task B: magnitude labeling (>=5) on the
    SAME inputs — a shared 784-dim feature space where B's correct responses are
    incompatible with A's. Premise gate: verify that moving toward B actually
    requires displacing the parameters that carry A's responses (measured by a
    correlation between protected-parameter displacement and A-probe damage).
    """
    per_seed = []
    t0 = time.time()
    total = 8000 if quick else 12000
    switch_every = total // 2
    lam_grid = (0.0, 1.0, 3.0, 10.0, 30.0)
    for seed in seeds:
        row = {"seed": seed}
        for gi, lam in enumerate(lam_grid):
            torch.manual_seed(seed + gi * 1000)
            rng = np.random.default_rng(seed + gi * 1000)
            X, y, Xt, yt = load_mnist()
            X = to_gpu(X); y = to_gpu(y); Xt = to_gpu(Xt); yt = to_gpu(yt)
            # conflict labels: task A parity, task B magnitude (same inputs)
            labels_A = (y % 2).long()
            labels_B = (y >= 5).long()
            testA = (yt % 2).long(); testB = (yt >= 5).long()
            net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 2)).cuda()
            opt = torch.optim.SGD(net.parameters(), 0.008)
            batch = 32
            flat = lambda: torch.cat([w.flatten() for w in net.parameters()])
            F = torch.zeros_like(flat())
            theta_ref = None
            lam_t = 0.0
            probe_hist, window_accs, transitions = [], [], []
            recent = []   # recent A-capability probes for the snapshot plateau rule
            capability = None
            premise_corr = None
            displacement_damage = []
            for t in range(0, total, batch):
                task = "A" if t < switch_every else "B"
                idx = torch.from_numpy(rng.integers(0, X.shape[0], batch)).cuda()
                xb = X[idx]
                target = labels_A[idx] if task == "A" else labels_B[idx]
                before = flat().detach().clone()
                out = net(xb)
                loss = nn.functional.cross_entropy(out, target)
                opt.zero_grad()
                # importance estimation in stable regions (task A phase only)
                if task == "A" and theta_ref is not None and loss.item() < np.median(
                        [h for h in probe_hist[-50:]] or [loss.item()]):
                    grads = torch.autograd.grad(loss, list(net.parameters()), retain_graph=True)
                    fg = torch.cat([g.flatten() for g in grads])
                    delta = (flat() - before)
                    with torch.no_grad():
                        newF = F + (fg * delta).clamp_min(0)
                        if float(newF.max()) > 0:
                            F = newF / newF.max()
                if theta_ref is None and t > 2000 and task == "A":
                    with torch.no_grad():
                        accA = float((net(Xt[:1000]).argmax(-1) == testA[:1000]).float().mean())
                    # "recently qualified" = stable A-capability plateau (not an
                    # absolute threshold): three consecutive probes within 2 points.
                    recent.append(accA)
                    if len(recent) >= 3 and max(recent[-3:]) - min(recent[-3:]) < 0.02:
                        theta_ref = flat().detach().clone()
                        F = torch.zeros_like(theta_ref)
                        transitions.append({"step": t, "to": "snapshot", "accA": accA})
                # lambda gating by reward-drop on the CURRENT task probe
                if (t // 250) != ((t - batch) // 250) and t > 0:
                    with torch.no_grad():
                        cur = testA if task == "A" else testB
                        pr = float((net(Xt[:1000]).argmax(-1) == cur[:1000]).float().mean())
                        accA_now = float((net(Xt[:1000]).argmax(-1) == testA[:1000]).float().mean())
                    window_accs.append({"t": t, "task": task, "acc": pr, "accA": accA_now})
                    probe_hist.append(pr)
                    if theta_ref is not None:
                        disp = float((flat() - theta_ref).norm())
                        damage = max(0.0, 0.95 - accA_now)   # A-capability damage
                        displacement_damage.append((disp, damage))
                    if len(probe_hist) > 8:
                        sm, lm = np.mean(probe_hist[-2:]), np.mean(probe_hist[-8:])
                        s_val = (sm - lm) / (np.std(probe_hist[-8:]) + 1e-6)
                        if lam_t == 0.0 and s_val < -1.0 and task == "B":
                            lam_t = lam if lam > 0 else 0.0
                            transitions.append({"step": t, "to": "S1", "lam": lam_t})
                        elif lam_t > 0 and s_val > -0.3 and pr > 0.85 and task == "B":
                            lam_t = 0.0
                            theta_ref = flat().detach().clone()
                            F = torch.zeros_like(theta_ref)
                            transitions.append({"step": t, "to": "release"})
                # protection gradient (manual, no graph reuse)
                if lam_t > 0 and theta_ref is not None:
                    with torch.no_grad():
                        reg = 2 * lam_t * F * (flat() - theta_ref)
                    for w, g in zip(net.parameters(), reg.split([w_.numel() for w_ in net.parameters()])):
                        if w.grad is None:
                            w.grad = g.view_as(w).clone()
                        else:
                            w.grad += g.view_as(w)
                loss.backward()
                opt.step()
            with torch.no_grad():
                accA_final = float((net(Xt[:2000]).argmax(-1) == testA[:2000]).float().mean())
                accB_final = float((net(Xt[:2000]).argmax(-1) == testB[:2000]).float().mean())
            # premise gate: correlation between parameter displacement and A-damage
            if len(displacement_damage) >= 8:
                d = np.array([x[0] for x in displacement_damage])
                dmg = np.array([x[1] for x in displacement_damage])
                if d.std() > 1e-9 and dmg.std() > 1e-9:
                    premise_corr = float(np.corrcoef(d, dmg)[0, 1])
            row[f"lam{lam:g}"] = {"accA": accA_final, "accB": accB_final,
                                  "premise_corr": premise_corr,
                                  "transitions": len(transitions)}
        # permuted-importance control at the best-guess lambda (10)
        gi = 3
        torch.manual_seed(seed + 9000); rng = np.random.default_rng(seed + 9000)
        X, y, Xt, yt = load_mnist(); X = to_gpu(X); y = to_gpu(y); Xt = to_gpu(Xt); yt = to_gpu(yt)
        labels_A = (y % 2).long(); labels_B = (y >= 5).long()
        testA = (yt % 2).long(); testB = (yt >= 5).long()
        net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 2)).cuda()
        opt = torch.optim.SGD(net.parameters(), 0.008)
        flat = lambda: torch.cat([w.flatten() for w in net.parameters()])
        F = torch.zeros_like(flat()); theta_ref = None; lam_t = 0.0
        probe_hist, transitions = [], []
        recent_perm = []
        permuted = True
        for t in range(0, total, 32):
            task = "A" if t < total // 2 else "B"
            idx = torch.from_numpy(rng.integers(0, X.shape[0], 32)).cuda()
            target = labels_A[idx] if task == "A" else labels_B[idx]
            before = flat().detach().clone()
            out = net(X[idx]); loss = nn.functional.cross_entropy(out, target)
            opt.zero_grad()
            if task == "A" and theta_ref is not None:
                grads = torch.autograd.grad(loss, list(net.parameters()), retain_graph=True)
                fg = torch.cat([g.flatten() for g in grads])
                with torch.no_grad():
                    newF = F + (fg * (flat() - before)).clamp_min(0)
                    if float(newF.max()) > 0:
                        F = newF / newF.max()
            if theta_ref is None and t > 2000 and task == "A":
                with torch.no_grad():
                    accA = float((net(Xt[:1000]).argmax(-1) == testA[:1000]).float().mean())
                recent_perm.append(accA)
                if len(recent_perm) >= 3 and max(recent_perm[-3:]) - min(recent_perm[-3:]) < 0.02:
                    theta_ref = flat().detach().clone(); F = torch.zeros_like(theta_ref)
            if (t // 250) != ((t - 32) // 250) and t > 0 and theta_ref is not None:
                with torch.no_grad():
                    pr = float((net(Xt[:1000]).argmax(-1) ==
                                (testA if task == "A" else testB)[:1000]).float().mean())
                probe_hist.append(pr)
                if len(probe_hist) > 8:
                    sm, lm = np.mean(probe_hist[-2:]), np.mean(probe_hist[-8:])
                    s_val = (sm - lm) / (np.std(probe_hist[-8:]) + 1e-6)
                    if lam_t == 0.0 and s_val < -1.0 and task == "B":
                        lam_t = 10.0
                    elif lam_t > 0 and s_val > -0.3 and pr > 0.85 and task == "B":
                        lam_t = 0.0
                        theta_ref = flat().detach().clone(); F = torch.zeros_like(theta_ref)
            if lam_t > 0 and theta_ref is not None:
                Fp = F[torch.randperm(len(F)).cuda()]
                with torch.no_grad():
                    reg = 2 * lam_t * Fp * (flat() - theta_ref)
                for w, g in zip(net.parameters(), reg.split([w_.numel() for w_ in net.parameters()])):
                    if w.grad is None:
                        w.grad = g.view_as(w).clone()
                    else:
                        w.grad += g.view_as(w)
            loss.backward(); opt.step()
        with torch.no_grad():
            row["permF_lam10"] = {"accA": float((net(Xt[:2000]).argmax(-1) == testA[:2000]).float().mean()),
                                  "accB": float((net(Xt[:2000]).argmax(-1) == testB[:2000]).float().mean())}
        per_seed.append(row)

    acc = lambda key, field: [r[key][field] for r in per_seed if r.get(key)]
    summary = {f"lam{lam:g}": {"accA": float(np.mean(acc(f"lam{lam:g}", "accA"))),
                               "accB": float(np.mean(acc(f"lam{lam:g}", "accB")))}
               for lam in lam_grid}
    summary["permF_lam10"] = {"accA": float(np.mean(acc("permF_lam10", "accA"))),
                              "accB": float(np.mean(acc("permF_lam10", "accB")))}
    # premise-gate statistics across seeds (averaged over the swept runs)
    corrs = [v["premise_corr"] for r in per_seed for v in r.values()
             if isinstance(v, dict) and v.get("premise_corr") is not None]
    premise_mean = float(np.mean(corrs)) if corrs else None
    gate_passed = premise_mean is not None and premise_mean > 0.3
    best = max((lam for lam in lam_grid if lam > 0),
               key=lambda lam: summary[f"lam{lam:g}"]["accA"])
    best_A, base_A = summary[f"lam{best:g}"]["accA"], summary["lam0"]["accA"]
    best_B = summary[f"lam{best:g}"]["accB"]
    frontier = (best_A - base_A > 0.05) and (summary["lam0"]["accB"] - best_B <= 0.03)
    selectivity = paired_ci(acc(f"lam{best:g}", "accA"), acc("permF_lam10", "accA"))
    verdict = []
    if not gate_passed:
        verdict.append("前提门未通过:参数位移与A损伤相关性不足,保护前提在该任务族不成立——如实报告为边界")
    else:
        verdict.append("前提门通过:适应确实迫使旧关键参数移动" if frontier
                       else "前提门通过但门内 λ 扫描仍与无保护等效 → 保护主张在前提满足时也被否定(强否定)")
        if frontier and selectivity[1] > 0.03:
            verdict.append(";正常重要性优于置换,选择性成立")
    return {"experiment": "E008", "hypothesis_id": "H010", "seeds": list(seeds),
            "grid": list(lam_grid), "per_seed": per_seed, "summary": summary,
            "premise_gate": {"mean_displacement_damage_correlation": premise_mean,
                             "passed": gate_passed},
            "best_lambda": best, "frontier_exists": frontier,
            "selectivity_paired": selectivity,
            "preregistered_verdict": "".join(verdict) or "不确定",
            "elapsed_seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("E007", "E008", "ALL"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    targets = ["E007", "E008"] if args.experiment == "ALL" else [args.experiment]
    out = {}
    if "E007" in targets:
        out["E007"] = run_e007_generated_reference(quick=args.quick)
        print("E007 verdict:", out["E007"]["preregistered_verdict"])
    if "E008" in targets:
        out["E008"] = run_e008_premise_gated_lambda(quick=args.quick)
        print("E008 verdict:", out["E008"]["preregistered_verdict"])
    path = Path(__file__).resolve().parents[2] / "artifacts/research_loop_20261014/round4_experiment_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"saved {path}")
