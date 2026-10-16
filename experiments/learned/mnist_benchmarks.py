"""Domain-standard MNIST scenarios for H006/H009/H010 (single GPU, no simulators).

Scenario provenance (from the verified literature evidence chain, not invented):
- H006: task-ambiguity MNIST. The same real digit image receives two incompatible
  label rules (parity vs magnitude); the hidden mode is inferable only from the
  agent's own action/reward history. Canonical form of the cited "Same State,
  Different Task" mechanism (P_bf66cce3ab03bda8) in its simplest real-data shape.
- H009: generative replay with reconstruction-error change detection on
  Permuted-MNIST, the canonical demonstration scenario of deep generative replay
  (source quotes in EVL for CLL_bc9e91a1f8a7c49b / CLL_a3ffefcdf44f1b69).
- H010: task-free online protection on a Permuted-MNIST stream with online
  importance weights consolidated in stable loss regions, the setting of the
  cited Task-Free Continual Learning work (P_be3530b7a98636ae).

Runtime inputs follow the confirmed research constraints: only observations, the
agent's own actions and permitted reward feedback. True modes, permutations and
change times are evaluator-only and never enter the learning path.
"""
import argparse
import gzip
import json
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

DATA = Path(__file__).resolve().parents[2] / "artifacts/mnist_scenarios_20261012"
DATA.mkdir(parents=True, exist_ok=True)
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
SEEDS = {"H006": 201, "H009": 202, "H010": 203}


# ----------------------------------------------------------------- MNIST (real data)
def load_mnist():
    files = {"train_img": "train-images-idx3-ubyte.gz", "train_lbl": "train-labels-idx1-ubyte.gz",
             "test_img": "t10k-images-idx3-ubyte.gz", "test_lbl": "t10k-labels-idx1-ubyte.gz"}
    for name in files.values():
        path = DATA / name
        if not path.exists():
            urllib.request.urlretrieve(MNIST_URL + name, path)
    def images(name):
        with gzip.open(DATA / files[name], "rb") as f:
            magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), np.uint8).reshape(count, rows * cols).astype(np.float32) / 255.0
    def labels(name):
        with gzip.open(DATA / files[name], "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return images("train_img"), labels("train_lbl"), images("test_img"), labels("test_lbl")


def to_gpu(t):
    return torch.from_numpy(np.ascontiguousarray(t)).to("cuda")


# ================================================================== H006
PARITY = torch.tensor([d % 2 for d in range(10)])          # evaluator-only rule A
MAGNITUDE = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])   # evaluator-only rule B


class H006Agent(nn.Module):
    """GRU keeps a running belief from interaction history; MLP embeds the image."""
    def __init__(self, hidden=128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64))
        self.gru = nn.GRU(64 + 2 + 1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        self.mode_head = nn.Linear(hidden, 2)  # auxiliary self-supervised belief readout

    def forward(self, images, actions=None, rewards=None, clamp_belief=False):
        # images: (B, T, 784); actions/rewards: (B, T) shifted by one step.
        B, T, _ = images.shape
        z = self.embed(images)
        if actions is None:
            actions = torch.zeros(B, T, 2, device=images.device)
            rewards = torch.zeros(B, T, 1, device=images.device)
        inp = torch.cat([z, actions, rewards], -1)
        h0 = torch.zeros(1, B, self.gru.hidden_size, device=images.device)
        out, _ = self.gru(inp, h0)
        return self.head(out)  # (B, T, 2)


def h006_label(digit, mode):
    table = (PARITY if mode == 0 else MAGNITUDE).to(digit.device)
    return table[digit]


def run_h006(seed, steps=1200, episodes=256, length=32, probe_length=16, n_probe=400,
            block=4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, y, Xt, yt = load_mnist()
    X, y, Xt, yt = to_gpu(X), to_gpu(y), to_gpu(Xt), to_gpu(yt)
    agent = H006Agent().cuda()
    opt = torch.optim.Adam(agent.parameters(), 3e-4)
    n = X.shape[0]
    curve = []
    for step in range(steps):
        # Hidden mode alternates in unannounced blocks; evaluator-only. One mode per
        # training step keeps the reward-to-mode credit assignment dense.
        mode = (step // block) % 2
        idx = torch.from_numpy(rng.integers(0, n, episodes)).cuda()
        digits = y[idx]
        images = X[idx].unsqueeze(1).expand(episodes, length, 784).contiguous()
        labels = h006_label(digits, mode).unsqueeze(1).expand(episodes, length)
        # Rollout 1: sample the episode's actions WITH history (no grad).
        with torch.no_grad():
            logits0 = agent(images)
            logits0 = logits0.nan_to_num(0.0).clamp(-30, 30)
            actions0 = (logits0.argmax(-1) if step % 2 else
                        torch.argmax(logits0 + torch.rand_like(logits0).log().neg(), -1))
            rewards0 = (actions0 == labels).float()
            onehot0 = torch.nn.functional.one_hot(actions0, 2).float()
            rewards0 = rewards0.nan_to_num(0.0)
        # Rollout 2: same actions, recomputed WITH grad through the true history so
        # the gradient teaches the GRU to use action/reward feedback.
        shifted_a = torch.zeros(episodes, length, 2).cuda()
        shifted_a[:, 1:] = onehot0[:, :-1]
        shifted_r = torch.zeros(episodes, length, 1).cuda()
        shifted_r[:, 1:] = rewards0[:, :-1].unsqueeze(-1)
        logits = agent(images, shifted_a, shifted_r)
        probs = logits.softmax(-1)
        logp = logits.log_softmax(-1).gather(-1, actions0.unsqueeze(-1)).squeeze(-1)
        advantage = rewards0 - rewards0.mean(1, keepdim=True)
        # Auxiliary self-supervised belief loss: the sign of (action, reward) pairs
        # over the episode is an internal label for the hidden mode, derived only
        # from the agent's own actions and permitted reward feedback.
        gru_out, _ = agent.gru(torch.cat([agent.embed(images), shifted_a, shifted_r], -1))
        with torch.no_grad():
            signed = 2 * onehot0[:, :, 0] - 1.0              # +1 chose action0, -1 chose action1
            agree = (rewards0 * signed).sum(1)
            mode_target = (agree > 0).long()
        aux = torch.nn.functional.cross_entropy(agent.mode_head(gru_out[:, -1]), mode_target)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()
        loss = -(logp * advantage.detach()).mean() + 0.5 * aux - 0.01 * entropy
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            curve.append({"step": step, "reward": float(rewards0.mean())})
    # ---------------- frozen-parameter probe (4 arms, evaluator-built histories)
    agent.eval()
    test_idx = torch.from_numpy(rng.integers(0, Xt.shape[0], n_probe)).cuda()
    t_images, t_digits = Xt[test_idx], yt[test_idx]
    forced = torch.tensor([t % 2 for t in range(probe_length)]).cuda()  # alternating actions
    labels_A = h006_label(t_digits, 0).unsqueeze(1).expand(n_probe, probe_length)
    labels_B = h006_label(t_digits, 1).unsqueeze(1).expand(n_probe, probe_length)
    rewards_A = (forced == labels_A).float()
    rewards_B = (forced == labels_B).float()
    perm = torch.stack([torch.randperm(probe_length) for _ in range(n_probe)]).cuda()

    def probe(rewards, arm):
        T = probe_length + 1
        imgs = t_images.unsqueeze(1).expand(n_probe, probe_length + 1, 784).clone()
        acts = torch.zeros(n_probe, T, 2).cuda()
        acts[:, torch.arange(1, T), forced] = 1  # one-hot of the forced action at each step
        r = torch.zeros(n_probe, T, 1).cuda()
        if arm == "shuffled":
            r[:, 1:, 0] = rewards.gather(1, perm)
        elif arm == "inverted":
            r[:, 1:, 0] = 1 - rewards
        else:
            r[:, 1:, 0] = rewards
        with torch.no_grad():
            if arm == "clamped":  # belief clamped: every step re-starts from the same h0
                z = agent.embed(imgs)
                inp = torch.cat([z, acts, r], -1)
                h0 = torch.zeros(1, n_probe, agent.gru.hidden_size).cuda()
                out, _ = agent.gru(inp[:, :1], h0)   # hidden state never advances
                logits = agent.head(out).expand(n_probe, T, 2)
            else:
                logits = agent(imgs, acts, r)
        return logits[:, -1].softmax(-1)  # p(a|image, history)

    results = {}
    for arm, rewards in (("normal_A", rewards_A), ("normal_B", rewards_B),
                         ("shuffled", rewards_A), ("inverted", rewards_A), ("clamped", rewards_B)):
        results[arm] = probe(rewards, arm.split("_")[0] if "normal" not in arm else "normal")
    sep = (results["normal_A"] - results["normal_B"]).abs().mean()
    # separation with destroyed information must shrink for the mechanism to hold
    sep_shuf = (results["normal_A"][:64] - results["shuffled"][:64]).abs().mean()
    sep_inv_dir = ((results["inverted"] - results["normal_B"]).abs().mean()
                   - (results["inverted"] - results["normal_A"]).abs().mean())
    conflict_digits = torch.tensor([1, 3, 6, 8]).cuda()
    mask = torch.isin(t_digits, conflict_digits)
    err_A = (results["normal_A"].argmax(-1) != h006_label(t_digits, 0)).float()[mask].mean()
    err_B = (results["normal_B"].argmax(-1) != h006_label(t_digits, 1)).float()[mask].mean()
    return {"hypothesis_id": "H006", "seed": seed, "training_curve": curve,
            "metrics": {"same_observation_separation": float(sep),
                        "separation_vs_shuffled_history": float(sep_shuf),
                        "inversion_direction_effect": float(sep_inv_dir),
                        "conflict_error_rate_modeA": float(err_A),
                        "conflict_error_rate_modeB": float(err_B)},
            "information_contract": ["real MNIST images", "own action history", "scalar reward feedback"],
            "evaluator_only": ["hidden mode schedule", "true labels", "probe construction rules"]}


# ================================================================== H009
class ConvVAE(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 2 * latent))
        self.dec = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())
        self.latent = latent

    def forward(self, x):
        h = self.enc(x)
        mu, lv = h[:, :self.latent], h[:, self.latent:]
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return self.dec(z), mu, lv


def vae_loss(m, x):
    rec, mu, lv = m(x)
    mse = ((rec - x) ** 2).sum(1).mean()
    kld = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
    return mse + kld, mse


def rec_errors(m, x):
    with torch.no_grad():
        rec, _, _ = m(x)
        return ((rec - x) ** 2).mean(1)


def detect_statistic(m, real_batch, prior_samples=512):
    """Reconstruction error of current real observations vs the calibrated
    no-change level (the cited statistical-test mechanism): returns (statistic,
    mean_real_error). The statistic contrasts current-block error against the
    model's error on its own generated states, which is drift-free under
    no-change and inflates after an observation-encoding change."""
    with torch.no_grad():
        z = torch.randn(prior_samples, m.latent).cuda()
        gen = m.dec(z)
        e_gen = rec_errors(m, gen)
    e_real = rec_errors(m, real_batch)
    pooled = float(e_real.std() ** 2 + e_gen.std() ** 2) ** 0.5 + 1e-8
    return float((e_real.mean() - e_gen.mean()) / pooled), float(e_real.mean())


def run_h009(seed, pretrain=1500, phase=3000, window=512, cand_steps=300, replay_batch=256):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, y, Xt, yt = load_mnist()
    X = to_gpu(X); y = to_gpu(y); Xt = to_gpu(Xt); yt = to_gpu(yt)
    perm_B = rng.permutation(784)
    perm_C = rng.permutation(784)  # plus rotation for a second, distinct encoding change
    rot_C = True

    model = ConvVAE().cuda()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(pretrain):  # environment A: original MNIST
        idx = torch.from_numpy(rng.integers(0, X.shape[0], replay_batch)).cuda()
        loss, _ = vae_loss(model, X[idx]); opt.zero_grad(); loss.backward(); opt.step()
    # frozen old-task head: linear classifier on A features (old interface)
    head = nn.Linear(model.latent, 10).cuda()
    with torch.no_grad():
        idx = torch.from_numpy(rng.integers(0, X.shape[0], 4096)).cuda()
    hopt = torch.optim.Adam(head.parameters(), 1e-2)
    for _ in range(300):
        hloss = nn.functional.cross_entropy(head(model.enc(X[idx])[:, :model.latent]), y[idx])
        hopt.zero_grad(); hloss.backward(); hopt.step()
    head.eval()

    def old_task_accuracy(m):
        """Latest-encoder + frozen-old-policy interface, per the design. The frozen
        head reads the latest encoder's latents on OLD observations; no training
        happens here. Separately report the representation metric (old MSE)."""
        with torch.no_grad():
            accs = []
            for i in range(0, 4096, 1024):
                xb = Xt[i:i + 1024]
                accs.append(float((head(m.enc(xb)[:, :m.latent]).argmax(-1) == yt[i:i + 1024].cuda()).float().mean()))
            return float(np.mean(accs))

    def old_linear_probe_accuracy(m):
        """Interface re-adaptation diagnostic (design: retrained head is a separate
        control, isolating representation quality from interface drift): a fresh
        linear head trained on the latest encoder's A-latents, tested on old data."""
        probe_head = nn.Linear(m.latent, 10).cuda()
        popt = torch.optim.Adam(probe_head.parameters(), 1e-2)
        pidx = torch.from_numpy(rng.integers(0, X.shape[0], 2048)).cuda()
        for _ in range(200):
            ploss = nn.functional.cross_entropy(probe_head(m.enc(X[pidx])[:, :m.latent]), y[pidx])
            popt.zero_grad(); ploss.backward(); popt.step()
        with torch.no_grad():
            accs = []
            for i in range(0, 4096, 1024):
                xb = Xt[i:i + 1024]
                accs.append(float((probe_head(m.enc(xb)[:, :m.latent]).argmax(-1) == yt[i:i + 1024].cuda()).float().mean()))
            return float(np.mean(accs))

    # calibrate the detection statistic on held-out A (no-change) blocks; the
    # pre-change reconstruction-error level is the acceptance target after updates
    calib = []
    calib_errors = []
    for _ in range(40):
        idx = torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()
        s, e = detect_statistic(model, X[idx])
        calib.append(s)
        calib_errors.append(e)
    threshold = float(np.quantile(calib, 0.99))
    # Settled target widened by recalibration uncertainty (the design's provisional
    # threshold rule): q99 error level * 1.5 until independent re-calibration.
    calib_error_level = float(np.quantile(calib_errors, 0.99)) * 1.5
    # independent held-out acceptance batches (never used for candidate training)
    accept_batches = [X[torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()]
                      for _ in range(4)]

    def render(mode, idx):
        x = X[idx]
        if mode == "A":
            return x
        if mode == "C" and rot_C:
            x = x.view(-1, 28, 28).rot90(1, (1, 2)).reshape(-1, 784)
            return x[:, perm_C]
        return x[:, perm_B]

    def adapt(m, buffer, use_replay):
        cand = ConvVAE().cuda()
        cand.load_state_dict(m.state_dict())
        copt = torch.optim.Adam(cand.parameters(), 1e-3)
        for _ in range(cand_steps):
            real = buffer[torch.from_numpy(rng.integers(0, len(buffer), replay_batch)).cuda()]
            if use_replay == "replay":
                with torch.no_grad():
                    old = m.dec(torch.randn(replay_batch, m.latent).cuda())
                batch = torch.cat([real, old])
            elif use_replay == "repeat":   # equal compute, no old-environment information
                batch = torch.cat([real, real[torch.from_numpy(rng.integers(0, len(real), replay_batch)).cuda()]])
            else:                           # no_replay
                batch = real
            loss, _ = vae_loss(cand, batch); copt.zero_grad(); loss.backward(); copt.step()
        return cand

    def arm_flow(arm):
        m = ConvVAE().cuda(); m.load_state_dict(model.state_dict())
        opt2 = torch.optim.Adam(m.parameters(), 1e-3)
        del opt2  # incumbent updates happen through candidate promotion only
        stream, mode, t = [], "A", 0
        events = []
        fixed_points = {phase + phase // 2, phase * 3 + phase // 2}  # fixed schedule times
        checkpoints, delay, false_alarms, unresolved = {}, None, 0, 0
        cooldown = 0
        alarm_state = None  # None | ('sustain', n, blocks, first_stat, step) | ('accept', cand, stat, step)
        sustain_needed = 2   # consecutive alarming blocks before a candidate is trained
        def stream_slice():
            idx = torch.from_numpy(rng.integers(0, X.shape[0], window)).cuda()
            return render(mode, idx)
        schedule_used = 0
        for t_block in range(0, phase * 4, window):
            if t_block >= phase > t_block - window: mode = "B"; checkpoints["B_start"] = t_block
            elif t_block >= phase * 2 > t_block - window: mode = "A"; checkpoints["A_return"] = t_block
            elif t_block >= phase * 3 > t_block - window: mode = "C"; checkpoints["C_start"] = t_block
            batch = stream_slice()
            stream.append(batch)
            changed = mode != "A"
            if arm == "fixed" and t_block in fixed_points and schedule_used < 2:
                schedule_used += 1
                m = adapt(m, torch.cat(stream[-4:]), "replay")
                events.append({"step": t_block, "kind": "scheduled_update"})
                continue
            # ---- sustained-alarm protocol (design: persistent crossing -> update
            #      blocks -> forward acceptance block) ----
            if alarm_state is not None and alarm_state[0] == "sustain":
                _, count, blocks, first_stat, trig = alarm_state
                stat, _ = detect_statistic(m, batch)
                if stat <= threshold:
                    # transient excursion: the alarm did not persist; no update happens
                    if not changed:
                        false_alarms += 1
                    events.append({"step": trig, "kind": "transient_excursion",
                                   "ended_at": t_block})
                    alarm_state = None
                    cooldown = 2
                    continue
                blocks.append(batch)
                count += 1
                alarm_state = ("sustain", count, blocks, first_stat, trig)
                if count < sustain_needed + 1:
                    continue
                if delay is None and changed:
                    delay = trig - checkpoints.get("B_start" if mode == "B" else "C_start", trig)
                cand = adapt(m, torch.cat(blocks), arm if arm in ("replay", "no_replay", "repeat") else "replay")
                alarm_state = ("accept", cand, first_stat, trig)
                continue
            if alarm_state is not None and alarm_state[0] == "accept":
                _, cand, stat, trig = alarm_state
                # Forward acceptance on THIS fresh block, which never participated in
                # candidate training. Design criterion: the candidate's reconstruction
                # error on new observations must return to the pre-change calibrated
                # level, and it must beat the incumbent on the same acceptance block.
                with torch.no_grad():
                    _, e_after = detect_statistic(cand, batch)
                    _, e_incumbent = detect_statistic(m, batch)
                settled = e_after <= calib_error_level and e_after < e_incumbent
                if settled:
                    m = cand
                    cooldown = 2
                    events.append({"step": trig, "kind": "accepted_update",
                                   "accepted_at": t_block, "error_after": e_after,
                                   "incumbent_error": e_incumbent})
                else:
                    unresolved += 1
                    events.append({"step": trig, "kind": "rejected_candidate",
                                   "after_error": e_after, "incumbent_error": e_incumbent})
                alarm_state = None
                continue
            stat, _ = detect_statistic(m, batch)
            alarm = stat > threshold
            if alarm and cooldown <= 0:
                events.append({"step": t_block, "kind": "alarm", "mode": mode, "stat": stat,
                               "changed": changed})
                if not changed:
                    false_alarms += 1
                alarm_state = ("sustain", 1, [batch], stat, t_block)
            elif not alarm:
                cooldown -= 1
        old_acc = old_task_accuracy(m)
        _, old_mse = detect_statistic(m, Xt[:1024])
        probe_acc = old_linear_probe_accuracy(m)
        return {"arm": arm, "old_task_accuracy": old_acc, "old_rec_mse": old_mse,
                "old_linear_probe_accuracy": probe_acc,
                "detection_delay_blocks": delay, "false_alarms": false_alarms,
                "rejected_candidates": unresolved, "events": events}

    arms = {arm: arm_flow(arm) for arm in ("replay", "no_replay", "repeat", "fixed")}
    full = arms["replay"]
    return {"hypothesis_id": "H009", "seed": seed,
            "metrics": {arm: {k: v for k, v in r.items() if k != "events"} for arm, r in arms.items()},
            "replay_minus_no_replay_old_accuracy": full["old_task_accuracy"] - arms["no_replay"]["old_task_accuracy"],
            "detection_threshold": threshold, "calibration": {"n_blocks": 40, "q99": threshold},
            "arms": arms,
            "information_contract": ["real MNIST images", "internally generated samples", "fixed internal buffer"],
            "evaluator_only": ["permutation matrices", "change times", "old-task labels"]}


# ================================================================== H010
def run_h010(seed, total=12500, switch_every=2500, batch=32, hidden=256, lr=0.008,
             lambda_override=None, seed_offset=0):
    """Slower online learner: permuted-MNIST recovery crosses the protection-
    relevant timescale so the alarm/release state machine is observable.
    lambda_override pins the protection strength (E005 sweep); seed_offset
    varies the stream draw per lambda so each grid point is an independent run."""
    torch.manual_seed(seed + seed_offset)
    rng = np.random.default_rng(seed + seed_offset)
    X, y, Xt, yt = load_mnist()
    X, y = to_gpu(X), to_gpu(y)
    Xt = to_gpu(Xt); yt = to_gpu(yt)
    perms = [rng.permutation(784) for _ in range(5)]  # evaluator-only change schedule
    n_perms = max(total // switch_every, 1)
    test_by_perm = []
    for p in range(n_perms):
        idx = torch.from_numpy(rng.integers(0, Xt.shape[0], 2000)).cuda()
        test_by_perm.append((Xt[idx][:, perms[p]], yt[idx]))

    def make_net():
        return nn.Sequential(nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 10)).cuda()

    def probe(net, p):
        xb, yb = test_by_perm[p]
        with torch.no_grad():
            return float((net(xb).argmax(-1) == yb).float().mean())

    def run_arm(arm):
        net = make_net()
        opt = torch.optim.SGD(net.parameters(), lr)
        flat = lambda: torch.cat([w.flatten() for w in net.parameters()])
        F = torch.zeros_like(flat())
        theta_ref = None
        lam = 0.0
        state, state_t, s_hist, trace, transitions = "S0", 0, [], [], []
        ma_s, ma_l = [], []
        window_accs, old_retentions = [], []
        probe_history = []   # evaluator-scheduled capability probe accuracies
        t0 = time.process_time()
        ptr = 0
        for t in range(0, total, batch):
            p = min(t // switch_every, n_perms - 1)
            if theta_ref is None or (state == "S0" and len(ma_l) > 40 and abs(np.mean(ma_s[-5:]) - np.mean(ma_l[-40:])) < 0.02
                                     and probe(net, p) >= 0.55):
                theta_ref = torch.cat([w.flatten() for w in net.parameters()]).clone()
                F = torch.zeros_like(theta_ref)
                transitions.append({"step": t, "to": "snapshot", "state": state})
            idx = torch.from_numpy(rng.integers(0, X.shape[0], batch)).cuda()
            xb, yb = X[idx][:, perms[p]], y[idx]
            before = torch.cat([w.flatten().detach().clone() for w in net.parameters()])
            out = net(xb)
            loss = nn.functional.cross_entropy(out, yb)
            acc = float((out.argmax(-1) == yb).float().mean())
            # online SI-style importance: consolidate when the loss is in a stable
            # region; gradients are taken with autograd.grad so the graph is intact.
            if theta_ref is not None and loss.item() < float(np.median([l for l in s_hist[-50:]] or [loss.item()])):
                grads = torch.autograd.grad(loss, list(net.parameters()), retain_graph=True)
                flat_g = torch.cat([g.flatten() for g in grads])
                delta = (flat() - before).detach()
                with torch.no_grad():
                    newF = F + (flat_g * delta).clamp_min(0)
                    if float(newF.max()) > 0:
                        F = newF / newF.max()
            if arm == "permuted_F" and theta_ref is not None:
                Fp = F[torch.randperm(len(F)).cuda()]
            else:
                Fp = F
            opt.zero_grad()
            if lam > 0 and theta_ref is not None:
                # manual per-parameter regularization gradients (no graph reuse)
                with torch.no_grad():
                    cur = flat()
                    reg_grad = 2 * lam * Fp * (cur - theta_ref)
                for w, g in zip(net.parameters(), reg_grad.split([w_.numel() for w_ in net.parameters()])):
                    if w.grad is None:
                        w.grad = g.view_as(w).clone()
                    else:
                        w.grad += g.view_as(w)
            loss.backward()
            opt.step()
            s_hist.append(float(loss.item()))
            ma_s.append(acc); ma_l.append(acc)
            if (t // 250) != ((t - batch) // 250) and t > 0:
                pr = probe(net, p)
                window_accs.append({"t": t, "perm": p, "acc": pr})
                probe_history.append(pr)
            if len(probe_history) > 8:
                sm = np.mean(probe_history[-2:])       # short window: recent probes
                lm = np.mean(probe_history[-8:])       # long window: recent period
                s_val = (sm - lm) / (np.std(probe_history[-8:]) + 1e-6)
            else:
                s_val = 0.0
            # --- state machine (evaluator-visible transitions; permitted capability probe)
            capability = probe(net, p)
            LAMBDA = lambda_override if lambda_override is not None else 10.0
            if arm == "fixed":  # matched-clock schedule, no signal use
                lam = LAMBDA if (t // switch_every) % 2 == 1 else 0.0
            elif arm == "none":
                lam = 0.0
            else:
                if state == "S0" and s_val < -1.0:   # sustained drop in task reward
                    state, state_t = "S1", 0
                    lam = LAMBDA
                    transitions.append({"step": t, "to": "S1", "s": s_val})
                elif state == "S1":
                    state_t += batch
                    recovered = s_val > -0.3            # reward returned to band
                    capable = capability >= 0.75         # independent capability gate
                    if recovered and capable:
                        state, state_t = "S2", 0
                        transitions.append({"step": t, "to": "S2", "s": s_val})
                    elif recovered and not capable:
                        state, state_t = "ST", 0        # reward back but capability unproven: wait
                        transitions.append({"step": t, "to": "ST", "s": s_val})
                    elif state_t > 2000:  # bounded failure exit, not silent stall
                        state, lam = "SF", 0.0
                        transitions.append({"step": t, "to": "SF"})
                elif state == "ST":
                    state_t += batch
                    if capability >= 0.75 and s_val > -0.3:
                        state, state_t = "S2", 0
                        transitions.append({"step": t, "to": "S2_from_ST"})
                    elif state_t > 1200:
                        state, lam = "SF_partial", 0.0
                        transitions.append({"step": t, "to": "SF_partial"})
                elif state == "S2":
                    state_t += batch
                    if state_t > 400:  # re-estimate reference at release, then decay
                        theta_ref = torch.cat([w.flatten() for w in net.parameters()]).clone()
                        F = torch.zeros_like(theta_ref)
                        state, lam = "S0", 0.0
                        transitions.append({"step": t, "to": "S0_release"})
                if arm == "continuous":
                    lam = LAMBDA
        old_retention = probe(net, 0)
        cur_acc = probe(net, min(p, n_perms - 1))
        return {"arm": arm, "final_old_perm_accuracy": old_retention,
                "final_current_perm_accuracy": cur_acc,
                "new_task_acc_by_phase": window_accs[-len(window_accs)//2:],
                "mean_current_acc": float(np.mean([w["acc"] for w in window_accs])),
                "state_transitions": transitions,
                "cpu_seconds": time.process_time() - t0}

    arms = {arm: run_arm(arm) for arm in ("full", "none", "fixed", "permuted_F", "continuous")}
    return {"hypothesis_id": "H010", "seed": seed,
            "metrics": {arm: {k: v for k, v in r.items() if k != "new_task_acc_by_phase"} for arm, r in arms.items()},
            "arms": arms,
            "information_contract": ["real MNIST pixels (permuted by evaluator)", "own batch accuracy/loss"],
            "evaluator_only": ["permutation schedule", "switch times", "held-out probes"]}


# ----------------------------------------------------------------- runner
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("H006", "H009", "H010", "ALL"))
    parser.add_argument("--seed", type=int, default=None, help="single-seed override")
    parser.add_argument("--seeds", type=int, default=1, help="number of replicate seeds")
    parser.add_argument("--quick", action="store_true", help="smoke settings")
    args = parser.parse_args()
    targets = ["H006", "H009", "H010"] if args.experiment == "ALL" else [args.experiment]
    device_name = torch.cuda.get_device_name(0)
    for target in targets:
        base_seed = args.seed or SEEDS[target]
        t0 = time.time()
        if target == "H006":
            if args.seeds > 1:
                outs = [run_h006(base_seed + i, steps=120 if args.quick else 1200) for i in range(args.seeds)]
            else:
                outs = [run_h006(base_seed, steps=120 if args.quick else 1200)]
        elif target == "H009":
            if args.seeds > 1:
                outs = [run_h009(base_seed + i, pretrain=200 if args.quick else 1200,
                          phase=1000 if args.quick else 2500, cand_steps=40 if args.quick else 200)
                        for i in range(args.seeds)]
            else:
                outs = [run_h009(base_seed, pretrain=200 if args.quick else 1200,
                          phase=1000 if args.quick else 2500, cand_steps=40 if args.quick else 200)]
        else:
            if args.seeds > 1:
                outs = [run_h010(base_seed + i, total=4000 if args.quick else 12000) for i in range(args.seeds)]
            else:
                outs = [run_h010(base_seed, total=4000 if args.quick else 12000)]
        # multi-seed paired summary
        def mean_metrics(target_key, runs):
            first = runs[0]["metrics"]
            if target_key == "H006":
                return {k: float(np.mean([r["metrics"][k] for r in runs])) for k in first}
            # nested arm dicts: average numeric leaves per arm
            mean = {}
            for arm in first:
                row = {}
                for k in first[arm]:
                    values = [r["metrics"][arm][k] for r in runs]
                    if all(isinstance(v, (int, float)) for v in values):
                        row[k] = float(np.mean(values))
                mean[arm] = row
            return mean

        out = outs[0] if len(outs) == 1 else {
            "hypothesis_id": target, "seeds": [o["seed"] for o in outs],
            "per_seed_metrics": [o.get("metrics", o) for o in outs],
            "mean_metrics": mean_metrics(target, outs),
            "replay_minus_no_replay_old_accuracy": (float(np.mean([o["replay_minus_no_replay_old_accuracy"] for o in outs]))
                                                     if target == "H009" else None),
            "information_contract": outs[0].get("information_contract"), "evaluator_only": outs[0].get("evaluator_only"),
        }
        out["gpu"] = device_name
        out["elapsed_seconds"] = round(time.time() - t0, 1)
        out["n_seeds"] = len(outs)
        out["scope"] = ("smoke run" if args.quick else
                        f"exploratory single-GPU run, {len(outs)} seed(s); not a preregistered primary experiment")
        path = DATA / f"{'smoke_' if args.quick else ''}result_{target}_{base_seed}{'_x' + str(len(outs)) if len(outs) > 1 else ''}.json"
        path.write_text(json.dumps(out, indent=1, default=float))
        print(f"[{target}] saved {path} ({out['elapsed_seconds']}s on {device_name}, {len(outs)} seed(s))")
        metrics = out.get("mean_metrics") or out.get("metrics") or {}
        if target == "H006":
            print(f"[{target}] separation:", metrics.get("same_observation_separation"))
        elif target == "H009":
            delta = (out.get("replay_minus_no_replay_old_accuracy")
                     if isinstance(out.get("replay_minus_no_replay_old_accuracy"), (int, float))
                     else (out["metrics"]["replay"]["old_task_accuracy"]
                           - out["metrics"]["no_replay"]["old_task_accuracy"]))
            print(f"[{target}] replay - no_replay old-task accuracy: {delta:+.4f}")
            for arm in ("replay", "no_replay"):
                row = metrics[arm]
                delay = row.get("detection_delay_blocks", "n/a")
                print(f"  {arm}: old_acc={row['old_task_accuracy']:.4f} delay={delay}"
                      f" false_alarms={row['false_alarms']:.2f} rejected={row['rejected_candidates']:.2f}")
        else:
            for arm in ("full", "none", "fixed", "permuted_F", "continuous"):
                row = metrics[arm]
                print(f"  {arm}: old_perm_acc={row['final_old_perm_accuracy']:.4f}"
                      f" current_perm_acc={row.get('final_current_perm_accuracy', 0):.4f}"
                      f" mean_current={row['mean_current_acc']:.4f}")


if __name__ == "__main__":
    main()
