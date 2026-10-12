"""Learned H009 experiment. Private simulator state is never a runtime label.

This is a finite local experiment, not a universal continual-learning benchmark.
Both candidate training blocks precede a disjoint acceptance and calibration block.
The generator is fitted to the encoder's latent samples and uses the learned decoder.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .history import weight_digest


@dataclass(frozen=True)
class ReplayConfig:
    hidden: int = 24
    initialization_updates: int = 300
    candidate_updates: int = 80
    batch: int = 128
    learning_rate: float = .003
    block: int = 128
    calibration_blocks: int = 256
    null_blocks: int = 256
    streams: int = 4
    evaluation_samples: int = 1024
    max_retries: int = 2
    calibration_quantile: float = .999
    improvement_fraction: float = .02
    candidate_consistency: float = .30
    quality_ratio: float = 2.
    modes: tuple = ('A', 'B', 'A', 'C')
    private_phase_lengths: tuple = (2048, 32768, 32768, 32768)
    arms: tuple = ('replay', 'no_replay', 'current_repeat', 'periodic',
                   'critical_removed', 'noncritical_removed')


class Autoencoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.decoder = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 8))

    def forward(self, values):
        return self.decoder(self.encoder(values))


class SensoryStream:
    """Calls expose only fresh observations; labels are returned by scorer separately."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def draw(self, count, mode='A'):
        cluster = self.rng.integers(0, 4, count)
        centers = np.asarray([[-.30, -.8], [.30, -.8], [-1.2, .8], [1.2, .8]])
        physical = centers[cluster] + self.rng.normal(0, .08, (count, 2))
        q, p = physical.T
        observation = np.stack([q, p, q*p, q*q*.4, p*p*.4, np.sin(q), np.cos(p), q-p], axis=1)
        if mode == 'B':
            observation = observation[:, [1, 0, 3, 2, 6, 7, 4, 5]] * np.array([1., -1., 1., 1., .8, 1.1, 1., 1.])
            observation += np.array([.5, -.4, .2, .1, .3, -.2, .1, .2])
        elif mode == 'C':
            observation = observation[:, [6, 4, 0, 7, 1, 5, 3, 2]] * np.array([-1., 1., 1.2, .8, -1., 1., 1., 1.])
            observation += np.array([-.4, .3, .5, -.2, .4, .2, -.1, .1])
        elif mode != 'A':
            raise ValueError('Unknown private renderer')
        return observation.astype(np.float32), (q > 0).astype(np.int64), cluster


class SensoryTimeline:
    """Counter-seeded online stream: identical observation at any paired clock index.

    These schedule fields belong to the environment driver, never to the learner.
    Candidate data collection can cross a change, so no stability oracle is implied.
    """
    def __init__(self, seed, config):
        self.seed, self.config, self.position = seed, config, 0
        self.boundaries = np.cumsum(config.private_phase_lengths)

    @property
    def remaining(self):
        return int(self.boundaries[-1]-self.position)

    def draw(self, count):
        if count > self.remaining or count % self.config.block:
            raise ValueError('Draw whole chronological blocks within stream budget')
        chunks = []
        for start in range(self.position,self.position+count,self.config.block):
            mode = self.config.modes[int(np.searchsorted(self.boundaries,start,side='right'))]
            chunks.append(SensoryStream(self.seed+start//self.config.block).draw(self.config.block,mode)[0])
        self.position += count
        return np.concatenate(chunks)


def fit_generator(model, observations):
    """Four latent Gaussian components estimated without scene labels or masks."""
    with torch.no_grad():
        latent = model.encoder(torch.from_numpy(observations)).numpy()
    centers = [latent[0]]
    for _ in range(3):
        distances = np.min(((latent[:, None]-np.asarray(centers)[None])**2).sum(-1), axis=1)
        centers.append(latent[distances.argmax()])
    centers = np.asarray(centers)
    for _ in range(30):
        assignment = ((latent[:, None]-centers[None])**2).sum(-1).argmin(-1)
        centers = np.stack([latent[assignment == j].mean(0) if (assignment == j).any() else centers[j] for j in range(4)])
    covariance = np.stack([np.cov(latent[assignment == j].T)+np.eye(2)*1e-4
                           if (assignment == j).sum() > 2 else np.eye(2)*1e-3 for j in range(4)])
    return {'centers': centers, 'covariance': covariance,
            'probabilities': np.bincount(assignment, minlength=4)/len(latent)}


def generate(model, generator, rng, count, excluded=None):
    probabilities = generator['probabilities'].copy()
    if excluded is not None:
        probabilities[excluded] = 0
        probabilities /= probabilities.sum()
    components = rng.choice(4, count, p=probabilities)
    z = np.empty((count, 2), np.float32)
    for j in range(4):
        mask = components == j
        z[mask] = rng.multivariate_normal(generator['centers'][j], generator['covariance'][j], mask.sum())
    with torch.no_grad():
        values = model.decoder(torch.from_numpy(z)).numpy()
    return values, components


def reconstruction_errors(model, values):
    with torch.no_grad():
        x = torch.from_numpy(values)
        return ((model(x)-x)**2).mean(-1).numpy()


def train_candidate(incumbent, current, replay, config, rng):
    candidate = deepcopy(incumbent)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=config.learning_rate)
    pool = np.concatenate([current, replay])
    for _ in range(config.candidate_updates):
        batch = torch.from_numpy(pool[rng.integers(len(pool), size=config.batch)])
        loss = ((candidate(batch)-batch)**2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return candidate


def cdf_difference(model, real, generated, edges):
    real_error, generated_error = reconstruction_errors(model, real), reconstruction_errors(model, generated)
    return (real_error[:, None] <= edges).mean(0) - (generated_error[:, None] <= edges).mean(0)


def calibrate(model, generator, observations, config, rng):
    """Use every independent supplied block, never select presumed 'nonalarm' blocks."""
    errors = reconstruction_errors(model, observations[:config.block*4])
    edges = np.quantile(errors, np.linspace(.05, .95, 19))
    generated, _ = generate(model, generator, rng, len(observations))
    differences = []
    for start in range(0, len(observations)-config.block+1, config.block):
        differences.append(cdf_difference(model, observations[start:start+config.block],
                                         generated[start:start+config.block], edges))
    differences = np.asarray(differences)
    bias = differences.mean(0)
    statistics = np.abs(differences-bias).max(1)
    threshold = float(np.quantile(statistics, config.calibration_quantile)) + .02
    return {'edges': edges, 'bias': bias, 'threshold': threshold,
            'blocks': len(statistics), 'maximum_observed': float(statistics.max()),
            'statistic': 'maximum bias-corrected CDF difference on frozen quantile grid'}


def detect(model, generator, calibration, observations, rng):
    generated, _ = generate(model, generator, rng, len(observations))
    value = float(np.abs(cdf_difference(model, observations, generated, calibration['edges']) - calibration['bias']).max())
    return value, value > calibration['threshold']


def evaluate_interface(model, frozen_policy, initial, observations, labels, cluster):
    with torch.no_grad():
        x = torch.from_numpy(observations)
        predicted = frozen_policy(model.encoder(x)).argmax(-1).numpy()
        baseline = frozen_policy(initial.encoder(x)).argmax(-1).numpy()
    errors, base_errors = reconstruction_errors(model, observations), reconstruction_errors(initial, observations)
    critical = cluster < 2
    return {'critical_accuracy': float((predicted[critical] == labels[critical]).mean()),
            'noncritical_accuracy': float((predicted[~critical] == labels[~critical]).mean()),
            'overall_accuracy': float((predicted == labels).mean()),
            'frozen_original_accuracy': float((baseline == labels).mean()),
            'critical_reconstruction_ratio': float(errors[critical].mean()/max(base_errors[critical].mean(), 1e-8)),
            'noncritical_reconstruction_ratio': float(errors[~critical].mean()/max(base_errors[~critical].mean(), 1e-8))}


def initialize(seed, config):
    torch.manual_seed(seed)
    initial = Autoencoder(config.hidden)
    rng = np.random.default_rng(seed+10000)
    data, labels, _ = SensoryStream(seed+11000).draw(4096)
    optimizer = torch.optim.Adam(initial.parameters(), lr=config.learning_rate)
    for _ in range(config.initialization_updates):
        x = torch.from_numpy(data[rng.integers(len(data), size=config.batch)])
        loss = ((initial(x)-x)**2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # Supervised initialization is explicitly permitted; these labels never exist in runtime pools.
    policy = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.03)
    with torch.no_grad(): z = initial.encoder(torch.from_numpy(data)).detach()
    for _ in range(200):
        loss = nn.functional.cross_entropy(policy(z), torch.from_numpy(labels))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    for parameter in policy.parameters(): parameter.requires_grad_(False)
    generator = fit_generator(initial, data)
    # Use learned policy margin to select a critical and matched-mass noncritical generator component.
    with torch.no_grad():
        margins = policy(torch.from_numpy(generator['centers'].astype(np.float32))).softmax(-1).max(-1).values.numpy()
    critical_component = int(margins.argmin())
    remaining = [j for j in range(4) if j != critical_component]
    noncritical_component = min(remaining, key=lambda j: abs(generator['probabilities'][j] - generator['probabilities'][critical_component])-.01*margins[j])
    calibration_data = SensoryStream(seed+12000).draw(config.block*config.calibration_blocks)[0]
    calibration = calibrate(initial, generator, calibration_data, config, rng)
    # A complete stationary shadow update, including held-out acceptance and recalibration.
    shadow_blocks = [SensoryStream(seed+13000+j).draw(config.block*4)[0] for j in range(4)]
    replay = generate(initial, generator, rng, config.block*4)[0]
    shadow = [train_candidate(initial, block, replay, config, rng) for block in shadow_blocks[:2]]
    shadow_acceptance = candidate_checks(shadow, initial, generator, shadow_blocks[2], calibration, config, rng)
    shadow_generator = fit_generator(shadow[0], np.concatenate([shadow_blocks[0], replay]))
    shadow_calibration = calibrate(shadow[0], shadow_generator, calibration_data.copy(), config, rng)
    null = SensoryStream(seed+14000).draw(config.block*config.null_blocks)[0]
    alarms = sum(detect(initial, generator, calibration, null[i:i+config.block], rng)[1]
                 for i in range(0, len(null), config.block))
    shadow_alarms = sum(detect(shadow[0], shadow_generator, shadow_calibration, null[i:i+config.block], rng)[1]
                        for i in range(0, len(null), config.block))
    return initial, policy, generator, calibration, {'initial_null_alarms':alarms,
        'shadow_null_alarms':shadow_alarms, 'null_observations_each':len(null),
        'shadow_acceptance':shadow_acceptance, 'critical_component':critical_component,
        'noncritical_component':noncritical_component,
        'excluded_component_mass_difference':float(abs(generator['probabilities'][critical_component]-generator['probabilities'][noncritical_component]))}


def candidate_checks(candidates, incumbent, generator, heldout, calibration, config, rng):
    errors = [reconstruction_errors(model, heldout).mean() for model in candidates]
    incumbent_error = reconstruction_errors(incumbent, heldout).mean()
    consistency = float(abs(errors[0]-errors[1])/max(np.mean(errors), 1e-8))
    improvement = float((incumbent_error-max(errors))/max(incumbent_error,1e-8))
    # Fresh candidate references use generator fitted on training blocks by the caller.
    checks = {'heldout_improvement':improvement, 'candidate_consistency':consistency,
              'improvement_pass':improvement >= config.improvement_fraction,
              'consistency_pass':consistency <= config.candidate_consistency}
    checks['passed'] = checks['improvement_pass'] and checks['consistency_pass']
    return checks


def run_replay_seed(seed, config, output):
    start = time.process_time(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    initial, policy, old_generator, old_calibration, baseline = initialize(seed, config)
    initial_hash, policy_hash = weight_digest(initial), weight_digest(policy)
    scoring = SensoryStream(seed+99000).draw(config.evaluation_samples)
    measurements, events, traces = [], [], []
    for stream_id in range(config.streams):
        for arm in config.arms:
            rng = np.random.default_rng(seed+20000+stream_id)
            environment = SensoryTimeline(seed+30000+stream_id*10000,config)
            model, generator, calibration = deepcopy(initial), deepcopy(old_generator), deepcopy(old_calibration)
            fixed_pool = generate(initial, old_generator, rng, config.block*4)[0]
            excluded = baseline['critical_component'] if arm == 'critical_removed' else baseline['noncritical_component'] if arm == 'noncritical_removed' else None
            if excluded is not None: fixed_pool = generate(initial, old_generator, rng, config.block*4, excluded)[0]
            pool_quality = float(reconstruction_errors(initial, fixed_pool).mean())
            with torch.no_grad():
                generated_margin = policy(initial.encoder(torch.from_numpy(fixed_pool))).softmax(-1).max(-1).values.numpy()
            record = {'seed':seed,'stream':stream_id,'arm':arm,'promotions':0,'unresolved':0,
                      'rollbacks':0,'detected_changes':0,'repeated_alarms':0,'optimization_steps':0,
                      'generated_self_reconstruction':pool_quality,
                      'generated_low_margin_fraction':float((generated_margin < .90).mean())}
            offset = 0; unresolved = False; cooldown = 0; periodic_index = 0
            next_periodic = 4096
            # Only evaluator uses these times; the detector itself runs on every block.
            change_times = np.cumsum(config.private_phase_lengths)[:-1]
            detected = set(); first_delays = {}; previous_promotion = None
            required_forward = config.block*(16+config.calibration_blocks)
            while environment.remaining >= config.block:
                probe_start = environment.position
                probe = environment.draw(config.block)
                before_hash = weight_digest(model)
                statistic, alarm = detect(model, generator, calibration, probe, rng)
                assert before_hash == weight_digest(model)
                active_alarm = alarm and cooldown <= 0
                traces.append({'seed':seed,'stream':stream_id,'arm':arm,'observation_offset':probe_start,
                               'statistic':statistic,'threshold':calibration['threshold'],'alarm':alarm})
                if active_alarm:
                    for change_index,change in enumerate(change_times):
                        if 0 <= probe_start-change < 1024 and change_index not in detected:
                            detected.add(change_index);first_delays[str(change_index)] = int(probe_start-change+config.block)
                    if previous_promotion is not None and 0 <= probe_start-previous_promotion < 1024:
                        record['repeated_alarms'] += 1
                # Periodic updates use a fixed clock, no reading of the other arm's alarms.
                periodic_due = environment.position >= next_periodic and periodic_index < 3
                trigger = active_alarm and not unresolved if arm != 'periodic' else periodic_due
                accepted = False
                if trigger and environment.remaining >= required_forward:
                    if arm == 'periodic':
                        periodic_index += 1; next_periodic += 32768
                    for retry in range(config.max_retries+1):
                        if environment.remaining < required_forward: break
                        split_start = environment.position
                        first, second, acceptance, calibration_stream = [environment.draw(config.block*4) for _ in range(4)]
                        training_pools = [first, second]
                        replay_pool = fixed_pool if arm not in ('no_replay','current_repeat') else None
                        # Equal minibatch size and optimizer updates. Current-repeat is deliberately the same
                        # sampling distribution as no-replay: extra optimization is not confounded with replay.
                        candidates = [train_candidate(model, data, data if replay_pool is None else replay_pool, config,rng)
                                      for data in training_pools]
                        record['optimization_steps'] += 2*config.candidate_updates
                        checks = candidate_checks(candidates, model, generator, acceptance, calibration,config,rng)
                        generators = [fit_generator(candidate,data) for candidate,data in zip(candidates, training_pools)]
                        # Temporary calibration comes from separate future blocks, not the heldout acceptance block.
                        temporary = [calibrate(candidate,g,calibration_stream,config,rng) for candidate,g in zip(candidates,generators)]
                        fresh_checks = [detect(candidate,g,cal,acceptance[:config.block],rng)
                                        for candidate,g,cal in zip(candidates,generators,temporary)]
                        quality = [float(reconstruction_errors(candidate, generate(candidate,g,rng,config.block*4)[0]).mean())
                                   for candidate,g in zip(candidates,generators)]
                        real_errors = [float(reconstruction_errors(candidate,acceptance).mean()) for candidate in candidates]
                        quality_pass = all(q <= config.quality_ratio*max(e,1e-6) for q,e in zip(quality,real_errors))
                        checks.update(candidate_reference_in_band=all(not alarm for _,alarm in fresh_checks),
                                      generation_quality_pass=quality_pass, fresh_generation_errors=quality,
                                      training_ranges=[[split_start,split_start+config.block*4],
                                                       [split_start+config.block*4,split_start+config.block*8]],
                                      acceptance_range=[split_start+config.block*8,split_start+config.block*12],
                                      calibration_range=[split_start+config.block*12,split_start+config.block*16])
                        checks['passed'] = bool(checks['passed'] and checks['candidate_reference_in_band'] and quality_pass)
                        promote = checks['passed'] or arm == 'periodic'
                        events.append({'seed':seed,'stream':stream_id,'arm':arm,'attempt':retry,
                                       'offset':probe_start,'checks':checks,'promoted':promote,'periodic_bypass':arm=='periodic'})
                        if promote:
                            model,generator,calibration = candidates[0],generators[0],temporary[0]
                            # Independently generated forward calibration blocks, all included, assume local stationarity.
                            independent = environment.draw(config.block*config.calibration_blocks)
                            calibration = calibrate(model,generator,independent,config,rng)
                            record['promotions'] += 1; accepted = True
                            previous_promotion = environment.position
                            break
                        record['rollbacks'] += 1
                    if not accepted:
                        record['unresolved'] += 1; unresolved = True
                    cooldown = 4
                cooldown -= 1; offset += 1
            record['detected_changes'] = len(detected)
            record['detection_delays'] = first_delays
            record['stream_observations'] = environment.position
            record['monitored_blocks'] = len([t for t in traces if t['seed']==seed and t['stream']==stream_id and t['arm']==arm])
            record.update(evaluate_interface(model,policy,initial,*scoring))
            measurements.append(record)
            torch.save(model.state_dict(),output/f'replay_{seed}_{stream_id}_{arm}.pt')
    # Rewards never enter reconstruction statistics. Compare identical observation stream under reward-only
    # changes: detector's trace is exactly invariant to unseen rewards (structural blindspot, not sensitivity evidence).
    assert initial_hash == weight_digest(initial) and policy_hash == weight_digest(policy)
    np.savez_compressed(output/f'replay_scoring_{seed}.npz',observations=scoring[0],evaluator_labels=scoring[1],evaluator_clusters=scoring[2])
    torch.save(initial.state_dict(),output/f'replay_{seed}_initial.pt')
    torch.save(policy.state_dict(),output/f'replay_{seed}_frozen_policy.pt')
    return {'hypothesis_id':'H009','seed':seed,'measurements':measurements,'events':events,'trajectory':traces,
            'baseline':baseline,'cpu_seconds':time.process_time()-start,'runtime_labels_used':False,
            'initialization_labels_used':True,'old_policy_frozen_verified':True,
            'reward_only_detector_invariant':True,
            'calibration_stationarity':'conditional: complete forward blocks supplied; algorithm receives no stationarity oracle',
            'information_contract':['fresh observation','generated replay','fixed internal minibatches','internal model state']}
