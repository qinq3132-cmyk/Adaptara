"""H010: an online reward learner and executable protection controller.

The learner fits rewards from its own interactions. It never receives task weights,
best-action labels or change times. All private ground truth stays in the driver.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class ConsolidationConfig:
    dimensions: int = 8
    actions: int = 16
    initialization_steps: int = 1000
    online_steps: int = 2000
    private_change_step: int = 600
    learning_rate: float = .04
    epsilon: float = .15
    short_window: int = 100
    long_window: int = 400
    threshold: float = 1.
    alarm_run: int = 5
    in_band_run: int = 20
    capability_run: int = 5
    capability_threshold: float = .80
    probe_interval: int = 25
    fail_after: int = 500
    probe_steps: int = 100
    wait_steps: int = 200
    lambda_grid: tuple = (10., 3., 1., .3)
    snapshot_interval: int = 100
    snapshot_lag: int = 50
    nominal_protection_steps: int = 700
    release_steps: int = 100
    arms: tuple = ('state_machine', 'gate_ablation', 'shuffled_importance', 'uniform_importance',
                   'fixed', 'random', 'continuous', 'no_protection', 'alarm_snapshot')


class ProtectionController:
    """Explicit guards; candidate parameter acceptance happens only after S2."""
    def __init__(self, config, *, gate_ablation=False):
        self.config = config
        self.gate_ablation = gate_ablation
        self.state = 'S0'; self.age = 0; self.band_run = 0; self.good_run = 0; self.alarm_count = 0
        self.probe_index = 0; self.probe_age = 0; self.wait_age = 0; self.release_age = 0
        self.lam = 0.; self.events = []; self.cycles = 0
        self.accepted = None; self.reference = None; self.importance = None; self.rollback = None
        self.rearmed = True; self.previous_in_band = True

    def move(self, target, reason, signal, capability):
        self.events.append({'from':self.state,'to':target,'reason':reason,
                            'signal':float(signal),'capability':float(capability),
                            'band_run':self.band_run,'good_run':self.good_run,'cycle':self.cycles})
        self.state = target; self.age = 0

    def observe(self, signal, capability, theta, snapshot, importance):
        c = self.config; self.age += 1
        in_band = abs(signal) < c.threshold
        renewed_crossing = self.previous_in_band and not in_band
        self.previous_in_band = in_band
        self.band_run = self.band_run+1 if in_band else 0
        self.good_run = self.good_run+1 if capability >= c.capability_threshold else 0
        self.alarm_count = self.alarm_count+1 if abs(signal) >= c.threshold else 0
        both = self.band_run >= c.in_band_run and self.good_run >= c.capability_run
        either = self.band_run >= c.in_band_run or self.good_run >= c.capability_run
        accepted_now = False; rollback_now = False
        if self.state in ('S0','SF','SF_partial'):
            self.lam = 0.
            if self.band_run >= c.in_band_run: self.rearmed = True
            if self.rearmed and self.alarm_count >= c.alarm_run:
                self.cycles += 1
                self.reference = snapshot.copy(); self.importance = importance.copy()
                self.accepted = theta.copy(); self.rollback = theta.copy()
                self.probe_index = 0; self.probe_age = 0; self.wait_age = 0
                self.lam = c.lambda_grid[0]
                self.move('S1','persistent_reward_shift',signal,capability)
        elif self.state == 'S1':
            if both or self.gate_ablation and either:
                self.move('S2','joint_gate' if both else 'ablated_either_gate',signal,capability)
            elif self.age >= c.fail_after or self.probe_index:
                if self.probe_index == 0:
                    self.probe_index = 1; self.probe_age = 0; self.rollback = theta.copy()
                self.lam = c.lambda_grid[self.probe_index]
                self.probe_age += 1
                # A renewed crossing cancels this weakened branch immediately.
                if self.probe_age > 1 and renewed_crossing:
                    theta[:] = self.rollback; rollback_now = True
                    self.probe_index += 1; self.probe_age = 0
                elif self.good_run >= c.capability_run:
                    self.wait_age = 0; self.move('ST','capability_only_wait_for_reward_band',signal,capability)
                elif self.probe_age >= c.probe_steps:
                    theta[:] = self.rollback; rollback_now = True
                    self.probe_index += 1; self.probe_age = 0
                if self.probe_index >= len(c.lambda_grid):
                    self.lam = 0.; self.rearmed = False
                    self.move('SF','bounded_probes_failed',signal,capability)
        elif self.state == 'ST':
            self.wait_age += 1
            if both:
                self.move('S2','joint_gate_after_wait',signal,capability)
            elif renewed_crossing:
                theta[:] = self.rollback; rollback_now = True
                self.probe_index += 1; self.probe_age = 0
                if self.probe_index >= len(c.lambda_grid):
                    self.lam = 0.; self.rearmed = False
                    self.move('SF','renewed_alarm_exhausted_probes',signal,capability)
                else:
                    self.lam = c.lambda_grid[0]; self.move('S1','renewed_alarm_rollback',signal,capability)
            elif self.wait_age >= c.wait_steps:
                theta[:] = self.accepted; rollback_now = True; self.lam = 0.
                self.rearmed = False
                self.move('SF_partial' if self.good_run >= c.capability_run else 'SF',
                          'wait_timeout_partial' if self.good_run >= c.capability_run else 'wait_capability_failed',signal,capability)
        elif self.state == 'S2':
            # A gate that fails between assessment and acceptance cancels acceptance.
            if not both and not self.gate_ablation:
                self.move('S1','gate_lost_before_acceptance',signal,capability)
            else:
                self.accepted = theta.copy(); self.reference = theta.copy()
                accepted_now = True; self.release_age = 0
                self.move('S3','accept_parameters_then_reestimate_importance_then_decay',signal,capability)
        elif self.state == 'S3':
            self.release_age += 1
            self.lam = c.lambda_grid[0]*max(0., 1-self.release_age/c.release_steps)
            if self.release_age >= c.release_steps:
                self.lam = 0.; self.move('S0','release_complete',signal,capability)
        return accepted_now, rollback_now


def action_values(features, theta):
    return features @ theta


def importance_estimate(theta, features, observed_rewards, delta=.05):
    # Smooth native capability readout, with rewards estimated from the learner's
    # own probes. Adding an intercept changes no action preference, so F is zero.
    def capability(parameters):
        logits = action_values(features,parameters)/.15
        probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
        return float(probabilities @ observed_rewards)
    baseline = capability(theta)
    sensitivities = []
    for j in range(len(theta)):
        perturbed = theta.copy(); perturbed[j] += delta
        opposite = theta.copy(); opposite[j] -= delta
        sensitivity = (capability(perturbed)-capability(opposite))/(2*delta)
        sensitivities.append(sensitivity*sensitivity)
    result = np.asarray(sensitivities)
    return result/max(result.sum(), 1e-12)*len(result)


def private_environment(seed, config):
    rng = np.random.default_rng(seed+40000)
    features = rng.normal(size=(config.actions,config.dimensions))
    features /= np.linalg.norm(features,axis=1,keepdims=True)
    old_weights = np.zeros(config.dimensions); old_weights[:2] = (1.2,.5)
    new_weights = np.zeros(config.dimensions); new_weights[:2] = (-1.2,.7)
    old = features @ old_weights; new = features @ new_weights
    old = .1+.9*(old-old.min())/(old.max()-old.min())
    new = .05+.9*(new-new.min())/(new.max()-new.min())
    # Linear learner needs an intercept to predict positive native rewards.
    features = np.column_stack([features,np.ones(config.actions)])
    return features,old,new


def reward_update(theta, feature, reward, importance, reference, lam, rate):
    gradient = 2*(feature @ theta-reward)*feature
    if lam: gradient += 2*lam*importance*(theta-reference)
    theta -= rate*np.clip(gradient,-3,3)


def reward_signal(rewards, reference_scale, config):
    if len(rewards) < config.long_window: return 0.
    return (np.mean(rewards[-config.short_window:])-np.mean(rewards[-config.long_window:]))/max(reference_scale,.03)


def run_consolidation_seed(seed, config, output):
    started = time.process_time(); output = Path(output); output.mkdir(parents=True,exist_ok=True)
    features,old_rewards,new_rewards = private_environment(seed,config)
    rng = np.random.default_rng(seed+41000)
    initialized = np.zeros(features.shape[1]); initialization_rewards = []
    feedback_sums = np.zeros(config.actions); feedback_counts = np.zeros(config.actions)
    for _ in range(config.initialization_steps):
        action = int(rng.integers(config.actions))
        reward = old_rewards[action]+rng.normal(0,.015)
        reward_update(initialized,features[action],reward,np.ones_like(initialized),initialized,0,config.learning_rate)
        initialization_rewards.append(reward)
        feedback_sums[action] += reward; feedback_counts[action] += 1
    learned_probe_rewards = feedback_sums/np.maximum(feedback_counts,1)
    initial_importance = importance_estimate(initialized,features,learned_probe_rewards)
    reference_scale = float(np.std(initialization_rewards))
    # Independent high-amplitude parameter lesions measure actual native old-task loss.
    initial_old = float(old_rewards[(features@initialized).argmax()])
    lesion = []
    for j in range(len(initialized)):
        loss = []
        for sign in (-1,1):
            changed = initialized.copy(); changed[j] += sign*.8
            loss.append(initial_old-old_rewards[(features@changed).argmax()])
        lesion.append(np.mean(loss))
    correlation = float(spearmanr(initial_importance,lesion).statistic)
    if not np.isfinite(correlation): correlation = 0.
    common_noise = np.random.default_rng(seed+42000).normal(0,.015,(config.online_steps,5))
    exploration = np.random.default_rng(seed+43000).random(config.online_steps)
    exploration_actions = np.random.default_rng(seed+44000).integers(config.actions,size=config.online_steps)
    permutation = np.random.default_rng(seed+45000).permutation(len(initialized))
    random_start = int(np.random.default_rng(seed+46000).integers(0,config.online_steps-config.nominal_protection_steps))
    measurements,trajectories,events,checkpoints = [],[],[],{}
    for arm in config.arms:
        theta = initialized.copy(); snapshot = initialized.copy(); imp = initial_importance.copy()
        controller = ProtectionController(config,gate_ablation=arm=='gate_ablation')
        observed_rewards = learned_probe_rewards.copy()
        rewards = []; task_scores = []; capability = initial_old
        pending_snapshot = None; snapshot_step = -config.snapshot_lag
        audit = []; previous = None; coefficient_sum = 0.; sf_learning = 0; outcomes = []
        for step in range(config.online_steps):
            # Only driver knows the hidden change. The learner receives a chosen scalar reward.
            truth_rewards = old_rewards if step < config.private_change_step else new_rewards
            action = int(exploration_actions[step]) if exploration[step] < config.epsilon else int((features@theta).argmax())
            reward = float(truth_rewards[action]+common_noise[step,0])
            rewards.append(reward); task_scores.append(float(truth_rewards[action]))
            observed_rewards[action] = .9*observed_rewards[action]+.1*reward
            # Independent permitted task readout: same current greedy action tested four times.
            # It is a separate feedback channel, not a moving reward-difference proxy or optimal-action label.
            if step % config.probe_interval == 0:
                probe_action = int((features@theta).argmax())
                capability = float(np.mean(truth_rewards[probe_action]+common_noise[step,1:]))
            signal = reward_signal(rewards,reference_scale,config)
            if pending_snapshot is not None and step-pending_snapshot[0] >= config.snapshot_lag:
                snapshot_step,snapshot,imp = pending_snapshot; pending_snapshot = None
            if step % config.snapshot_interval == 0 and abs(signal) < .5*config.threshold and capability >= config.capability_threshold and controller.state=='S0':
                pending_snapshot = (step,theta.copy(),importance_estimate(theta,features,observed_rewards))
            current_imp = imp[permutation] if arm=='shuffled_importance' else np.ones_like(imp) if arm=='uniform_importance' else imp
            if arm in ('state_machine','gate_ablation','shuffled_importance','uniform_importance','alarm_snapshot'):
                accepted,rolled_back = controller.observe(signal,capability,theta,
                    theta if arm=='alarm_snapshot' else snapshot,current_imp)
                if accepted:
                    # Accept parameters before changing importance and reference.
                    snapshot = theta.copy(); snapshot_step = step
                    imp = importance_estimate(theta,features,observed_rewards); controller.importance = imp.copy()
                    outcomes.append('normal_recovery')
                lam = controller.lam; reference = controller.reference if controller.reference is not None else snapshot
                actual_imp = controller.importance if controller.importance is not None else current_imp
            else:
                accepted = rolled_back = False; reference = snapshot; actual_imp = current_imp
                if arm=='fixed': lam = config.lambda_grid[0] if 650 <= step < 650+config.nominal_protection_steps else 0.
                elif arm=='random': lam = config.lambda_grid[0] if random_start <= step < random_start+config.nominal_protection_steps else 0.
                elif arm=='continuous': lam = config.lambda_grid[0]*config.nominal_protection_steps/config.online_steps
                else: lam = 0.
            coefficient_sum += lam
            reward_update(theta,features[action],reward,actual_imp,reference,lam,config.learning_rate)
            if controller.state in ('SF','SF_partial'): sf_learning += 1
            if controller.state != previous or step % 25 == 0:
                audit.append({'seed':seed,'arm':arm,'step':step,'state':controller.state,
                    'signal':float(signal),'capability':capability,'lambda':lam,
                    'snapshot_age':step-snapshot_step,'accepted':accepted,'rolled_back':rolled_back,
                    'parameter_displacement':float(np.linalg.norm(theta-initialized))})
                previous = controller.state
        changes = theta-initialized
        selected = np.argsort(initial_importance)[-max(1,len(initialized)//4):]
        after = np.asarray(task_scores[config.private_change_step:])
        rolling = np.convolve(after,np.ones(50)/50,mode='valid')
        success = np.flatnonzero(rolling >= .9*new_rewards.max())
        illegal = sum(e['to']=='S2' and not (e['band_run']>=config.in_band_run and e['good_run']>=config.capability_run)
                      for e in controller.events)
        if arm=='gate_ablation': illegal = int(illegal)  # deliberate violation for diagnostic comparison
        result = {'seed':seed,'arm':arm,'new_reward_auc':float(after.mean()/new_rewards.max()),
            'old_retention':float(old_rewards[(features@theta).argmax()]/old_rewards.max()),
            'recovery_delay':int(success[0]+50 if len(success) else len(after)),
            'recovery_censored':not bool(len(success)), 'importance_lesion_correlation':correlation,
            'high_importance_displacement':float(np.linalg.norm(changes[selected])/max(np.linalg.norm(changes),1e-12)),
            'illegal_joint_gate_transitions':int(illegal),'lambda_integral':float(coefficient_sum),
            'nominal_lambda_integral':float(config.lambda_grid[0]*config.nominal_protection_steps),
            'optimization_steps':config.online_steps,'terminal_state':controller.state,
            'normal_recovery_cycles':len(outcomes),'failure_then_learning_steps':sf_learning,
            'initial_old_task_reward':initial_old,'parameter_count':len(theta)}
        measurements.append(result);trajectories.extend(audit)
        events.extend(dict(e,seed=seed,arm=arm) for e in controller.events)
        checkpoints[arm+'_parameters'] = theta;checkpoints[arm+'_rewards'] = np.asarray(rewards)
    np.savez_compressed(output/f'consolidation_{seed}.npz',initialized_parameters=initialized,
        learned_importance=initial_importance,lesion_losses=np.asarray(lesion),public_features=features,
        evaluator_old_rewards=old_rewards,evaluator_new_rewards=new_rewards,**checkpoints)
    return {'hypothesis_id':'H010','seed':seed,'measurements':measurements,'events':events,'trajectory':trajectories,
            'cpu_seconds':time.process_time()-started,'runtime_labels_used':False,
            'information_contract':['candidate action features','chosen native scalar reward','native capability probe feedback','internal state'],
            'reward_signal_uses_capability':False,'private_change_signal_sent_to_agent':False}
