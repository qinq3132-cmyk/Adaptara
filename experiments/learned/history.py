"""H006: reward-trained history policies with frozen-weight causal interventions.

Context is private simulator truth, never an input or a training label. Initialization
uses off-policy policy gradients on environment feedback from a random behavior
policy; both architectures receive the identical stream. No supervised context
classifier, analytical posterior, or task-labelled response bank is substituted.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class HistoryConfig:
    hidden: int = 16
    training_updates: int = 600
    batch: int = 128
    sequence_length: int = 16
    learning_rate: float = .003
    training_reward_probability: float = .8
    training_switch_probability: float = .03
    evaluation_episodes: int = 1024
    probabilities: tuple = (.6, .8, .95, .5)
    history_lengths: tuple = (1, 2, 4, 8, 16)


def generate_feedback(rng, count, length, *, good_probability, switch_probability=0.):
    """The only teacher signal is native reward; returned truth is scoring-only."""
    context = rng.integers(0, 2, count)
    actions = rng.integers(0, 2, (count, length))
    rewards = np.empty((count, length), dtype=np.float32)
    truth = np.empty((count, length), dtype=np.int64)
    for step in range(length):
        context = np.where(rng.random(count) < switch_probability, 1-context, context)
        truth[:, step] = context
        probability = np.where(actions[:, step] == context, good_probability, 1-good_probability)
        rewards[:, step] = (rng.random(count) < probability).astype(np.float32)
    # Feature at t includes feedback from t-1, never its current outcome or context.
    inputs = np.zeros((count, length, 4), dtype=np.float32)
    if length > 1:
        inputs[:, 1:, 1] = (actions[:, :-1] == 0)
        inputs[:, 1:, 2] = (actions[:, :-1] == 1)
        inputs[:, 1:, 3] = rewards[:, :-1]
    return inputs, actions, rewards, truth


class HistoryPolicy(nn.Module):
    def __init__(self, hidden, architecture):
        super().__init__()
        self.architecture = architecture
        self.memory = nn.GRU(4, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 2)
        if architecture == 'mixture':
            self.response_logits = nn.Parameter(torch.randn(2, 2)*.3)
        elif architecture != 'single':
            raise ValueError('Unknown policy architecture')

    def forward(self, inputs, intervention='normal'):
        states, _ = self.memory(inputs)
        logits = self.output(states)
        if self.architecture == 'single':
            if intervention != 'normal':
                raise ValueError('Routing interventions only apply to mixture policies')
            return logits.softmax(-1), None
        belief = logits.softmax(-1)
        if intervention == 'clamp': belief = torch.full_like(belief, .5)
        elif intervention == 'swap': belief = belief.flip(-1)
        elif intervention != 'normal': raise ValueError('Unknown routing intervention')
        return belief @ self.response_logits.softmax(-1), belief


def weight_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_policies(seed, config):
    torch.manual_seed(seed)
    models = {name: HistoryPolicy(config.hidden, name) for name in ('mixture', 'single')}
    # Pair recurrent initialization; the four extra expert logits are <1% of capacity.
    models['single'].memory.load_state_dict(models['mixture'].memory.state_dict())
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=config.learning_rate) for name, model in models.items()}
    initial = {name: weight_digest(model) for name, model in models.items()}
    rng = np.random.default_rng(seed+90000)
    losses = []
    for update in range(config.training_updates):
        inputs, actions, rewards, _ = generate_feedback(rng, config.batch, config.sequence_length,
            good_probability=config.training_reward_probability, switch_probability=config.training_switch_probability)
        history = torch.from_numpy(inputs)
        chosen = torch.from_numpy(actions).unsqueeze(-1)
        advantage = torch.from_numpy(rewards)-.5
        row = {'update': update}
        for name, model in models.items():
            probabilities, _ = model(history)
            likelihood = probabilities.gather(-1, chosen).squeeze(-1).clamp_min(1e-6)
            # Behavior probability is 1/2. Detach the importance ratio in the score-function estimator.
            ratio = (likelihood/.5).detach()
            entropy = -(probabilities*probabilities.clamp_min(1e-6).log()).sum(-1).mean()
            loss = -(ratio*likelihood.log()*advantage).mean()-.005*entropy
            optimizers[name].zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizers[name].step()
            row[name+'_loss'] = float(loss.detach())
        if update % 20 == 0: losses.append(row)
    for name, model in models.items():
        assert weight_digest(model) != initial[name], 'Learning must actually change parameters'
        model.eval()
    return models, losses


def js_divergence(first, second):
    first, second = np.clip(first, 1e-9, 1), np.clip(second, 1e-9, 1)
    middle = (first+second)/2
    return .5*(np.sum(first*np.log2(first/middle), -1)+np.sum(second*np.log2(second/middle), -1))


def evaluate_policies(seed, models, config, output):
    records = []; probes = {}
    before = {name: weight_digest(model) for name, model in models.items()}
    rng = np.random.default_rng(seed+180000)
    coverage = models['mixture'].response_logits.detach().softmax(-1).numpy()
    for probability in config.probabilities:
        for length in config.history_lengths:
            inputs, actions, rewards, contexts = generate_feedback(rng, config.evaluation_episodes, length+1,
                                                                  good_probability=probability)
            history = torch.from_numpy(inputs)
            shuffle = rng.permutation(len(inputs))
            reward_shuffle = inputs.copy()
            for row in reward_shuffle:
                row[1:,3] = rng.permutation(row[1:,3])
            inverted = inputs.copy(); inverted[:,1:,3] = 1-inverted[:,1:,3]
            masked = np.zeros_like(inputs)
            with torch.no_grad():
                normal, beliefs = models['mixture'](history)
                variants = {'normal': normal[:, -1].numpy(),
                    'shuffle': models['mixture'](history[shuffle])[0][:, -1].numpy(),
                    'reward_shuffle': models['mixture'](torch.from_numpy(reward_shuffle))[0][:,-1].numpy(),
                    'reward_inversion': models['mixture'](torch.from_numpy(inverted))[0][:,-1].numpy(),
                    'no_history': models['mixture'](torch.from_numpy(masked))[0][:, -1].numpy(),
                    'clamp': models['mixture'](history, 'clamp')[0][:, -1].numpy(),
                    'swap': models['mixture'](history, 'swap')[0][:, -1].numpy(),
                    'single': models['single'](history)[0][:, -1].numpy()}
            truth = contexts[:, -1]
            if probability == .8 and length == 8:
                probes.update(public_history=inputs, public_actions=actions, public_rewards=rewards,
                              evaluator_context=truth, normal_probabilities=variants['normal'],
                              shuffle_probabilities=variants['shuffle'], swap_probabilities=variants['swap'])
            # Bayes-optimal reference uses simulator parameters only as an evaluator diagnostic.
            evidence = (2*actions[:, :length]-1)*(2*rewards[:, :length]-1)
            vote = evidence.sum(axis=1)
            oracle_error = np.mean(np.where(vote==0,.5,(vote>0).astype(int)!=truth))
            for name, predictions in variants.items():
                conditional = [predictions[truth==context].mean(axis=0) for context in (0, 1)]
                records.append({'seed':seed,'arm':name,'reward_probability':probability,'history_length':length,
                    'expected_error':float((1-predictions[np.arange(len(truth)),truth]).mean()),
                    'greedy_error':float((predictions.argmax(-1)!=truth).mean()),
                    'action_separation_bits':float(js_divergence(conditional[0],conditional[1])),
                    'same_observation_separation':float(abs(conditional[0][1]-conditional[1][1])),
                    'minimum_context_greedy_accuracy':float(min((predictions[truth==context].argmax(-1)==context).mean() for context in (0,1))),
                    'oracle_information_error':float(oracle_error),
                    'expert_coverage':float(coverage.max(axis=0).min()),
                    'parameters':sum(p.numel() for p in models['single' if name=='single' else 'mixture'].parameters()),
                    'frozen_weights_verified':True})
    # Endogenous frozen inference after an unannounced change in feedback, with a shared zero start.
    count = config.evaluation_episodes; steps = 33
    context = rng.integers(0,2,count)
    truth = np.tile(context[:,None],(1,steps)); truth[:,16:] = 1-truth[:,16:]
    actions = rng.integers(0,2,(count,steps))
    reward_probability = np.where(actions==truth,.8,.2)
    rewards = (rng.random((count,steps)) < reward_probability).astype(np.float32)
    inputs = np.zeros((count,steps,4),np.float32)
    inputs[:,1:,1] = actions[:,:-1]==0; inputs[:,1:,2] = actions[:,:-1]==1; inputs[:,1:,3] = rewards[:,:-1]
    with torch.no_grad():
        for name,model in models.items():
            probabilities = model(torch.from_numpy(inputs))[0].numpy()
            correct = np.take_along_axis(probabilities,truth[:,:,None],axis=-1).squeeze(-1)
            after = correct[:,17:] >= .5
            delays = np.where(after.any(-1),after.argmax(-1)+1,16)
            records.append({'seed':seed,'arm':name+'_hidden_switch','reward_probability':.8,
                'history_length':16,'switch_latency':float(delays.mean()),'switch_censored_fraction':float((~after.any(-1)).mean()),
                'pre_switch_expected_error':float((1-correct[:,12:16]).mean()),
                'post_switch_expected_error':float((1-correct[:,25:]).mean()),'frozen_weights_verified':True})
            probes[name+'_hidden_switch_probabilities'] = probabilities
    probes['hidden_switch_public_history'] = inputs; probes['hidden_switch_evaluator_truth'] = truth
    assert {name:weight_digest(model) for name,model in models.items()} == before
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/f'history_probes_{seed}.npz',**probes)
    for name,model in models.items():torch.save(model.state_dict(),output/f'history_{seed}_{name}.pt')
    return records


def run_history_seed(seed, config, output):
    start = time.process_time()
    models, training = train_policies(seed, config)
    records = evaluate_policies(seed, models, config, Path(output))
    return {'hypothesis_id':'H006','seed':seed,'measurements':records,'training_curve':training,
            'cpu_seconds':time.process_time()-start,'training_updates_per_architecture':config.training_updates,
            'information_contract':['neutral observation','previous action','previous native reward'],
            'context_labels_used_for_training':False,'initialization_mode':'off-policy reward policy gradient',
            'runtime_parameter_updates':0}
