"""Information boundaries, real learning, acceptance and protection closure."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.learned.history import HistoryConfig, train_policies, weight_digest, generate_feedback
from experiments.learned.replay import (ReplayConfig, SensoryTimeline, Autoencoder, candidate_checks,
                                      fit_generator, train_candidate, reconstruction_errors)
from experiments.learned.consolidation import (ConsolidationConfig, ProtectionController,
                                              importance_estimate, reward_update)
from experiments.learned.statistics import paired_interval, classification
from scientist.experiment_validation import ImplementationCheck, ImplementationReview, validate_targets


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_current_rewards_and_context_do_not_enter_history_at_same_step():
    inputs, actions, rewards, truth = generate_feedback(np.random.default_rng(91),64,8,good_probability=.8)
    assert np.all(inputs[:,0] == 0) and np.all(inputs[:,:,0] == 0)
    np.testing.assert_array_equal(inputs[:,1:,3],rewards[:,:-1])
    np.testing.assert_array_equal(inputs[:,1:,1],actions[:,:-1] == 0)
    assert inputs.shape[-1] == 4 and truth.shape == rewards.shape


def test_reward_learning_is_deterministic_and_changes_action_probabilities():
    config = HistoryConfig(training_updates=12,batch=32)
    models, trace = train_policies(91,config)
    repeated, repeated_trace = train_policies(91,config)
    assert trace == repeated_trace
    assert {k:weight_digest(v) for k,v in models.items()} == {k:weight_digest(v) for k,v in repeated.items()}
    assert trace and all(np.isfinite(row['single_loss']) for row in trace)
    probe = torch.zeros(2,8,4); probe[0,1:,1] = 1; probe[1,1:,2] = 1
    probe[:,1:,3] = 1
    before = weight_digest(models['mixture'])
    normal = models['mixture'](probe)[0].detach().numpy()
    swapped = models['mixture'](probe,'swap')[0].detach().numpy()
    assert not np.array_equal(normal,swapped)
    assert weight_digest(models['mixture']) == before


def test_online_observations_match_at_the_same_clock_despite_different_draw_sizes():
    config = replace(ReplayConfig(),block=16,private_phase_lengths=(32,64,64,64))
    left,right = SensoryTimeline(81,config),SensoryTimeline(81,config)
    grouped = left.draw(64)
    split = np.concatenate([right.draw(16) for _ in range(4)])
    np.testing.assert_array_equal(grouped,split)
    assert grouped.shape == (64,8)


def test_autoencoder_candidate_learns_observations_and_leaves_incumbent_frozen():
    torch.manual_seed(81)
    model = Autoencoder(12)
    values = np.random.default_rng(81).normal(size=(128,8)).astype(np.float32)
    before = weight_digest(model)
    config = replace(ReplayConfig(),candidate_updates=60,learning_rate=.01)
    candidate = train_candidate(model,values,values,config,np.random.default_rng(81))
    assert weight_digest(model) == before and weight_digest(candidate) != before
    assert reconstruction_errors(candidate,values).mean() < reconstruction_errors(model,values).mean()
    generator = fit_generator(candidate,values)
    assert np.isclose(generator['probabilities'].sum(),1) and generator['centers'].shape == (4,2)


def test_candidate_failing_on_forward_acceptance_cannot_promote():
    torch.manual_seed(81)
    incumbent = Autoencoder(12)
    heldout = np.ones((64,8),np.float32)
    damaged = Autoencoder(12)
    for p in damaged.parameters(): p.data.fill_(5.)
    checks = candidate_checks([damaged,damaged],incumbent,None,heldout,None,ReplayConfig(),None)
    assert not checks['passed'] and not checks['improvement_pass']


def controller(config=None):
    config = config or replace(ConsolidationConfig(),alarm_run=1,in_band_run=3,capability_run=2,fail_after=4,probe_steps=3,wait_steps=4)
    agent = ProtectionController(config); parameters = np.array([1.,2.]); snapshot = parameters.copy()
    agent.observe(-2.,.2,parameters,snapshot,np.ones(2))
    assert agent.state == 'S1'
    return agent,parameters,snapshot


def test_reward_band_alone_cannot_release_protection():
    agent,parameters,snapshot = controller()
    for _ in range(5): agent.observe(0.,.2,parameters,snapshot,np.ones(2))
    assert agent.state != 'S2' and agent.state != 'S3'


def test_capability_alone_waits_then_partial_failure_rolls_back():
    agent,parameters,snapshot = controller()
    for _ in range(4): agent.observe(-2.,.95,parameters,snapshot,np.ones(2))
    assert agent.state == 'ST'
    parameters += .5
    for _ in range(4): agent.observe(-2.,.95,parameters,snapshot,np.ones(2))
    assert agent.state == 'SF_partial'
    np.testing.assert_array_equal(parameters,snapshot)
    assert not any(e['to']=='S2' for e in agent.events)


def test_joint_gate_accepts_before_importance_refresh_and_release():
    agent,parameters,snapshot = controller()
    parameters += .1
    for _ in range(3): agent.observe(0.,.95,parameters,snapshot,np.ones(2))
    assert agent.state == 'S2' and not np.array_equal(agent.accepted,parameters)
    accepted,rollback = agent.observe(0.,.95,parameters,snapshot,np.ones(2))
    assert accepted and not rollback and agent.state == 'S3'
    np.testing.assert_array_equal(agent.accepted,parameters)


def test_failed_cycle_is_not_reclassified_or_restarted_by_same_persistent_alarm():
    agent,parameters,snapshot = controller()
    for _ in range(20): agent.observe(-2.,.1,parameters,snapshot,np.ones(2))
    assert agent.state == 'SF' and agent.cycles == 1
    cycles = agent.cycles
    for _ in range(10): agent.observe(-2.,.1,parameters,snapshot,np.ones(2))
    assert agent.cycles == cycles
    prior = parameters.copy()
    reward_update(parameters,np.array([1.,.5]),.8,np.ones(2),snapshot,0.,.1)
    assert not np.array_equal(prior,parameters)
    for _ in range(3): agent.observe(0.,.1,parameters,snapshot,np.ones(2))
    agent.observe(-2.,.1,parameters,snapshot,np.ones(2))
    assert agent.cycles == 2


def test_importance_uses_native_capability_and_common_intercept_has_zero_sensitivity():
    theta = np.array([.5,.2]); features = np.array([[1.,1.],[-1.,1.]])
    importance = importance_estimate(theta,features,np.array([1.,0.]))
    reversed_importance = importance_estimate(theta,features,np.array([.5,.5]))
    assert importance[0] > 0 and importance[1] < 1e-10
    np.testing.assert_allclose(reversed_importance,0,atol=1e-10)


def test_interval_requires_independent_seed_summaries_and_does_not_infer_equivalence_from_nonsignificance():
    with pytest.raises(ValueError):paired_interval([[1,2],[3,4]],[[1,2],[3,4]])
    with pytest.raises(ValueError):paired_interval([1,2,float('nan')],[1,2,3])
    interval = paired_interval([.3,.3,.3],[0,0,0])
    assert classification(interval,direction='positive',minimum_effect=.2,equivalence_margin=.1)=='supports_local_prediction'
    assert classification({'lower':-.5,'upper':.5},direction='positive',minimum_effect=.2,equivalence_margin=.1)=='inconclusive'


def test_model_review_cannot_accept_unclosed_mandatory_repairs():
    data = dict(hypothesis_id='H006',verdict='ready_for_exploratory_run',implemented_mechanism='learned memory',
                mandatory_repairs=['fix leakage'],scientific_limits=['finite test'],human_summary='修复泄漏')
    with pytest.raises(ValueError):ImplementationCheck(**data)
    data.update(mandatory_repairs=[])
    result = ImplementationReview(checks=(ImplementationCheck(**data),),overall_summary='ready')
    with pytest.raises(ValueError):validate_targets(result,['H006','H009'])


def test_preregistration_rejects_unreviewed_source_and_does_not_run_seeds(tmp_path,monkeypatch):
    from scientist import learned_validation as workflow
    root = tmp_path/'repository'; root.mkdir()
    (root/'module.py').write_text('new source')
    monkeypatch.setattr(workflow,'ROOT',root)
    request = {'new_experiment_source':{'module.py':'old source'}}
    review = {'request_sha256':workflow.sha(request),'review':{'checks':[
        {'hypothesis_id':h,'verdict':'ready_for_exploratory_run','implemented_mechanism':'learned policy',
         'mandatory_repairs':[],'scientific_limits':['finite experiment'],'human_summary':'可进行局部试跑'}
        for h in workflow.TARGETS],'overall_summary':'local pilot ready'}}
    request_path=tmp_path/'request.json';review_path=tmp_path/'review.json'
    workflow.save_json(request_path,request);workflow.save_json(review_path,review)
    with pytest.raises(ValueError,match='code changed'):
        workflow.preregister(request_path,review_path,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_preregistration_never_treats_open_repairs_as_ready(tmp_path):
    from scientist import learned_validation as workflow
    request = {};review={'request_sha256':workflow.sha(request),'review':{'checks':[
        {'hypothesis_id':h,'verdict':'revise','implemented_mechanism':'partial learner',
         'mandatory_repairs':['remove oracle context'],'scientific_limits':['finite experiment'],'human_summary':'需要修补'}
        for h in workflow.TARGETS],'overall_summary':'revise'}}
    workflow.save_json(tmp_path/'request.json',request);workflow.save_json(tmp_path/'review.json',review)
    with pytest.raises(ValueError,match='mandatory implementation repair'):
        workflow.preregister(tmp_path/'request.json',tmp_path/'review.json',tmp_path/'output')


def test_raw_checkpoints_are_immutable_and_only_cpu_timing_is_excluded_from_replay(tmp_path):
    from scientist.learned_validation import save_json, _without_timing
    record = {'measurements':[{'reward':.5}],'cpu_seconds':.4}
    path = tmp_path/'raw.json';save_json(path,record);save_json(path,record)
    with pytest.raises(ValueError,match='replace existing record'):save_json(path,{'measurements':[]})
    assert _without_timing(record) == _without_timing({**record,'cpu_seconds':2.})
    assert _without_timing(record) != _without_timing({'measurements':[{'reward':.6}],'cpu_seconds':.4})
