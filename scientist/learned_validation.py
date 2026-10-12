"""Explicit learned-experiment entry points, with immutable requests and checkpoints.

Usage: python -m scientist.learned_validation {review,pilot,preregister,run,analyze}
No toy fallback. The review command transmits only the supplied JSON request.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import yaml

from .experiment_validation import ImplementationReviewAgent, ResultAnalysisAgent, validate_targets
from .hypothesis_generation import ReasoningConfig, make_reasoning_backend
from .provenance import capture_source
from .tools.experiment_tracker import json_bytes

TARGETS = ('H006','H009','H010')
ROOT = Path(__file__).resolve().parents[1]


def sha(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data = json_bytes(value)
    if path.exists():
        if path.read_bytes() != data: raise ValueError(f'Refusing to replace existing record: {path}')
        return
    temporary = path.with_suffix(path.suffix+'.tmp'); temporary.write_bytes(data); temporary.replace(path)


def glm_settings():
    config = yaml.safe_load((ROOT/'config/models.yaml').read_text())
    return ReasoningConfig.model_validate({**config,'model':'glm-5.3','timeout_seconds':600})


def review(request, output):
    request = json.loads(Path(request).read_bytes()); output = Path(output); output.mkdir(parents=True,exist_ok=True)
    agent = ImplementationReviewAgent(make_reasoning_backend,glm_settings(),output/'cache')
    result = agent.run(request,lambda value:validate_targets(value,TARGETS))
    save_json(output/'implementation_review.json',{'request_sha256':sha(request),'review':result.model_dump(mode='json')})
    save_json(output/'implementation_review_attempts.json',agent.attempts)
    for check in result.checks: print(check.hypothesis_id,check.verdict,check.human_summary,flush=True)
    return result


def dependencies():
    import numpy as np
    import scipy
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    from experiments.learned.history import HistoryConfig, run_history_seed
    from experiments.learned.replay import ReplayConfig, run_replay_seed
    from experiments.learned.consolidation import ConsolidationConfig, run_consolidation_seed
    return {'H006':(HistoryConfig,run_history_seed),'H009':(ReplayConfig,run_replay_seed),
            'H010':(ConsolidationConfig,run_consolidation_seed)}, {'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__}


def configurations():
    runners,_ = dependencies()
    return {'H006':asdict(runners['H006'][0]()),
            'H009':asdict(runners['H009'][0](initialization_updates=250,candidate_updates=60,calibration_blocks=128,null_blocks=128)),
            'H010':asdict(runners['H010'][0]())}


def validated_configuration(cls, values):
    # JSON turns tuples into lists; normalize to declared dataclass default types.
    defaults = cls()
    return cls(**{k:tuple(v) if isinstance(getattr(defaults,k),tuple) else v for k,v in values.items()})


def pilot(output, seed):
    if seed in range(101,121): raise ValueError('Pilot seeds must be separate from the reserved primary seeds')
    runners,versions = dependencies(); configs = configurations(); output = Path(output)
    for hid,(cls,runner) in runners.items():
        result = runner(seed,validated_configuration(cls,configs[hid]),output/hid/'weights')
        save_json(output/hid/f'pilot_{seed}.json',{'configuration':configs[hid],'versions':versions,'result':result})
        print(hid,'pilot CPU seconds',result['cpu_seconds'],flush=True)


def preregister(request_path, review_path, output):
    request = json.loads(Path(request_path).read_bytes()); review = json.loads(Path(review_path).read_bytes())
    if review['request_sha256'] != sha(request): raise ValueError('Review does not match the exact request')
    from .experiment_validation import ImplementationReview
    accepted = ImplementationReview.model_validate(review['review']);validate_targets(accepted,TARGETS)
    if any(c.verdict!='ready_for_exploratory_run' for c in accepted.checks):
        raise ValueError('Close every mandatory implementation repair before primary measurements')
    for name,content in request['new_experiment_source'].items():
        if (ROOT/name).read_text() != content: raise ValueError('Experiment code changed since the review')
    configs = configurations()
    if json.loads(json.dumps(configs)) != request['planned_configuration']:
        raise ValueError('Configurations changed since the implementation review')
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    sources = capture_source(ROOT)
    _,versions = dependencies()
    protocol = {'schema_version':1,'stage':'preregistered_before_primary_measurements',
        'targets':TARGETS,'primary_seeds':tuple(range(101,121)),
        'configurations':configs,'request_sha256':sha(request),'review_sha256':sha(review),
        'source_sha256':sha(sources),'versions':versions,
        'compute_unit':'CPU seconds measured inside single-thread learned experiment runners',
        'max_compute_units':1000.,'reproducibility_runs_count_against_budget':True,
        'scope':'finite local learned simulations, exploratory sample, no adequate-power or universal-mechanism claim',
        'seed_aggregation':'average nested streams within independent trained seed before comparisons',
        'primary_comparisons':[
            {'id':'H006_history_information','left':'normal','right':'shuffle','metric':'same_observation_separation',
             'minimum_effect':.20,'equivalence_margin':.10,'direction':'positive'},
            {'id':'H009_old_interface','left':'replay','right':'no_replay','metric':'critical_accuracy',
             'minimum_effect':.05,'equivalence_margin':.02,'direction':'positive'},
            {'id':'H010_old_retention','left':'state_machine','right':'no_protection','metric':'old_retention',
             'minimum_effect':.05,'equivalence_margin':.02,'direction':'positive'}],
        'primary_confidence':1-.05/3,'correction':'Bonferroni family of three primary comparisons; secondary intervals descriptive 95%',
        'validity_gates':{'H006':['context coverage >= .70 at reward probability .8 and 16 history steps','frozen parameter hashes identical'],
                         'H009':['update counts match before causal replay attribution','generator quality and removed probability mass matched before selective-coverage attribution',
                                 'stationarity during forward recalibration conditional, not given to algorithm','independent null and shadow alarms reported with finite-sample bounds'],
                         'H010':['no illegal joint gate transitions','actual lambda integral discrepancy limits timing attribution',
                                 'importance lesion correlation >= .30 before selective-protection attribution','endogenous warning must occur for protection applicability']}}
    save_json(output/'preregistration.json',protocol)
    save_json(output/'source_snapshot.json',sources)
    save_json(output/'implementation_review.json',review)
    save_json(output/'review_request.json',request)
    print('Preregistered',output,flush=True)


def verify_protocol(output):
    output = Path(output); protocol = json.loads((output/'preregistration.json').read_bytes())
    sources = json.loads((output/'source_snapshot.json').read_bytes())
    if sha(sources) != protocol['source_sha256']: raise ValueError('Source snapshot checksum mismatch')
    # Compare exact executable sources, including uncommitted modifications.
    if capture_source(ROOT) != sources: raise ValueError('Working source differs from preregistered source')
    _,versions = dependencies()
    if versions != protocol['versions']: raise ValueError('Dependency versions changed')
    return protocol


def _without_timing(value):
    if isinstance(value,dict): return {k:_without_timing(v) for k,v in value.items() if k!='cpu_seconds'}
    if isinstance(value,list): return [_without_timing(v) for v in value]
    return value


def run(output, *, replay=False):
    output = Path(output); protocol = verify_protocol(output); runners,_ = dependencies()
    elapsed = 0.; completed = []; costs = []
    for path in sorted((output/'raw').glob('*/*.json')):
        checkpoint = json.loads(path.read_bytes())
        if checkpoint['preregistration_sha256'] != sha(protocol): raise ValueError('Stale checkpoint protocol')
        elapsed += checkpoint['result']['cpu_seconds'];costs.append(checkpoint['result']['cpu_seconds'])
    replay_path = output/'reproducibility.json'
    if replay_path.exists():
        previous = json.loads(replay_path.read_bytes())
        elapsed += sum(r['cpu_seconds'] for r in previous['checks'])
        if replay: return previous
    for hid,(cls,runner) in runners.items():
        seeds = (protocol['primary_seeds'][0],) if replay else protocol['primary_seeds']
        for seed in seeds:
            raw_path = output/'raw'/hid/f'{seed}.json'
            if not replay and raw_path.exists(): continue
            if replay and not raw_path.exists(): raise ValueError('Replay requires existing raw result')
            reserve = 1.5*max(costs or [20.])
            if elapsed+reserve > protocol['max_compute_units']:
                raise RuntimeError('Measured compute reserve cannot fit; retain checkpoints and stop before starting another seed')
            print(hid,'replay' if replay else 'run','seed',seed,flush=True)
            result = runner(seed,validated_configuration(cls,protocol['configurations'][hid]),output/('replay_weights' if replay else 'weights')/hid)
            elapsed += result['cpu_seconds']; costs.append(result['cpu_seconds'])
            if replay:
                original = json.loads(raw_path.read_bytes())['result']
                matches = _without_timing(original) == _without_timing(result)
                completed.append({'hypothesis_id':hid,'seed':seed,'identical_measurements':matches,
                                  'cpu_seconds':result['cpu_seconds'],'method':'full deterministic learned runner executed again; timing excluded'})
                if not matches: raise ValueError('Full deterministic replay mismatch')
            else:
                save_json(raw_path,{'preregistration_sha256':sha(protocol),'result':result})
            if elapsed > protocol['max_compute_units']: raise RuntimeError('Actual measured budget exceeded; do not commit a completed run')
    if replay:
        save_json(replay_path,{'checks':completed,'total_measured_compute_units':elapsed})
    print('Measured compute units',elapsed,flush=True)


def numerical_analysis(output):
    import numpy as np
    from scipy.stats import poisson
    from experiments.learned.statistics import paired_interval, classification
    output = Path(output); protocol = verify_protocol(output); rows = {}
    costs = []; baseline = []
    for hid in TARGETS:
        records = []
        for seed in protocol['primary_seeds']:
            saved = json.loads((output/'raw'/hid/f'{seed}.json').read_bytes())
            if saved['preregistration_sha256'] != sha(protocol): raise ValueError('Result protocol differs')
            value = saved['result']; records.extend(value['measurements']);costs.append(value['cpu_seconds'])
            if hid == 'H009': baseline.append(value['baseline'])
        rows[hid] = records
    def per_seed(hid,arm,metric):
        selected = [r for r in rows[hid] if r['arm']==arm]
        if hid == 'H006': selected = [r for r in selected if r['reward_probability']==.8 and r['history_length']==16]
        return [float(np.mean([r[metric] for r in selected if r['seed']==seed])) for seed in protocol['primary_seeds']]
    comparisons = {}
    for comparison in protocol['primary_comparisons']:
        hid = comparison['id'][:4]
        interval = paired_interval(per_seed(hid,comparison['left'],comparison['metric']),
                                   per_seed(hid,comparison['right'],comparison['metric']),confidence=protocol['primary_confidence'])
        comparisons[comparison['id']] = {**comparison,**interval,'numerical_classification':classification(interval,
            direction=comparison['direction'],minimum_effect=comparison['minimum_effect'],equivalence_margin=comparison['equivalence_margin'])}
    for hid,left,right,metric in [('H006','normal','reward_shuffle','same_observation_separation'),
            ('H006','normal','swap','expected_error'),('H006','normal','single','expected_error'),
            ('H009','replay','critical_removed','critical_accuracy'),('H009','critical_removed','noncritical_removed','critical_accuracy'),
            ('H010','state_machine','fixed','new_reward_auc'),('H010','state_machine','random','new_reward_auc'),
            ('H010','state_machine','continuous','new_reward_auc'),('H010','state_machine','shuffled_importance','old_retention')]:
        name = f'{hid}_{left}_vs_{right}_{metric}'
        comparisons[name] = {**paired_interval(per_seed(hid,left,metric),per_seed(hid,right,metric)),
                            'hypothesis_id':hid,'left':left,'right':right,'metric':metric,'secondary_descriptive':True}
    null_alarms = sum(b['initial_null_alarms'] for b in baseline)
    shadow_alarms = sum(b['shadow_null_alarms'] for b in baseline)
    null_exposure = sum(b['null_observations_each'] for b in baseline)
    # Exact one-sided Poisson upper rate; zero failures is not proof of a zero rate.
    from scipy.stats import chi2
    null_upper = float(.5*chi2.ppf(.95,2*(null_alarms+1))/null_exposure*1000)
    shadow_upper = float(.5*chi2.ppf(.95,2*(shadow_alarms+1))/null_exposure*1000)
    matching = [a==b for a,b in zip(per_seed('H009','replay','optimization_steps'),per_seed('H009','no_replay','optimization_steps'))]
    integral = np.asarray(per_seed('H010','state_machine','lambda_integral'))
    nominal = np.asarray(per_seed('H010','state_machine','nominal_lambda_integral'))
    quality_critical = np.asarray(per_seed('H009','critical_removed','generated_self_reconstruction'))
    quality_noncritical = np.asarray(per_seed('H009','noncritical_removed','generated_self_reconstruction'))
    guards = {'H006':{'coverage_valid_all_seeds':all(v>=.70 for v in per_seed('H006','normal','minimum_context_greedy_accuracy'))},
        'H009':{'equal_update_budget_all_seeds':all(matching),'matched_seed_fraction':float(np.mean(matching)),
                'null_upper_alarms_per_1000':null_upper,'shadow_upper_alarms_per_1000':shadow_upper,
                'null_rate_below_original_target':bool(null_upper<=.01 and shadow_upper<=.01),
                'generation_quality_matched_all_seeds':bool(np.all(abs(quality_critical-quality_noncritical)/np.maximum(quality_noncritical,1e-8)<=.1)),
                'mean_promotions':float(np.mean(per_seed('H009','replay','promotions'))),
                'mean_unresolved':float(np.mean(per_seed('H009','replay','unresolved')))},
        'H010':{'illegal_joint_gate_transitions':float(np.sum(per_seed('H010','state_machine','illegal_joint_gate_transitions'))),
                'actual_integral_matches_nominal_all_seeds':bool(np.all(abs(integral-nominal)<=.02*nominal)),
                'mean_importance_lesion_correlation':float(np.mean(per_seed('H010','state_machine','importance_lesion_correlation'))),
                'protection_triggered_seed_fraction':float(np.mean(integral>0))}}
    analysis = {'preregistration_sha256':sha(protocol),'comparisons':comparisons,'validity_gates':guards,
                'per_seed_means':{hid:{arm:{metric:per_seed(hid,arm,metric) for metric in (
                   ('expected_error','same_observation_separation') if hid=='H006' else
                   ('critical_accuracy','optimization_steps','promotions') if hid=='H009' else
                   ('new_reward_auc','old_retention','lambda_integral'))}
                   for arm in sorted({r['arm'] for r in rows[hid] if 'hidden_switch' not in r['arm']})} for hid in TARGETS},
                'runner_cpu_seconds':float(sum(costs)),'sample_scope':'20 exploratory independent training seeds; nested flows averaged first'}
    save_json(output/'numerical_analysis.json',analysis)
    return analysis


def analyze(output):
    output = Path(output); numerical = numerical_analysis(output)
    request = json.loads((output/'review_request.json').read_bytes())
    agent = ResultAnalysisAgent(make_reasoning_backend,glm_settings(),output/'analysis_cache')
    inputs = {'hypotheses':request['hypotheses'],'preregistration':json.loads((output/'preregistration.json').read_bytes()),
              'computed_results':numerical,'implementation_review':json.loads((output/'implementation_review.json').read_bytes()),
              'reproducibility':json.loads((output/'reproducibility.json').read_bytes())}
    result = agent.run(inputs,lambda v:validate_targets(v,TARGETS,comparisons=numerical['comparisons']))
    save_json(output/'glm_result_analysis.json',result.model_dump(mode='json'))
    save_json(output/'glm_analysis_attempts.json',agent.attempts)
    report = ['# 学习型假设实验结果','','局部合成探索性实验；完整假设及开放环境结论另需验证。','']
    for row in result.interpretations:
        report.extend([f'## {row.hypothesis_id}：{row.verdict}','',row.human_summary,'',row.interpretation,'',
                       *['- 未满足前提：'+v for v in row.unmet_assumptions],*['- 限制：'+v for v in row.confounds],''])
    (output/'report.md').write_text('\n'.join(report))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('review','pilot','preregister','run','replay','analyze'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--review',type=Path)
    parser.add_argument('--seed',type=int,default=90)
    args = parser.parse_args(argv)
    if args.action=='review':
        if not args.request:parser.error('review requires --request')
        review(args.request,args.output)
    elif args.action=='pilot':pilot(args.output,args.seed)
    elif args.action=='preregister':
        if not args.request or not args.review:parser.error('preregister requires --request and --review')
        preregister(args.request,args.review,args.output)
    elif args.action in ('run','replay'):run(args.output,replay=args.action=='replay')
    else:analyze(args.output)


if __name__=='__main__': main()
