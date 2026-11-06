"""Training data, deterministic rewards, lineage and fail-closed job submission.

Current public APIs do not expose a job cost cap. A verified service-side
limit adapter is required for submission; a client timeout is not such a cap.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Callable

from .budget import Budget
from .hosted import messages
from .tool_task import TOOLS, digest, score, supervised_rows, tasks

TRAINING_ENDPOINT = 'https://forge.coreweave.com/api/training/v1'


def prepare_sft(directory: Path, source_commit: str) -> dict:
    if not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('Immutable source revision required')
    rows = supervised_rows()
    metadata = {'source_commit': source_commit, 'train_sha256': digest(tasks('train')),
                'evaluation_sha256': digest(tasks('evaluation')), 'num_trajectories': len(rows),
                'format': 'jsonl', 'result_kind': 'supplied_case',
                'provenance': 'synthetic exact tool selections; not model executions'}
    destination = directory / digest(metadata)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'trajectories.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return {**metadata, 'directory': str(destination), 'status': 'prepared; no upload or training'}


def rlvr_groups(reports: list[dict]) -> dict:
    """Use actual training-split rollouts, never held-out answers or invented calls."""
    if len(reports) != 2 or len({r['attempt'] for r in reports}) != 2:
        raise ValueError('Two distinct recorded rollout attempts are required')
    for report in reports:
        if report.get('split') != 'train' or report.get('dataset_sha256') != digest(tasks('train')):
            raise ValueError('RLVR accepts only the frozen training split')
        if report.get('result_kind') != 'hosted_model_execution':
            raise ValueError('Rewards require actual hosted rollouts')
    if reports[0]['config'] != reports[1]['config']:
        raise ValueError('Rollout groups must share one policy and generation configuration')
    indexed = [{r['task_id']: r for r in report['outputs']} for report in reports]
    groups, unavailable, uniform = [], [], []
    for task in tasks('train'):
        outputs = [index.get(task['id'], {}) for index in indexed]
        if not all(output.get('status') == 'completed' for output in outputs):
            unavailable.append(task['id'])
            continue
        rewards = [float(score(task, output)['exact_tool_and_arguments']) for output in outputs]
        if len(set(rewards)) == 1:
            uniform.append(task['id'])
        trajectories = []
        for report, output, reward in zip(reports, outputs, rewards):
            choices = output['response']['choices']
            if len(choices) != 1:
                raise ValueError('Expected one preserved completion choice per rollout')
            trajectories.append({'messages_and_choices': [*messages(task), choices[0]],
                                 'tools': TOOLS, 'reward': reward,
                                 'metadata': {'task_id': task['id'], 'attempt': report['attempt'],
                                              'response_sha256': digest(output['response'])}})
        groups.append({'trajectories': trajectories, 'metadata': {'task_id': task['id']}})
    return {'trajectory_groups': groups, 'unavailable_task_ids': unavailable,
            'uniform_reward_task_ids': uniform, 'train_sha256': digest(tasks('train')),
            'policy': reports[0]['config']['model'],
            'status': 'ready' if len(groups) > len(uniform) else 'no_reward_variation',
            'rollout_action_ids': [r['action_id'] for r in reports]}


def paired_comparison(before: dict, after: dict, *, checkpoint_lineage: dict) -> dict:
    for report in (before, after):
        if report.get('split') != 'evaluation' or report.get('dataset_sha256') != digest(tasks('evaluation')):
            raise ValueError('Before/after comparison requires the frozen held-out split')
    for key in ('max_tokens', 'temperature', 'endpoint'):
        if before['config'][key] != after['config'][key]:
            raise ValueError('Generation settings changed between comparison arms')
    if checkpoint_lineage.get('before_model') != before['config']['model'] or checkpoint_lineage.get('after_model') != after['config']['model']:
        raise ValueError('Checkpoint lineage must identify both evaluated model IDs')
    if not checkpoint_lineage.get('checkpoint_artifact') or not checkpoint_lineage.get('training_job_id'):
        raise ValueError('An actual checkpoint artifact and training job are required')
    arms = [{s['task_id']: s for s in report['scores']} for report in (before, after)]
    result = {'gained': [], 'lost': [], 'both_correct': [], 'both_incorrect': [], 'unavailable': []}
    for task in tasks('evaluation'):
        a, b = (arm.get(task['id'], {}) for arm in arms)
        if not a.get('available') or not b.get('available'):
            category = 'unavailable'
        else:
            category = {(False, True): 'gained', (True, False): 'lost',
                        (True, True): 'both_correct', (False, False): 'both_incorrect'}[
                            bool(a['exact_tool_and_arguments']), bool(b['exact_tool_and_arguments'])]
        result[category].append(task['id'])
    return {'pairs': result, 'total': 6, 'checkpoint_lineage': checkpoint_lineage,
            'before_action_id': before['action_id'], 'after_action_id': after['action_id'],
            'claim_scope': 'one synthetic plumbing smoke; no general quality-improvement claim'}


def submit_once(*, mode: str, model_id: str, payload: dict, project: str, attempt: str,
                budget: Budget, directory: Path,
                verify_service_limit: Callable | None = None) -> dict:
    """Submit an already prepared request only with a verifiable service-side cap.

    The adapter must read an enforceable account/job policy and return
    {maximum_usd, policy_reference, project, request_sha256}. This package does
    not pretend the current API's unsupported max_cost/max_duration fields work.
    Without an adapter no credentials are loaded, artifacts uploaded or job started.
    """
    if mode not in {'sft', 'rlvr'} or not re.fullmatch(r'[\w.-]+/[\w.-]+', project):
        raise ValueError('Explicit mode and entity/project required')
    request = {**payload, 'model_id': model_id}
    if verify_service_limit is None:
        raise RuntimeError('Training blocked: no verified service-enforced cost limit is available')
    bound = verify_service_limit(TRAINING_ENDPOINT, project, request)
    maximum = bound.get('maximum_usd')
    if (bound.get('project') != project or bound.get('request_sha256') != digest(request)
            or not bound.get('policy_reference') or not isinstance(maximum, (int, float))
            or not math.isfinite(maximum) or maximum <= 0):
        raise ValueError('Service limit evidence does not cover this exact request and project')
    action = 'training-' + digest({'mode': mode, 'project': project, 'attempt': attempt, 'request': request})
    previous = budget.result(action)
    if previous:
        return previous
    # Reserve before loading credentials or making any service write.
    if not budget.reserve(action, maximum, json.dumps(bound, sort_keys=True)):
        raise RuntimeError('Uncertain submission: inspect the saved job/request; do not resubmit')
    destination = directory / action
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'request.json').write_text(json.dumps({'request': request, 'bound': bound}, indent=2) + '\n')
    import requests
    import wandb
    api = wandb.Api(timeout=30)
    entity, name = project.split('/')
    info = api._service_api.execute_graphql('query($e:String!,$p:String!){project(name:$p,entityName:$e){access}}', {'e': entity, 'p': name})
    if info['project']['access'] != 'PRIVATE':
        raise ValueError('Private project required')
    path = '/preview/sft-training-jobs' if mode == 'sft' else '/preview/training-jobs'
    # requests defaults to no retries. A timeout is an uncertain write.
    response = requests.post(TRAINING_ENDPOINT + path, headers={'Authorization': 'Bearer ' + api.api_key},
                             json=request, timeout=30)
    response.raise_for_status()
    job = response.json()
    (destination / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
    result = {'action_id': action, 'job_id': job['id'], 'status': 'submitted',
              'request_sha256': digest(request), 'bound': bound, 'actual_billed_usd': None}
    budget.complete(action, result)
    return result


def monitor_job(job_id: str, directory: Path, *, deadline_seconds: int = 300) -> dict:
    if not re.fullmatch(r'[0-9a-f-]{36}', job_id) or not 1 <= deadline_seconds <= 600:
        raise ValueError('Explicit job UUID and bounded polling window required')
    import requests
    import wandb
    headers = {'Authorization': 'Bearer ' + wandb.Api(timeout=30).api_key}
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    index = 0
    while time.monotonic() - started < deadline_seconds:
        response = requests.get(TRAINING_ENDPOINT + '/preview/training-jobs/' + job_id, headers=headers, timeout=30)
        response.raise_for_status()
        status = response.json()
        (directory / f'{job_id}-{index:04d}.json').write_text(json.dumps(status, indent=2) + '\n')
        index += 1
        if status['status'] in {'COMPLETED', 'FAILED', 'CANCELLED'}:
            return {'status': status['status'], 'terminated': True, 'job': status}
        time.sleep(2)
    return {'status': 'polling_deadline', 'terminated': False, 'job_id': job_id,
            'message': 'The remote job may still be running; the full reservation remains charged.'}
