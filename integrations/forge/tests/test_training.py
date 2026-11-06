import json
from pathlib import Path

import pytest

from mlsysbook_forge.budget import Budget
from mlsysbook_forge.tool_task import digest, tasks
from mlsysbook_forge.training import paired_comparison, prepare_sft, rlvr_groups, submit_once


def test_sft_contains_only_frozen_training_rows(tmp_path):
    result = prepare_sft(tmp_path, 'a' * 40)
    rows = [json.loads(line) for line in (Path(result['directory']) / 'trajectories.jsonl').read_text().splitlines()]
    assert len(rows) == 24
    prompts = {row['messages'][1]['content'] for row in rows}
    assert prompts == {task['prompt'] for task in tasks('train')}
    assert not prompts & {task['prompt'] for task in tasks('evaluation')}
    assert prepare_sft(tmp_path, 'a' * 40) == result


def test_submission_without_enforceable_limit_does_not_reserve_or_create_files(tmp_path):
    budget = Budget(tmp_path / 'budget.sqlite', ceiling_usd=100)
    with pytest.raises(RuntimeError, match='service-enforced'):
        submit_once(mode='sft', model_id='existing-model', payload={}, project='test/private', attempt='one',
                    budget=budget, directory=tmp_path / 'jobs')
    assert budget.summary()['reserved_usd'] == 0
    assert not (tmp_path / 'jobs').exists()


def test_training_limit_must_cover_exact_request(tmp_path):
    budget = Budget(tmp_path / 'budget.sqlite', ceiling_usd=100)
    with pytest.raises(ValueError, match='exact request'):
        submit_once(mode='sft', model_id='existing-model', payload={}, project='test/private', attempt='one',
                    budget=budget, directory=tmp_path / 'jobs', verify_service_limit=lambda *a: {
                        'project': 'test/private', 'maximum_usd': 1, 'policy_reference': 'test-only', 'request_sha256': 'wrong'})
    assert budget.summary()['reserved_usd'] == 0


def test_rlvr_rejects_evaluation_leakage_and_uniform_rewards():
    reports = [{'attempt': str(i), 'split': 'evaluation', 'dataset_sha256': digest(tasks('evaluation'))}
               for i in range(2)]
    with pytest.raises(ValueError, match='training split'):
        rlvr_groups(reports)
    for report in reports:
        report.update(split='train', dataset_sha256=digest(tasks('train')), result_kind='hosted_model_execution',
                      config={'model': 'test-policy'}, action_id='test-only', outputs=[{
                          'task_id': t['id'], 'status': 'completed', 'selection': t['expected'],
                          'response': {'choices': [{'message': {'role': 'assistant', 'content': 'fixture'}}]}}
                          for t in tasks('train')])
    result = rlvr_groups(reports)
    assert result['status'] == 'no_reward_variation'
    assert len(result['uniform_reward_task_ids']) == 24


def test_before_after_preserves_missing_pairs():
    base = {'split': 'evaluation', 'dataset_sha256': digest(tasks('evaluation')), 'action_id': 'test-only',
            'config': {'model': 'base', 'temperature': 0, 'max_tokens': 256, 'endpoint': 'test'},
            'scores': [{'task_id': t['id'], 'available': True, 'exact_tool_and_arguments': True}
                       for t in tasks('evaluation')]}
    after = {**base, 'config': {**base['config'], 'model': 'checkpoint'}, 'scores': []}
    result = paired_comparison(base, after, checkpoint_lineage={'before_model': 'base', 'after_model': 'checkpoint',
        'checkpoint_artifact': 'test-only:v0', 'training_job_id': 'test-only'})
    assert len(result['pairs']['unavailable']) == result['total'] == 6
    assert not result['pairs']['lost']
