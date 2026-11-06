import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlsysbook_forge.budget import Budget
from mlsysbook_forge.hosted import INFERENCE_ENDPOINT, run_evaluation, validate_config
from mlsysbook_forge.tool_task import execute_tool, score, supervised_rows, tasks


def config():
    return dict(endpoint=INFERENCE_ENDPOINT, project='test/private', model='synthetic-test-model',
                source_commit='a' * 40, max_tokens=256, temperature=0,
                input_usd_per_million=.1, output_usd_per_million=.15,
                pricing_source='https://coreweave.com/forge-pricing/inference-agents', pricing_checked='2026-11-06')


def test_splits_and_exact_checker():
    train, evaluation = tasks('train'), tasks('evaluation')
    assert len(train) == 24 and len(evaluation) == 6
    assert not {r['id'] for r in train} & {r['id'] for r in evaluation}
    assert not {r['prompt'] for r in train} & {r['prompt'] for r in evaluation}
    assert len(supervised_rows()) == 24
    for row in train + evaluation:
        assert score(row, {'status': 'completed', 'selection': row['expected']})['exact_tool_and_arguments']
        assert not score(row, {'status': 'service_error'})['available']
        assert execute_tool(**row['expected']) is not None
    with pytest.raises(ValueError):
        execute_tool('add', {'a': True, 'b': 2})
    with pytest.raises(ValueError):
        execute_tool('__import__', {'module': 'os'})


def test_budget_shared_ceiling_and_uncertainty(tmp_path):
    budget = Budget(tmp_path / 'budget.db', ceiling_usd=100)
    assert budget.reserve('notebooks', 99, 'enforced lifetime and resource allowance')
    assert not budget.reserve('notebooks', 99, 'enforced lifetime and resource allowance')
    with pytest.raises(ValueError, match='ceiling'):
        Budget(tmp_path / 'budget.db', ceiling_usd=100).reserve('training', 2, 'bound')
    with pytest.raises(ValueError):
        budget.reserve('unbounded', float('nan'), 'unknown')
    assert budget.result('notebooks') is None
    budget.complete('notebooks', {'terminated': True, 'actual_billed_usd': None})
    assert budget.summary()['reserved_usd'] == 99
    with pytest.raises(ValueError):
        budget.complete('notebooks', {})


def test_no_credentials_to_other_endpoints():
    invalid = config() | {'endpoint': 'https://untrusted.example/v1'}
    with pytest.raises(ValueError, match='endpoint'):
        validate_config(invalid)


def test_hosted_failure_preserved_and_reexecution_deduplicated(tmp_path, monkeypatch):
    import sys
    calls = []
    class API:
        api_key = 'TEST_ONLY'
        def __init__(self, **kwargs):
            self._service_api = self
        def execute_graphql(self, *args):
            return {'project': {'access': 'PRIVATE'}}
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(model_dump=lambda: {'data': [{'id': config()['model']}]}))
            self.chat = SimpleNamespace(completions=self)
        def create(self, **kwargs):
            calls.append(kwargs)
            raise TimeoutError('remote result is unknown')
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(Api=API))
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=Client))
    options = dict(budget_path=tmp_path / 'budget.db', output_dir=tmp_path / 'outputs', attempt='smoke', trace=False)
    result = run_evaluation(config(), **options)
    assert result['available'] == result['correct'] == 0
    assert result['total'] == len(calls) == 6
    assert result['actual_billed_usd'] is None
    assert len(list((tmp_path / 'outputs').glob('*/evaluation-*.json'))) == 6
    assert run_evaluation(config(), **options) == result
    assert len(calls) == 6


def test_heterogeneous_tool_results_do_not_break_native_summary():
    pytest.importorskip('weave')
    from weave.flow.scorer import auto_summarize
    records = [{'tool_result_json': json.dumps(execute_tool(**row['expected']))}
               for row in tasks('evaluation')]
    assert auto_summarize(records) is None


def test_verified_usage_reconciliation_preserves_original_allowance(tmp_path):
    budget = Budget(tmp_path / 'budget.db', ceiling_usd=100)
    budget.reserve('notebook', 99, 'enforced maximum lifetime')
    with pytest.raises(ValueError):
        budget.reconcile('notebook', maximum_usd=1, evidence={'kind': 'estimate'})
    budget.reconcile('notebook', maximum_usd=1, evidence={'kind': 'terminated_resource_usage',
                     'terminated': True, 'maximum_usd': 1, 'source_reference': 'test-only provider receipt'})
    budget.reserve('next-notebook', 2, 'enforced maximum lifetime')
    assert budget.summary()['reserved_usd'] == 3
    assert budget.summary()['original_reservations_usd'] == 101
    assert budget.summary()['actual_billed_usd'] is None
