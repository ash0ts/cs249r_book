import json
from pathlib import Path

from mlsysbook_forge.execution import execute_commands


def test_explicit_execution_preserves_original_and_reuses_receipt(tmp_path):
    runtime = tmp_path / 'runtime'
    (runtime / 'lesson').mkdir(parents=True)
    (runtime / 'lesson' / 'run.py').write_text("from pathlib import Path\nPath('output.txt').write_text('result')\n")
    entry = {'source_path': 'lesson/run.py', 'workspace_roots': ['lesson'],
             'working_directory': 'lesson', 'commands': [['run.py']],
             'result_kind': 'functional_smoke', 'timeout_seconds': 10}
    kwargs = {'source_commit': 'a' * 40, 'directory': tmp_path / 'runs'}
    first = execute_commands(runtime, entry, **kwargs)
    second = execute_commands(runtime, entry, **kwargs)
    assert first['status'] == 'passed'
    assert second == first
    assert first['action_id'] == second['action_id']
    assert first['commands'][0]['elapsed_seconds'] >= 0
    assert not (runtime / 'lesson' / 'output.txt').exists()
    assert len(list((tmp_path / 'runs').glob('*/report.json'))) == 1


def test_timeout_is_evidence_and_cannot_automatically_repeat(tmp_path):
    runtime = tmp_path / 'runtime'
    (runtime / 'lesson').mkdir(parents=True)
    entry = {'source_path': 'lesson', 'workspace_roots': ['lesson'],
             'working_directory': 'lesson', 'commands': [['-c', 'import time; time.sleep(10)']],
             'result_kind': 'functional_smoke', 'timeout_seconds': 0.05}
    report = execute_commands(runtime, entry, source_commit='b' * 40, directory=tmp_path / 'runs')
    assert report['status'] == 'timed_out'
    assert report['commands'][0]['returncode'] == 124
    assert json.loads(next((tmp_path / 'runs').glob('*/report.json')).read_text()) == report


def test_training_numbers_are_labeled_with_printed_precision():
    from mlsysbook_forge.training_evidence import printed_epoch_evidence
    log = b"Student reflection stays local\nEpoch  5/20  Loss: 0.1200  Train: 92.5%  Test: 88.0%\n"
    evidence = printed_epoch_evidence(log)
    assert evidence['epochs'][0]['line_number'] == 2
    assert evidence['epochs'][0]['test_accuracy'] == 88.0
    assert evidence['units']['test_accuracy'] == '%'
    assert evidence['precision'] == 'printed, not full precision'
    assert 'reflection' not in json.dumps(evidence)


def test_child_commands_find_tools_in_isolated_dependency_target(tmp_path, monkeypatch):
    dependency = tmp_path / 'dependencies'
    (dependency / 'bin').mkdir(parents=True)
    tool = dependency / 'bin' / 'forge-test-console-script'
    tool.write_text('#!/bin/sh\nexit 0\n')
    tool.chmod(0o755)
    monkeypatch.syspath_prepend(str(dependency))
    runtime = tmp_path / 'runtime'
    (runtime / 'lesson').mkdir(parents=True)
    entry = {'source_path': 'lesson', 'workspace_roots': ['lesson'],
             'working_directory': 'lesson', 'commands': [['-c',
                 "import subprocess; subprocess.run(['forge-test-console-script'], check=True)"]],
             'result_kind': 'functional_smoke', 'timeout_seconds': 10}
    report = execute_commands(runtime, entry, source_commit='c' * 40, directory=tmp_path / 'runs')
    assert report['status'] == 'passed'


def test_notebook_cache_is_inside_isolated_workspace(tmp_path):
    runtime = tmp_path / 'runtime'
    (runtime / 'lesson').mkdir(parents=True)
    entry = {'source_path': 'lesson', 'workspace_roots': ['lesson'],
             'working_directory': 'lesson', 'commands': [['-c',
                 "import os; from pathlib import Path; p=Path(os.environ['JUPYTER_DATA_DIR']); assert p.parent==Path.cwd().parent; p.mkdir(); (p/'marker').write_text('isolated')"]],
             'result_kind': 'functional_smoke', 'timeout_seconds': 10}
    assert execute_commands(runtime, entry, source_commit='d' * 40, directory=tmp_path / 'runs')['status'] == 'passed'
