import json
import sys
from types import SimpleNamespace

import pytest

from mlsysbook_forge.bootstrap import locked_versions
from mlsysbook_forge.mlperf import read_report


def test_failed_quality_gate_and_smoke_scope_survive_projection(tmp_path):
    report = {'schema':'mlperf-edu-lab-result/0.1','result_scope':'functional-smoke',
              'status':'quality-failed','metrics':{'latency_ms': 1.25},
              'classroom_target':{'passed':False},'student_name':'private',
              'run_fingerprint':{'software':{'python':'3.13','python_executable':'/Users/student/python'}}}
    path = tmp_path / 'report.json'; path.write_text(json.dumps(report))
    evidence = read_report(path)
    assert evidence['result_kind'] == 'functional_smoke'
    assert evidence['report']['status'] == 'quality-failed'
    assert evidence['report']['metrics'] == report['metrics']
    assert evidence['report']['classroom_target']['passed'] is False
    assert '/Users/' not in json.dumps(evidence)
    assert 'student_name' not in json.dumps(evidence)


def test_provenance_failure_prevents_publication_payload(tmp_path, monkeypatch):
    path = tmp_path / 'report.json'; path.write_text('{}')
    manifest = tmp_path / 'report.provd.json'; manifest.write_text('{}')
    monkeypatch.setattr('mlsysbook_forge.mlperf.subprocess.run', lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match='verification failed'):
        read_report(path, manifest=manifest)


def test_cpu_lock_respects_platform_markers(tmp_path):
    path = tmp_path / 'requirements.lock'
    path.write_text("torch==2.13.0+cpu ; sys_platform == 'linux' \\\n    --hash=sha256:abc\ntorch==2.13.0 ; sys_platform != 'linux' \\\n    --hash=sha256:def\n")
    assert locked_versions(path)['torch'] == ('2.13.0+cpu' if sys.platform == 'linux' else '2.13.0')


def test_companion_preserves_scoped_wheel_sources_in_pep723(tmp_path, monkeypatch):
    pytest.importorskip('marimo')
    tomllib = pytest.importorskip('tomllib')
    from mlsysbook_forge import companions
    wheels = ['https://download.pytorch.org/whl/cpu/torch/',
              'https://download.pytorch.org/whl/cpu/torchvision/']
    lock = ''.join(f'--find-links {url}\n' for url in wheels) + 'marimo==0.25.1\n'
    monkeypatch.setattr(companions, 'runtime_bundle', lambda *a, **k: (b'', {'bundle_sha256': 'a' * 64}))
    monkeypatch.setattr(companions, 'read_source', lambda root, commit, path:
                        lock.encode() if path == 'requirements.lock' else b'print(1)\n')
    entry = {'id': 'test', 'title': 'Synthetic metadata check', 'reason': 'Test only',
             'workspace_roots': ['lesson.py'], 'dependency_lock': 'requirements.lock',
             'source_path': 'lesson.py', 'result_kind': 'functional_smoke',
             'working_directory': '.', 'commands': [['lesson.py']]}
    output = tmp_path / 'notebook.py'
    companions.build_companion(tmp_path, 'b' * 40, entry, output, ['marimo==0.25.1'])
    block = output.read_text().split('# ///')[1].removeprefix(' script\n')
    metadata = tomllib.loads('\n'.join(line.removeprefix('# ') for line in block.splitlines()))
    assert metadata['tool']['uv']['find-links'] == wheels
    assert 'extra-index-url' not in metadata['tool']['uv']


def test_complete_lock_has_hashes_for_every_platform_requirement():
    import re
    from pathlib import Path
    lock = (Path(__file__).parents[1] / 'mlperf-requirements.lock').read_text()
    blocks = re.split(r'(?m)(?=^[A-Za-z0-9_-]+==)', lock)[1:]
    assert blocks
    missing = [block.splitlines()[0] for block in blocks
               if not re.search(r'--hash=sha256:[0-9a-f]{64}(?:\s|$)', block)]
    assert missing == []


def test_original_tutorial_cells_are_preserved_and_tracking_failure_is_optional(tmp_path, monkeypatch):
    import ast
    import runpy
    from pathlib import Path
    from mlsysbook_forge.mlperf_notebook import instrument_tutorial
    pytest.importorskip('marimo')
    root = Path(__file__).parents[3]
    original = (root / 'mlperf-edu/tutorials/01_first_benchmark.py').read_text()
    source = instrument_tutorial(original, f'    _runtime = Path({str(tmp_path)!r})\n', 'a' * 40)
    old = [n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)]
    new = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    assert [ast.dump(n) for n in old[1:]] == [ast.dump(n) for n in new[1:-2]]
    calls = []
    report = {'metrics': {'loss': 1.25}, 'run_fingerprint': {'hardware': {'backend': 'CPU'}, 'software': {}}}
    def benchmark(path):
        calls.append(path)
        return {'stdout': 'Original workload completed\n', 'report': report,
                'report_path': tmp_path / 'report.json', 'manifest_path': tmp_path / 'result.provd.json'}
    monkeypatch.setitem(sys.modules, 'smoke_first_benchmark', SimpleNamespace(PROJECT_ROOT=tmp_path, run_tutorial_benchmark=benchmark))
    def tracking_failure(*a, **k):
        raise ValueError('Private diagnostic that must not be shown')
    monkeypatch.setattr('mlsysbook_forge.mlperf.read_report', tracking_failure)
    path = tmp_path / 'original-with-tracking.py'; path.write_text(source)
    _, definitions = runpy.run_path(str(path))['app'].run()
    assert calls == [(tmp_path / 'tutorials/_runs/01').resolve()]
    assert definitions['report'] == report
    assert definitions['forge_publish'] is None
    assert 'Run canonical companion' not in source
