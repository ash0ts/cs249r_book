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
