"""Read MLPerf EDU reports only after measurement and provenance verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .evidence import sanitize

REPORT_FIELDS = ('schema', 'lab', 'workload', 'workload_id', 'profile', 'status', 'result_scope',
                 'canonical_result', 'seed', 'device', 'model', 'data_mode', 'dataset_mode',
                 'preset', 'config', 'epochs', 'history', 'validation', 'functional_check',
                 'classroom_target', 'metrics', 'quality', 'quality_gate', 'quality_target',
                 'quality_target_passed', 'kv_cache_p50_speedup', 'comparison', 'results',
                 'measurement', 'parameters', 'dense', 'sparse')


def read_report(path: Path, *, manifest: Path | None = None) -> dict:
    """Preserve quality status, never promote a functional smoke to a score.

    Original files remain local. Digests identify them; uploaded evidence is a
    sanitized projection and therefore does not claim the original file hash.
    """
    raw = path.read_bytes()
    report = json.loads(raw)
    scope = report.get('result_scope')
    smoke = scope == 'functional-smoke' or report.get('data_mode') == 'synthetic'
    metrics = report.get('metrics', {})
    smoke = smoke or metrics.get('dataset_mode') == 'synthetic-smoke' or metrics.get('functional_smoke') is True
    # Tutorial 01 explicitly defines this one min profile as synthetic smoke.
    identity = str(report.get('workload_id', report.get('workload', '')))
    smoke = smoke or ('time-series-forecasting' in identity and report.get('profile') == 'min')
    result = {'result_kind': 'functional_smoke' if smoke else 'measured_experiment',
              'classification_basis': 'original report fields; no public-score claim',
              'report_sha256': hashlib.sha256(raw).hexdigest(),
              'report': sanitize({k: report[k] for k in REPORT_FIELDS if k in report}),
              'provenance': {'status': 'not_supplied'}}
    fingerprint = report.get('run_fingerprint', {})
    if fingerprint:
        # Software executable paths and free-form environment values can contain
        # local identities. Retain versions and hardware, not those paths.
        software = fingerprint.get('software', {})
        result['run_fingerprint'] = sanitize({
            'hardware': fingerprint.get('hardware', {}),
            'software': {k: software[k] for k in ('python', 'platform', 'torch', 'torchvision', 'transformers', 'mlperf_edu') if k in software},
            'comparison_fingerprint_sha256': fingerprint.get('comparison_fingerprint_sha256'),
        })
    if manifest is not None:
        checked = subprocess.run([sys.executable, '-m', 'mlperf_edu', 'verify', str(manifest.resolve())],
                                 capture_output=True, timeout=120)
        # Diagnostic output remains local. The verifier is invoked after every
        # measurement and before optional service initialization.
        if checked.returncode:
            raise ValueError('MLPerf provenance verification failed; evidence was not published')
        provenance = json.loads(manifest.read_text())
        result['provenance'] = {'status': 'verified', 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                'schema': provenance.get('schema'), 'merkle_root': provenance.get('merkle_root'),
                                'claim': 'content integrity, not producer authentication'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(read_report(args.report, manifest=args.manifest), indent=2) + '\n')


if __name__ == '__main__':
    main()
