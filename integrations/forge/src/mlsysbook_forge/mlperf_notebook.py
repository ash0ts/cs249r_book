"""Package the original MLPerf tutorial, with additive report instrumentation."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

from .packaging import MAX_SOURCE_BYTES, bootstrap_source, read_source, runtime_bundle


def instrument_tutorial(source: str, bootstrap: str, commit: str) -> str:
    """Keep all lesson cells and the original measurement call unchanged."""
    marker = '    from smoke_first_benchmark import PROJECT_ROOT, run_tutorial_benchmark\n'
    if source.count(marker) != 1:
        raise ValueError('Unrecognized original MLPerf tutorial bootstrap')
    setup = '    import sys\n    from pathlib import Path\n' + bootstrap
    setup += '''    for _root in ("mlperf-edu", "mlperf-edu/src", "mlperf-edu/tutorials"):
        sys.path.insert(0, str(_runtime / _root))
    # Both the canonical measurement and the post-measurement verifier launch
    # child Python processes; they need the same isolated dependency environment.
    import os as _forge_os
    _forge_os.environ["PYTHONPATH"] = _forge_os.pathsep.join(
        p for p in sys.path if p and Path(p).is_dir()
    )
'''
    result = source.replace(marker, setup + '\n' + marker)
    cells = f'''

@app.cell(hide_code=True)
def _(artifacts, mo):
    from mlsysbook_forge.mlperf import read_report as _read_report
    from mlsysbook_forge.notebook import capture_panel as _capture_panel
    try:
        _evidence = _read_report(artifacts["report_path"], manifest=artifacts["manifest_path"])
        _panel, forge_publish = _capture_panel(
            mo, lambda: _evidence, lab_id="mlperf-tutorial-01",
            source_commit={commit!r}, result_kind="functional_smoke",
        )
    except Exception as _error:
        # Tracking must not invalidate the completed original lesson or expose
        # SDK diagnostics that might contain credentials or signed URLs.
        forge_publish = None
        _panel = mo.md("Optional tracking unavailable (" + type(_error).__name__ +
                       "). Original benchmark results remain available above.")
    _panel
    return (forge_publish,)


@app.cell(hide_code=True)
def _(forge_publish, mo):
    mo.md(forge_publish.value if forge_publish is not None else "")
    return
'''
    guard = '\nif __name__ == "__main__":'
    if result.count(guard) != 1:
        raise ValueError('Unrecognized original MLPerf tutorial entry point')
    result = result.replace(guard, cells + guard)
    old = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    new = [n for n in ast.parse(result).body if isinstance(n, ast.FunctionDef)]
    if len(new) != len(old) + 2 or [ast.dump(n) for n in old[1:]] != [ast.dump(n) for n in new[1:-2]]:
        raise ValueError('Original lesson cells changed during instrumentation')
    if not all(any(ast.dump(n) == ast.dump(m) for m in new[0].body) for n in old[0].body):
        raise ValueError('Original imports changed during instrumentation')
    return result


def build_tutorial(root: Path, commit: str, entry: dict, output: Path, requirements: list[str],
                   *, asset_reference: str | None = None) -> dict:
    from .companions import normalize_marimo

    source = read_source(root, commit, entry['source_path'])
    payload, manifest = runtime_bundle(root, commit, (*entry['workspace_roots'], entry['dependency_lock']),
                                       roots=('integrations/forge/src/mlsysbook_forge',))
    manifest['dependency_lock'] = entry['dependency_lock']
    portable = instrument_tutorial(source.decode(), bootstrap_source(payload, manifest, asset_reference), commit)
    lock = read_source(root, commit, entry['dependency_lock']).decode()
    metadata = '# /// script\n# requires-python = ">=3.13,<3.15"\n# dependencies = [\n'
    metadata += ''.join(f'#   {json.dumps(dep)},\n' for dep in requirements) + '# ]\n'
    links = re.findall(r'^--find-links\s+(\S+)', lock, re.M)
    if links:
        metadata += '# [tool.uv]\n# find-links = ' + json.dumps(links) + '\n'
    portable = metadata + '# ///\n' + f'# Original notebook: {entry["source_path"]} @ {commit}\n' + portable
    if len(portable.encode()) > MAX_SOURCE_BYTES:
        raise ValueError('Use an immutable runtime artifact for this larger notebook bundle')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(portable)
    encoded = normalize_marimo(output)
    runtime = output.parent / 'runtime' / f'{manifest["bundle_sha256"]}.xz'
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_bytes(payload)
    receipt = {'lab_id': entry['id'], 'source_path': entry['source_path'], 'source_commit': commit,
               'canonical_sha256': hashlib.sha256(source).hexdigest(),
               'content_sha256': hashlib.sha256(encoded).hexdigest(), 'bundle_sha256': manifest['bundle_sha256'],
               'runtime_artifact': asset_reference, 'requirements': requirements, 'output': str(output),
               'original_cells_preserved': len([n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]),
               'additional_tracking_cells': 2, 'validation': 'built; not published or executed'}
    output.with_suffix('.manifest.json').write_text(json.dumps({**receipt, 'runtime': manifest}, indent=2) + '\n')
    return receipt
