"""Marimo companions for canonical CLI workflows; originals remain unchanged."""
from __future__ import annotations

import ast
import hashlib
from importlib import metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .packaging import MAX_SOURCE_BYTES, bootstrap_source, read_source, runtime_bundle


def normalize_marimo(path: Path) -> bytes:
    """Match the pinned editor's saved format before computing its digest.

    Marimo canonicalizes cell signatures, unused returns and Markdown layout.
    The check command never executes lesson cells; only safe fixes are enabled.
    """
    if metadata.version("marimo") != "0.25.1":
        raise RuntimeError("Build companions with the pinned marimo==0.25.1 environment")
    process = subprocess.run(
        [sys.executable, "-m", "marimo", "check", "--fix", str(path)],
        env={**os.environ, "MARIMO_MANAGE_SCRIPT_METADATA": "false"},
        capture_output=True, timeout=60,
    )
    if process.returncode:
        raise ValueError("Generated companion failed Marimo's source checks: "
                         + process.stdout.decode(errors="replace")
                         + process.stderr.decode(errors="replace"))
    return path.read_bytes()


def execution_cells(entry: dict, commit: str) -> str:
    if entry.get('generator') == 'hardware':
        return f'''
@app.cell(hide_code=True)
def _(mo):
    forge_input = mo.ui.file(filetypes=[".csv", ".json"], multiple=False)
    forge_kind = mo.ui.dropdown({{"Dataset split manifest": "dataset", "Model parameter manifest": "model", "Device measurements": "measurements"}}, value="Dataset split manifest", label="Evidence to analyze")
    mo.vstack([forge_kind, forge_input])
    return forge_input, forge_kind


@app.cell(hide_code=True)
def _(mo, forge_input, forge_kind):
    from mlsysbook_forge.hardware import analyze
    def _execute(_):
        try:
            return analyze(forge_input.contents(), forge_kind.value, platform={entry['platform']!r})
        except Exception as _error:
            return {{"status": "failed", "error_type": type(_error).__name__, "message": str(_error)}}
    forge_execute = mo.ui.button(value=None, label="Analyze imported evidence", on_click=_execute, disabled=not forge_input.value)
    forge_execute
    return (forge_execute,)
'''
    return f'''
@app.cell(hide_code=True)
def _(mo, forge_runtime):
    from mlsysbook_forge.execution import execute_commands
    def _execute(_):
        return execute_commands(forge_runtime, {entry!r}, source_commit={commit!r})
    forge_execute = mo.ui.button(value=None, label="Run canonical companion", on_click=_execute)
    forge_execute
    return (forge_execute,)
'''


def build_companion(root: Path, commit: str, entry: dict, output: Path, requirements: list[str],
                    *, asset_reference: str | None = None) -> dict:
    if entry.get("generator") == "mlperf-marimo":
        from .mlperf_notebook import build_tutorial
        return build_tutorial(root, commit, entry, output, requirements, asset_reference=asset_reference)
    extras = (*entry["workspace_roots"], entry["dependency_lock"])
    payload, manifest = runtime_bundle(root, commit, extras,
                                      roots=("integrations/forge/src/mlsysbook_forge",))
    manifest["dependency_lock"] = entry["dependency_lock"]
    bootstrap = bootstrap_source(payload, manifest, asset_reference)
    introduction = (f"# {entry['title']}\n\n{entry['reason']}\n\n"
                    f"Canonical source: `{entry['source_path']}` at `{commit}`.\n\n"
                    "Run the companion explicitly. It works in a separate copy and does not change "
                    "the graded notebook or canonical source. Its elapsed times include process overhead "
                    "and are not benchmark scores. Logs stay local; publishing evidence is a separate action.")
    source = '# /// script\n# requires-python = ">=3.13,<3.15"\n# dependencies = [\n'
    source += ''.join(f'#   {json.dumps(dep)},\n' for dep in requirements)
    source += '# ]\n'
    lock = read_source(root, commit, entry["dependency_lock"]).decode()
    indexes = re.findall(r'^--extra-index-url\s+(\S+)', lock, re.M)
    wheel_links = re.findall(r'^--find-links\s+(\S+)', lock, re.M)
    if indexes or wheel_links:
        source += '# [tool.uv]\n'
        if indexes:
            source += '# extra-index-url = ' + json.dumps(indexes) + '\n'
        if wheel_links:
            source += '# find-links = ' + json.dumps(wheel_links) + '\n'
    source += '# ///\n'
    source += f'''import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title={entry['title']!r})


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    import marimo as mo
{bootstrap}
    forge_runtime = _runtime
    return mo, forge_runtime


@app.cell(hide_code=True)
def _(mo):
    mo.md({introduction!r})
    return


{execution_cells(entry, commit)}


@app.cell(hide_code=True)
def _(mo, forge_execute):
    mo.json(forge_execute.value or {{"status": "not run"}})
    return


@app.cell(hide_code=True)
def _(mo, forge_execute):
    from mlsysbook_forge.notebook import capture_panel
    _panel, forge_publish = capture_panel(mo, lambda: forge_execute.value,
        lab_id={entry['id']!r}, source_commit={commit!r}, result_kind=(forge_execute.value or {{}}).get("result_kind", {entry['result_kind']!r}))
    _panel
    return (forge_publish,)


@app.cell(hide_code=True)
def _(mo, forge_publish):
    mo.md(forge_publish.value)
    return


if __name__ == "__main__":
    app.run()
'''
    ast.parse(source)
    encoded = source.encode()
    if len(encoded) > MAX_SOURCE_BYTES:
        raise ValueError("Use an immutable private runtime artifact for this larger companion bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    encoded = normalize_marimo(output)
    if len(encoded) > MAX_SOURCE_BYTES:
        raise ValueError("Normalized companion exceeds the hosted source size limit")
    runtime_path = output.parent / "runtime" / f"{manifest['bundle_sha256']}.xz"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_bytes(payload)
    source_bytes = read_source(root, commit, entry["source_path"])
    receipt = {"lab_id": entry["id"], "source_path": entry["source_path"], "source_commit": commit,
               "canonical_sha256": hashlib.sha256(source_bytes).hexdigest(),
               "content_sha256": hashlib.sha256(encoded).hexdigest(), "bundle_sha256": manifest["bundle_sha256"],
               "runtime_artifact": asset_reference, "requirements": requirements, "output": str(output),
               "validation": "built; not published or executed"}
    output.with_suffix(".manifest.json").write_text(json.dumps({**receipt, "runtime": manifest}, indent=2) + "\n")
    return receipt
