"""Marimo companions for canonical CLI workflows; originals remain unchanged."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from .packaging import MAX_SOURCE_BYTES, bootstrap_source, read_source, runtime_bundle


def execution_cells(entry: dict, commit: str) -> str:
    if entry.get('generator') == 'training-sft':
        return f'''
@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path
    from mlsysbook_forge.training import prepare_sft
    forge_execute = mo.ui.button(value=None, label="Prepare local SFT dataset", on_click=lambda _: prepare_sft(Path(".forge/training"), {commit!r}))
    mo.vstack([mo.md("Prepare 24 frozen training demonstrations locally. Job submission remains blocked until a service-enforced cost limit can be verified. See TRAINING.md in the bundled source."), forge_execute])
    return (forge_execute,)
'''
    if entry.get('generator') == 'training-rlvr':
        return '''
@app.cell(hide_code=True)
def _(mo):
    forge_rollouts = mo.ui.file(filetypes=[".json"], multiple=True)
    mo.vstack([mo.md("Import two actual training-split rollout reports from distinct, separately budgeted attempts. Held-out evaluation reports are rejected. No model or training job runs on load."), forge_rollouts])
    return (forge_rollouts,)


@app.cell(hide_code=True)
def _(mo, forge_rollouts):
    from mlsysbook_forge.training import rlvr_groups
    def _prepare(_):
        import json
        from pathlib import Path
        try:
            _files = forge_rollouts.value
            if len(_files) != 2 or any(len(_file.contents) > 2 * 1024 * 1024 for _file in _files):
                raise ValueError("Choose exactly two reports, each at most 2 MiB")
            _result = rlvr_groups([json.loads(_file.contents) for _file in _files])
            Path(".forge").mkdir(exist_ok=True)
            Path(".forge/rlvr-groups.json").write_text(json.dumps(_result, indent=2))
            return {"status": _result["status"], "groups": len(_result["trajectory_groups"]), "uniform_reward_task_ids": _result["uniform_reward_task_ids"], "unavailable_task_ids": _result["unavailable_task_ids"], "train_sha256": _result["train_sha256"], "result_kind": "hosted_model_execution", "provenance": "analysis of imported actual rollout reports; no new execution"}
        except (ValueError, KeyError, TypeError) as _error:
            return {"status": "invalid reports", "error_type": type(_error).__name__}
    forge_execute = mo.ui.button(value=None, label="Prepare deterministic RLVR rewards", on_click=_prepare, disabled=len(forge_rollouts.value) != 2)
    forge_execute
    return (forge_execute,)
'''
    if entry.get('generator') == 'hosted':
        template = {"endpoint": "https://api.inference.wandb.ai/v1", "project": "", "model": "",
                    "max_tokens": 256, "temperature": 0, "input_usd_per_million": None,
                    "output_usd_per_million": None, "pricing_source": "https://coreweave.com/forge-pricing/inference-agents",
                    "pricing_checked": "", "source_commit": commit}
        return f'''
@app.cell(hide_code=True)
def _(mo):
    import json
    forge_config = mo.ui.text_area(value={json.dumps(template, indent=2)!r}, label="Explicit endpoint, model, private project and verified rates", rows=16)
    forge_attempt = mo.ui.text(value="smoke-1", label="Attempt identifier (reuse prevents duplicate calls)")
    forge_budget = mo.ui.text(value=".forge/campaign-budget.sqlite", label="Shared campaign budget database")
    mo.vstack([mo.md("The campaign ledger must include notebook compute, retries, storage and training allowances. Never reset it to obtain more budget. This button makes six paid model calls after checking the private project and model catalog."), forge_config, forge_attempt, forge_budget])
    return forge_config, forge_attempt, forge_budget


@app.cell(hide_code=True)
def _(mo, forge_config, forge_attempt, forge_budget):
    from mlsysbook_forge.hosted import run_evaluation
    def _execute(_):
        import json
        from pathlib import Path
        try:
            return run_evaluation(json.loads(forge_config.value), budget_path=Path(forge_budget.value), output_dir=Path(".forge/hosted"), attempt=forge_attempt.value)
        except Exception as _error:
            return {{"status": "failed", "error_type": type(_error).__name__, "message": "Preflight or service failure. Local evidence and reservations are retained; inspect before a new attempt."}}
    forge_execute = mo.ui.button(value=None, label="Run bounded hosted evaluation", on_click=_execute)
    forge_execute
    return (forge_execute,)
'''
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
    if indexes:
        source += '# [tool.uv]\n# extra-index-url = ' + json.dumps(indexes) + '\n'
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
