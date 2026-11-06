"""Explicit, bounded local lesson execution with durable duplicate protection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .evidence import digest
from .training_evidence import printed_epoch_evidence


def execute_commands(runtime: Path, entry: dict, *, source_commit: str,
                     directory: Path | None = None) -> dict:
    """Execute trusted curriculum commands only when invoked by a Run button.

    These are local CPU commands, never hosted model/training API submissions.
    A claimed action is not automatically repeated after interruption.
    """
    identity = digest({"entry": entry, "source_commit": source_commit})
    directory = (directory or Path.cwd() / ".forge" / "executions") / identity
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    try:
        if os.name != "posix":
            raise ValueError("Bounded companion process groups require a POSIX runtime")
        with (directory / "claimed").open("x") as claim:
            claim.write(identity)
    except FileExistsError:
        return {"status": "interrupted_or_running", "action_id": identity,
                "message": "This action was already claimed. Inspect its local logs before starting a new experiment."}
    report = {"action_id": identity, "source_commit": source_commit,
              "source_path": entry["source_path"], "result_kind": entry["result_kind"],
              "seed": entry.get("seed"), "config": entry.get("config", {}),
              "units": {"elapsed_seconds": "s"}, "commands": [], "status": "running"}
    try:
        workspace = directory / "workspace"
        workspace.mkdir()
        for relative in entry["workspace_roots"]:
            source = runtime / relative
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
        env["VENV_PATH"] = sys.prefix  # TinyTorch's documented runtime selector.
        env["MPLBACKEND"] = "Agg"
        # Never inherit tracking auto-init defaults into the underlying lesson.
        env["WANDB_MODE"] = "disabled"
        cwd = workspace / entry["working_directory"]
        timeout = entry.get("timeout_seconds", 180)
        if not 0 < timeout <= 600:
            raise ValueError("Local companion commands require a timeout of at most 600 seconds")
        for index, args in enumerate(entry["commands"]):
            command = [sys.executable, *args]
            started = time.monotonic()
            with (directory / f"command-{index}.log").open("wb") as log:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    code = 124
            report["commands"].append({"argv": ["python", *args], "returncode": code,
                                       "elapsed_seconds": time.monotonic() - started})
            if entry.get("capture_printed_epochs"):
                report["commands"][-1]["training"] = printed_epoch_evidence(
                    (directory / f"command-{index}.log").read_bytes())
            if code:
                report["status"] = "timed_out" if code == 124 else "failed"
                break
        else:
            report["status"] = "passed"
    except Exception as error:  # optional companion failure is evidence, not a lesson mutation
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
    temporary = directory / "report.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(report_path)
    return report
