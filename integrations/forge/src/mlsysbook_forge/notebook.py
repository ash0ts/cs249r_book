"""Explicit-action widgets for generated copies; no Marimo dependency on import."""
from __future__ import annotations

import json
from importlib import metadata

from .adapters import publish_capture
from .evidence import digest, sanitize
from .session import ForgeSession, ResultKind, RunContext


def capture_panel(mo, getter, *, lab_id: str, source_commit: str, result_kind: str):
    """Only a click publishes the saved snapshot. Reactive evaluation only renders."""
    saved = sanitize(getter())
    mode = mo.ui.dropdown(["disabled", "offline", "online"], value="disabled", label="Tracking mode")
    entity = mo.ui.text(label="Your Forge entity")
    project = mo.ui.text(label="Your private project")

    def publish(_):
        try:
            # Identity is derived from immutable evidence, not a cell execution.
            versions = {}
            for package in ("marimo", "numpy", "pydantic", "pint", "wandb"):
                try:
                    versions[package] = metadata.version(package)
                except metadata.PackageNotFoundError:
                    pass
            context = RunContext(lab_id, source_commit, ResultKind(result_kind), versions=versions)
            clean = sanitize(getter())
            if not clean:
                return "Capture a comparison in the lesson first."
            identity = digest({"context": sanitize(context), "evidence": clean,
                               "mode": mode.value, "entity": entity.value, "project": project.value})[:40]
            with ForgeSession(context, mode=mode.value, session_id=identity,
                              entity=entity.value or None, project=project.value or None) as session:
                receipt = publish_capture(session, clean)
            return json.dumps(receipt, indent=2)
        except Exception as error:  # noqa: BLE001 - optional tracking must not break the lesson
            # Do not render SDK messages that can contain signed URLs/credentials.
            return f"Publishing failed ({type(error).__name__}). Your lesson evidence is unchanged."

    button = mo.ui.button(value="Nothing published.", label="Publish saved comparisons", on_click=publish, disabled=not saved)
    # The local preview is exactly the filtered payload that will be uploaded.
    # Student identity, interpretations and credentials are excluded by default.
    return mo.vstack([
        mo.md("### Optional Forge evidence\nSaved comparisons retain their original result provenance. "
              "Analytical and supplied results are not device measurements or model executions."),
        mo.hstack([mode, entity, project]), button,
        mo.accordion({"Preview evidence": mo.json(saved)}),
    ]), button
