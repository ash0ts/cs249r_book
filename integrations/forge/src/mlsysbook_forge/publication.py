"""Private publication receipts and overwrite checks, independent of service SDKs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def content_digest(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def replacement_action(local: bytes, remote: bytes, previous: dict | None) -> str:
    """Never silently replace manual edits; compare saved source, not notebook name."""
    wanted, current = content_digest(local), content_digest(remote)
    if wanted == current:
        return "reuse"
    if previous is None or current != previous["content_sha256"]:
        raise ValueError("Remote source has unrecognized/manual edits; preserve it and create a copy")
    return "update"


def save_receipt(path: Path, *, build: dict, notebook_id: str, notebook_url: str,
                 saved_source: bytes, committed_version: str, rendered: bool,
                 reopened: bool, execution: str) -> dict:
    if content_digest(saved_source) != build["content_sha256"]:
        raise ValueError("Saved notebook source does not match the built source")
    if not notebook_id or not committed_version or not rendered or not reopened:
        raise ValueError("Publication requires committed source, rendered output and reopening")
    if execution not in {"passed", "failed", "unvalidated"}:
        raise ValueError("Execution validation must be explicit")
    receipt = {**build, "notebook_id": notebook_id, "notebook_url": notebook_url,
               "committed_version": committed_version, "rendered": rendered,
               "reopened": reopened, "execution": execution,
               "validation": f"publication verified; execution {execution}",
               "verified_at": datetime.now(timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    path.chmod(0o600)
    return receipt
