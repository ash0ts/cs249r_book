"""Inventory canonical entrypoints without importing or executing their lessons."""
from __future__ import annotations

import re
from pathlib import Path

from .packaging import git, registered_entries


def inventory(root: Path, commit: str) -> list[dict]:
    registered = {entry["source_path"]: entry for entry in registered_entries(root)}
    paths = git(root, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
    rows = []
    for path in paths:
        if re.fullmatch(r"labs/vol[1-4]/lab_\d+_.*\.py", path):
            group, disposition, reason = "colabs", "hosted notebook", "Package the canonical Marimo source with an explicit saved-evidence panel."
        elif re.fullmatch(r"tinytorch/src/\d\d_[^/]+/\d\d_[^/]+\.py", path):
            group, disposition, reason = "tinytorch-modules", "companion notebook", "Preserve graded exports; execute and analyze in a separate companion."
        elif re.fullmatch(r"tinytorch/milestones/\d\d_[^/]+/\d\d_[^/]+\.py", path):
            group, disposition, reason = "tinytorch-milestones", "companion notebook", "Training/profiling companion; datasets remain external with source attribution."
        elif re.fullmatch(r"mlsysim/(docs/tutorials/[^/]+\.qmd|examples/[^/]+\.py)", path):
            group, disposition, reason = "mlsysim", "companion notebook", "Execute canonical analytical examples; preserve assumptions and units."
            if path.endswith("/index.qmd"):
                disposition, reason = "excluded", "Navigation page; executable tutorial pages are inventoried individually."
        elif re.fullmatch(r"mlperf-edu/(tutorials/[^/]+\.py|examples/lab[123]_[^/]+\.py)", path):
            group, disposition, reason = "mlperf", "companion notebook", "Publish report/provenance after measurement; label synthetic runs functional smoke."
            if "/tutorials/smoke_" in path:
                disposition, reason = "local/hardware-only", "CI smoke entrypoint validates its corresponding teaching notebook."
        elif path.startswith("kits/") and path.endswith(".qmd"):
            group, disposition, reason = "hardware", "local/hardware-only", "Device operations require the specified physical kit; companions analyze imported measurements."
        else:
            continue
        row = {"id": path.removesuffix(Path(path).suffix).replace("/", "-"), "source_path": path,
               "group": group, "disposition": disposition, "reason": reason,
               "implementation": "pending", "validation": "unvalidated"}
        if path in registered:
            row.update(registered[path])
            row["implementation"] = "registered"
        rows.append(row)
    unseen = set(registered) - set(paths)
    if unseen:
        raise ValueError(f"Registrations reference missing source paths: {sorted(unseen)}")
    return rows
