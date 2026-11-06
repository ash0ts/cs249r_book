"""Check registrations against canonical metadata and source contracts."""
import ast
import json
from pathlib import Path

import pytest

from mlsysbook_forge.packaging import portable_marimo, registered_entries, tracking_cells

ROOT = Path(__file__).resolve().parents[3]
ENTRIES = [e for e in registered_entries(ROOT) if e.get("group") == "colabs"]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["id"])
def test_canonical_cells_are_preserved(entry):
    source = (ROOT / entry["source_path"]).read_text()
    original = ast.parse(source)
    portable = ast.parse(portable_marimo(source, "    # hosted bootstrap\n"))
    # Every teaching cell after the native bootstrap is preserved byte for byte
    # at the AST level; changing its math cannot be hidden by the builder.
    assert [ast.dump(n) for n in original.body[4:]] == [ast.dump(n) for n in portable.body[4:]]
    assert entry["capture_getter"] in {n.id for n in ast.walk(original) if isinstance(n, ast.Name)}
    ast.parse(tracking_cells(entry, "a" * 40))
    assert entry["result_kind"] in {"analytical_simulation", "supplied_case"}


def test_each_registered_volume_is_complete():
    for volume, expected in [(1, 17), (2, 18), (3, 21), (4, 19)]:
        manifest = ROOT / f"integrations/forge/catalog/vol{volume}.json"
        if not manifest.exists():
            continue
        entries = json.loads(manifest.read_text())["entries"]
        sources = {str(p.relative_to(ROOT)) for p in (ROOT / f"labs/vol{volume}").glob("lab_*.py")}
        assert len(entries) == expected
        assert {entry["source_path"] for entry in entries} == sources
