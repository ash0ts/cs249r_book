import hashlib
import io
import json
import lzma
import subprocess
import textwrap
import zipfile

import pytest
from mlsysbook_forge.packaging import bootstrap_source, portable_marimo
from mlsysbook_forge.publication import replacement_action, save_receipt


def test_standalone_bootstrap_has_no_checkout_dependency(tmp_path):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as z:
        z.writestr("labs/example.py", "VALUE = 17\n")
        z.writestr("integrations/forge/src/mlsysbook_forge/bootstrap.py", "def ensure_dependencies(path):\n    pass\n")
    payload = lzma.compress(output.getvalue())
    boot = textwrap.dedent(bootstrap_source(payload, {"bundle_sha256": hashlib.sha256(payload).hexdigest()}))
    script = "from pathlib import Path\nimport sys\n" + boot + "\nimport example\nassert example.VALUE == 17\n"
    # Re-executions reuse the same source tree with no server dependency.
    for _ in range(2):
        subprocess.run(["python3", "-I", "-c", script], cwd=tmp_path, check=True)
    assert len(list((tmp_path / ".forge/runtime").iterdir())) == 1


def test_unknown_bootstrap_rejected_and_math_preserved():
    source = '''import sys
if sys.platform == "emscripten":
    pass
else:
    import local_only
result = 19 * 7
'''
    result = portable_marimo(source, "# bundled runtime\n")
    assert result.endswith("result = 19 * 7\n")
    assert "local_only" not in result
    with pytest.raises(ValueError, match="Unrecognized"):
        portable_marimo("result = 3", "")


def test_receipt_requires_persisted_source_and_reopen(tmp_path):
    source = b"print(1)"
    build = {"content_sha256": hashlib.sha256(source).hexdigest(),
             "validation": "built; not published or executed"}
    kwargs = {"build": build, "notebook_id": "n", "notebook_url": "https://example.test/n", "saved_source": source,
                  "committed_version": "v1", "rendered": True, "reopened": True, "execution": "passed"}
    receipt = save_receipt(tmp_path / "receipt.json", **kwargs)
    assert json.loads((tmp_path / "receipt.json").read_text()) == receipt
    assert receipt['validation'] == 'publication verified; execution passed'
    with pytest.raises(ValueError, match="committed"):
        save_receipt(tmp_path / "bad.json", **{**kwargs, "reopened": False})
    with pytest.raises(ValueError, match="does not match"):
        save_receipt(tmp_path / "bad.json", **{**kwargs, "saved_source": b"different"})
    assert replacement_action(source, source, None) == "reuse"
    assert replacement_action(b"updated", source, build) == "update"
    with pytest.raises(ValueError, match="manual edits"):
        replacement_action(b"updated", b"user edit", build)


def test_empty_lesson_evidence_renders_without_publishing(tmp_path, monkeypatch):
    mo = pytest.importorskip("marimo")
    from mlsysbook_forge.notebook import capture_panel
    monkeypatch.chdir(tmp_path)
    panel, button = capture_panel(mo, lambda: None, lab_id="empty", source_commit="a" * 40,
                                  result_kind="supplied_case")
    assert "Optional Forge evidence" in panel.text
    assert button.value == "Nothing published."
    assert not (tmp_path / ".forge").exists()


def test_hosted_asset_must_be_immutable_and_verified():
    manifest = {"bundle_sha256": "a" * 64}
    with pytest.raises(ValueError, match="pin"):
        bootstrap_source(b"payload", manifest, "owner/project/runtime:latest")
    source = bootstrap_source(b"payload", manifest, "owner/project/runtime:v0")
    assert "owner/project/runtime:v0" in source
    assert "checksum mismatch" in source
    assert "WANDB_API_KEY" not in source


def test_dependency_metadata_normalizes_dots_underscores_and_dashes(tmp_path, monkeypatch):
    import hashlib
    import platform
    import sys
    from types import SimpleNamespace
    from mlsysbook_forge import bootstrap
    lock = tmp_path / 'requirements.lock'
    lock.write_text('pdfminer-six==20260107\n')
    cache = tmp_path / 'cache'
    key = hashlib.sha256(lock.read_bytes() + sys.version.encode() + platform.machine().encode()).hexdigest()
    target = cache / key
    target.mkdir(parents=True)
    (target / '.complete').write_text(key)
    monkeypatch.setattr(bootstrap, '_installed', lambda name: None)
    monkeypatch.setattr(bootstrap.metadata, 'distributions', lambda **kwargs: [SimpleNamespace(metadata={'Name': 'pdfminer.six'}, version='20260107')])
    monkeypatch.delitem(sys.modules, 'marimo', raising=False)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    assert bootstrap.ensure_dependencies(lock, cache=cache) == {'pdfminer-six': '20260107'}
