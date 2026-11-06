"""Build self-contained Marimo copies from an immutable Git revision.

Generated source and private publication receipts belong in .forge/, never Git.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import lzma
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

RUNTIME_ROOTS = (
    "mlsysim/mlsysim", "labs/mlsysbook_labs",
    "integrations/forge/src/mlsysbook_forge",
)
# Leave headroom below the sandbox transport's message limit, even though the
# notebook artifact store itself accepts larger source files.
MAX_SOURCE_BYTES = 3500 * 1024


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def revision(root: Path, ref: str) -> str:
    commit = git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Expected a full source commit")
    return commit


def read_source(root: Path, commit: str, path: str) -> bytes:
    return git(root, "show", f"{commit}:{path}")


def runtime_bundle(root: Path, commit: str, extra_paths=(), *, roots=RUNTIME_ROOTS) -> tuple[bytes, dict]:
    """Archive only explicit tracked sources, with fixed ZIP timestamps/modes."""
    paths = (*roots, "LICENSE.md", "labs/LICENSE", "mlsysim/LICENSE.md",
             "integrations/forge/notebook-requirements.lock", *extra_paths)
    archive = git(root, "archive", "--format=tar", commit, *dict.fromkeys(paths))
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.isfile():
                files[member.name] = tar.extractfile(member).read()
            elif not member.isdir():
                raise ValueError(f"Refusing non-regular bundle member: {member.name}")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    # Compress across files, so shared source/docstring vocabulary is not reset
    # per ZIP member. This keeps standalone source below runtime transport caps.
    payload = lzma.compress(output.getvalue())
    manifest = {
        "source_commit": commit, "bundle_sha256": hashlib.sha256(payload).hexdigest(),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    }
    return payload, manifest


def bootstrap_source(payload: bytes, manifest: dict, asset_reference: str | None = None) -> str:
    """Code inside a notebook's first cell; extraction never invokes a shell."""
    encoded = base64.b85encode(payload).decode()
    if asset_reference:
        if not re.fullmatch(r"[^/\s]+/[^/\s]+/[^/:\s]+:v[0-9]+", asset_reference):
            raise ValueError("Runtime artifacts must pin entity/project/collection:vN")
        load = f'''    # Explicit hosted build: resolve this immutable private asset using
    # the runtime's injected credentials. Never put credentials in this source.
    import wandb as _asset_wandb
    _asset = _asset_wandb.Api(timeout=30).artifact({asset_reference!r}, type="mlsysbook-runtime")
    _asset_dir = _asset.download(root=str(Path.cwd() / ".forge" / "assets" / {manifest['bundle_sha256']!r}))
    _bundle = Path(_asset_dir, "runtime.xz").read_bytes()
'''
    else:
        load = f"    _bundle = _b64.b85decode({encoded!r})\n"
    return f'''    # MLSysBook runtime sources; see bundled component licenses.
    import base64 as _b64
    import hashlib as _hashlib
    import io as _io
    import lzma as _lzma
    import tempfile as _tempfile
    import zipfile as _zipfile
    import os as _os
{load.rstrip()}
    _bundle_sha = {manifest['bundle_sha256']!r}
    if _hashlib.sha256(_bundle).hexdigest() != _bundle_sha:
        raise ValueError("Notebook runtime bundle checksum mismatch")
    _cache = Path.cwd() / ".forge" / "runtime"
    _runtime = _cache / _bundle_sha
    _cache.mkdir(parents=True, exist_ok=True)
    if not (_runtime / ".complete").exists():
        with _tempfile.TemporaryDirectory(dir=_cache) as _temp:
            with _zipfile.ZipFile(_io.BytesIO(_lzma.decompress(_bundle))) as _archive:
                for _member in _archive.infolist():
                    _relative = Path(_member.filename)
                    if _relative.is_absolute() or ".." in _relative.parts:
                        raise ValueError("Unsafe bundled source path")
                _archive.extractall(_temp)
            Path(_temp, ".complete").write_text(_bundle_sha)
            if not _runtime.exists():
                _os.rename(_temp, _runtime)
    for _package_root in ("mlsysim", "labs", "integrations/forge/src"):
        _path = str(_runtime / _package_root)
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from mlsysbook_forge.bootstrap import ensure_dependencies as _ensure_dependencies
    _ensure_dependencies(_runtime / {manifest.get('dependency_lock', 'integrations/forge/notebook-requirements.lock')!r})
'''


def portable_marimo(source: str, bootstrap: str) -> str:
    """Replace a known native bootstrap structurally, leaving all lesson cells."""
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and ast.unparse(node.test) == "sys.platform == 'emscripten'"]
    if len(matches) != 1 or not matches[0].orelse:
        raise ValueError("Unrecognized native bootstrap: review source before packaging")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    # Hosted copies use native Python only. Canonical browser notebooks are untouched.
    result = "".join(lines[:node.lineno - 1]) + bootstrap + "".join(lines[node.end_lineno:])
    return result.replace("@app.cell\nasync def", "@app.cell(hide_code=True)\nasync def", 1)


def tracking_cells(entry: dict, commit: str) -> str:
    getter = entry["capture_getter"]
    if getter not in {"get_capture", "get_evidence", "get_saved"}:
        raise ValueError("Unsupported capture getter")
    return f'''

@app.cell(hide_code=True)
def _(mo, {getter}):
    from mlsysbook_forge.notebook import capture_panel
    forge_panel, forge_publish = capture_panel(
        mo, {getter}, lab_id={entry['id']!r}, source_commit={commit!r},
        result_kind={entry['result_kind']!r},
    )
    forge_panel
    return (forge_publish,)


@app.cell(hide_code=True)
def _(forge_publish, mo):
    mo.md(forge_publish.value)
    return
'''


def build_marimo(root: Path, commit: str, entry: dict, output: Path, requirements: list[str],
                 *, asset_reference: str | None = None) -> dict:
    source_bytes = read_source(root, commit, entry["source_path"])
    payload, manifest = runtime_bundle(root, commit)
    source = portable_marimo(source_bytes.decode(), bootstrap_source(payload, manifest, asset_reference))
    marker = '\nif __name__ == "__main__":'
    if source.count(marker) != 1:
        raise ValueError("Unrecognized Marimo entry point")
    cells = tracking_cells(entry, commit)
    source = source.replace(marker, cells + marker)
    metadata = '# /// script\n# requires-python = ">=3.13,<3.15"\n# dependencies = [\n'
    metadata += ''.join(f'#   {json.dumps(dep)},\n' for dep in requirements)
    metadata += '# ]\n# ///\n'
    source = metadata + f"# Canonical source: {entry['source_path']} @ {commit}\n" + source
    ast.parse(source)
    encoded = source.encode()
    if len(encoded) > MAX_SOURCE_BYTES:
        raise ValueError("Notebook exceeds the conservative 3500 KiB runtime transport budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    runtime_path = output.parent / "runtime" / f"{manifest['bundle_sha256']}.xz"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    if not runtime_path.exists():
        runtime_path.write_bytes(payload)
    receipt = {
        "lab_id": entry["id"], "source_path": entry["source_path"], "source_commit": commit,
        "canonical_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "bundle_sha256": manifest["bundle_sha256"], "requirements": requirements,
        "runtime_artifact": asset_reference,
        "output": str(output), "validation": "built; not published or executed",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps({**receipt, "runtime": manifest}, indent=2) + "\n")
    return receipt


def registered_entries(root: Path, commit: str | None = None) -> list[dict]:
    entries = []
    if commit:
        paths = git(root, "ls-tree", "-r", "--name-only", commit,
                    "integrations/forge/catalog").decode().splitlines()
        for path in sorted(paths):
            if path.endswith(".json"):
                entries.extend(json.loads(read_source(root, commit, path))["entries"])
    else:
        for path in sorted((root / "integrations/forge/catalog").glob("*.json")):
            entries.extend(json.loads(path.read_text())["entries"])
    ids = [entry["id"] for entry in entries]
    paths = [entry["source_path"] for entry in entries]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("Duplicate curriculum registration")
    return entries
