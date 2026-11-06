"""Restore pinned lesson dependencies without modifying the platform environment."""
from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def locked_versions(lock: Path) -> dict[str, str]:
    return dict(re.findall(r"^([a-zA-Z0-9_-]+)==([^\s;\\]+)", lock.read_text(), re.M))


def ensure_dependencies(lock: Path, *, cache: Path | None = None) -> dict[str, str]:
    """PEP 723 hosts can preinstall the lock; other native hosts restore it here.

    The platform's Marimo module is already imported before this function runs.
    Do not replace a live server's modules: require the tested Marimo version.
    """
    expected = locked_versions(lock)
    loaded_marimo = sys.modules.get("marimo")
    if loaded_marimo and loaded_marimo.__version__ != expected["marimo"]:
        raise RuntimeError(f"This copy requires marimo=={expected['marimo']}; "
                           "use its pinned environment or rebuild for the current hosted runtime.")
    if all(_installed(name) == version for name, version in expected.items()):
        return expected
    key = hashlib.sha256(lock.read_bytes() + sys.version.encode() + platform.machine().encode()).hexdigest()
    cache = cache or Path.cwd() / ".forge" / "dependencies"
    target = cache / key
    cache.mkdir(parents=True, exist_ok=True)
    if not (target / ".complete").exists():
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("Install uv, or preinstall notebook-requirements.lock with --require-hashes.")
        with tempfile.TemporaryDirectory(dir=cache) as staging:
            install = subprocess.run(
                [uv, "pip", "install", "--python", sys.executable, "--target", staging,
                 "--require-hashes", "-r", str(lock)], capture_output=True, timeout=300,
            )
            if install.returncode:
                (cache / "install.log").write_bytes(install.stdout + install.stderr)
                raise RuntimeError("Pinned dependency installation failed; inspect .forge/dependencies/install.log.")
            Path(staging, ".complete").write_text(key)
            if not target.exists():
                Path(staging).rename(target)
    sys.path.insert(0, str(target))
    importlib.invalidate_caches()
    found = {dist.metadata["Name"].lower().replace("_", "-"): dist.version
             for dist in metadata.distributions(path=[str(target)])}
    if any(found.get(name.lower().replace("_", "-")) != version for name, version in expected.items()):
        raise RuntimeError("Incomplete pinned environment; rebuild the dependency cache.")
    return expected


def _installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
