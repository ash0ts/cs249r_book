"""Local catalog/build commands. Publication uses Forge's authenticated editor."""
import argparse
import json
import re
from pathlib import Path

from .coverage import inventory
from .companions import build_companion
from .packaging import build_marimo, read_source, registered_entries, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["catalog", "build"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--lab-id")
    parser.add_argument("--runtime-artifact", help="Explicit hosted asset: entity/project/collection:vN")
    parser.add_argument("--output", type=Path, default=Path(".forge/bundles"))
    args = parser.parse_args()
    root = args.root.resolve()
    commit = revision(root, args.revision)
    entries = registered_entries(root, commit)
    if args.command == "catalog":
        print(json.dumps(inventory(root, commit), indent=2))
        return
    selected = [e for e in entries if not args.lab_id or e["id"] == args.lab_id]
    if not selected:
        parser.error("No registered entry matches")
    requirements = [line.strip() for line in read_source(root, commit, "integrations/forge/notebook-requirements.in").decode().splitlines()
                    if line.strip() and not line.startswith("#")]
    for entry in selected:
        builder = {"marimo": build_marimo, "companion": build_companion, "hardware": build_companion, "hosted": build_companion}.get(entry.get("generator"))
        if builder is None:
            parser.error(f"No implemented builder for {entry['id']}")
        selected_requirements = requirements
        if entry.get("dependency_lock"):
            lock = read_source(root, commit, entry["dependency_lock"]).decode()
            selected_requirements = [line.strip() for line in re.findall(r"^[a-zA-Z0-9_-]+==[^\\\n]+", lock, re.M)]
        receipt = builder(root, commit, entry, args.output / f"{entry['id']}.py", selected_requirements,
                          asset_reference=args.runtime_artifact)
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
