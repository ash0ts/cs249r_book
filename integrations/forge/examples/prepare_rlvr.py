"""Compute rewards from two actual training-split rollout reports; no job submission."""
import argparse
import json
from pathlib import Path
from mlsysbook_forge.training import rlvr_groups

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs=2, type=Path)
    parser.add_argument('--output', type=Path, default=Path('.forge/rlvr-groups.json'))
    args = parser.parse_args()
    result = rlvr_groups([json.loads(path.read_text()) for path in args.reports])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'])
