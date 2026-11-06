"""Prepare local SFT demonstrations; this command never uploads or starts jobs."""
import argparse
import json
from pathlib import Path
from mlsysbook_forge.training import prepare_sft

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, default=Path('.forge/training'))
    args = parser.parse_args()
    print(json.dumps(prepare_sft(args.output, args.source_commit), indent=2))
