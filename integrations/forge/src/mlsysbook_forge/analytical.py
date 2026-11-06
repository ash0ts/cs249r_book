"""Execute canonical analytical tutorials and retain actual solver results.

The observer records Python return values, not model/agent traces. It never
changes arguments or return values and is unsuitable for benchmark timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

from .evidence import sanitize


def executable_cells(source: str) -> list[tuple[int, str]]:
    """Only Quarto executable Python fences; prose examples remain prose."""
    lines = source.splitlines(keepends=True)
    cells = []
    start = None
    for index, line in enumerate(lines):
        if start is None and re.fullmatch(r'```\{python[^}]*\}\s*', line.strip()):
            start = index + 1
        elif start is not None and line.strip() == '```':
            code = ''.join(lines[start:index])
            if not re.search(r'^#\|\s*eval:\s*false\s*$', code, re.M | re.I):
                cells.append((start + 1, code))
            start = None
    if start is not None:
        raise ValueError('Unterminated executable Python fence')
    return cells


def execute_analytical(path: Path, *, limit: int = 4096) -> dict:
    # The framework initializes registry examples on import. Those solves are
    # not evidence produced by this lesson and must precede observation.
    import mlsysim  # noqa: F401

    observations = []
    omitted = 0
    unsupported = 0

    def observer(frame, event, result):
        nonlocal omitted, unsupported
        if event != 'return' or frame.f_code.co_name != 'solve':
            return
        module = frame.f_globals.get('__name__', '')
        if not module.startswith('mlsysim.') or result is None:
            return
        if len(observations) >= limit:
            omitted += 1
            return
        try:
            value = sanitize(result)
            args = {}
            for name in frame.f_code.co_varnames[:frame.f_code.co_argcount + frame.f_code.co_kwonlyargcount]:
                if name not in {'self', 'cls'} and name in frame.f_locals:
                    try:
                        args[name] = sanitize(frame.f_locals[name])
                    except (ValueError, TypeError):
                        args[name] = {'serialization': 'unsupported'}
            observations.append({'solver': module + '.solve', 'inputs': args, 'result': value})
        except Exception:
            unsupported += 1

    previous = sys.getprofile()
    namespace = {'__name__': '__main__', '__file__': str(path), 'display': lambda *args, **kwargs: None}
    report = {'result_kind': 'analytical_simulation', 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'observations': observations, 'status': 'running', 'executed_cell_lines': []}
    try:
        sys.setprofile(observer)
        if path.suffix == '.qmd':
            for line, code in executable_cells(path.read_text()):
                exec(compile('\n' * (line - 1) + code, str(path), 'exec'), namespace)
                report['executed_cell_lines'].append(line)
        else:
            runpy.run_path(str(path), run_name='__main__')
        report['status'] = 'passed' if observations else 'no_solver_results'
    except Exception as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        # Full diagnostics stay in local command logs, not uploaded evidence.
        import traceback
        traceback.print_exc()
    finally:
        sys.setprofile(previous)
    report['omitted_after_limit'] = omitted
    report['unsupported_results'] = unsupported
    report['timing_note'] = 'Observer overhead is present. These are analytical results, never benchmark measurements.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = execute_analytical(args.source)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
