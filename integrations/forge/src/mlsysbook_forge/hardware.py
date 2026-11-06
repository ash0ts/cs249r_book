"""Analyze imported kit evidence without claiming to run a physical device."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
import re
import statistics

from .evidence import sanitize


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value):
        raise ValueError('Model and firmware lineage require lowercase SHA-256 digests')
    return value


def analyze(data: bytes, kind: str, *, platform: str) -> dict:
    if len(data) > 2 * 1024 * 1024:
        raise ValueError('Import a compact manifest or measurement summary (at most 2 MiB)')
    report = {'input_sha256': hashlib.sha256(data).hexdigest(), 'platform': platform,
              'execution_location': 'host analysis of imported evidence', 'status': 'passed'}
    if kind == 'dataset':
        rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        ids = set()
        counts = Counter()
        classes = Counter()
        for row in rows:
            sample, label, split = (row.get(k, '').strip() for k in ('sample_id', 'label', 'split'))
            if not sample or not label or split not in {'train', 'validation', 'test'}:
                raise ValueError('CSV requires sample_id,label,split and explicit train/validation/test assignments')
            if sample in ids:
                raise ValueError('Duplicate sample identity: check split leakage before training')
            ids.add(sample); counts[split] += 1
            # Raw examples, sample identities and class names remain local.
            classes[(split, hashlib.sha256(label.encode()).hexdigest())] += 1
        if not counts.get('train') or not counts.get('test'):
            raise ValueError('A dataset manifest must include nonempty train and held-out test splits')
        report.update(result_kind='supplied_case', sample_count=len(rows), split_counts=dict(counts),
                      class_counts=[{'split': split, 'label_digest': label, 'count': count}
                                    for (split, label), count in sorted(classes.items())],
                      limitation='Manifest consistency only; content duplicates and label correctness are not verified.')
    elif kind == 'model':
        model = json.loads(data)
        widths = {'int8': 1, 'uint8': 1, 'float16': 2, 'bfloat16': 2, 'int16': 2,
                  'float32': 4, 'int32': 4, 'float64': 8, 'int64': 8}
        parameters = storage = 0
        tensors = model.get('parameter_tensors', [])
        if not tensors:
            raise ValueError('Supply a parameter_tensors manifest exported from the actual model')
        for tensor in tensors:
            shape, dtype = tensor['shape'], tensor['dtype']
            if dtype not in widths or not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError('Unsupported dtype or invalid tensor shape')
            size = math.prod(shape)
            parameters += size; storage += size * widths[dtype]
        report.update(result_kind='supplied_case', model_sha256=_sha(model['model_sha256']),
                      parameter_count=parameters, parameter_storage_bytes=storage, tensor_count=len(tensors),
                      units={'parameter_storage_bytes':'byte'},
                      limitation='Computed from the supplied parameter manifest; excludes activations, runtime arena, metadata and device overhead.')
        if 'flash_budget_bytes' in model:
            budget = model['flash_budget_bytes']
            if type(budget) is not int or budget <= 0:
                raise ValueError('Flash budget must be a positive integer number of bytes')
            report.update(flash_budget_bytes=budget, parameter_bytes_fit_flash=storage <= budget)
    elif kind == 'measurements':
        source = json.loads(data)
        if not source.get('capture_method') or not source.get('board'):
            raise ValueError('Identify the board and measurement method; device serial numbers are not needed')
        rows = source.get('measurements', [])
        if not rows:
            raise ValueError('No authentic device measurements were supplied')
        units = {'latency_ms':'ms', 'energy_mj':'mJ', 'power_mw':'mW'}
        summaries = {}
        for metric, unit in units.items():
            values = [row[metric] for row in rows if metric in row]
            if not values:
                continue
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('Device metrics must be finite nonnegative numbers in the stated units')
            summaries[metric] = {'unit':unit, 'n':len(values), 'mean':statistics.mean(values),
                                 'median':statistics.median(values), 'min':min(values), 'max':max(values)}
        if not summaries:
            raise ValueError('Provide latency_ms, energy_mj or power_mw observations')
        report.update(result_kind='measured_experiment', model_sha256=_sha(source['model_sha256']),
                      firmware_sha256=_sha(source['firmware_sha256']), board=source['board'],
                      capture_method=source['capture_method'], summary=summaries,
                      conditions={key: source.get('conditions', {}).get(key) for key in
                                  ('ambient_temperature_c', 'clock_mhz', 'batch_size', 'warmup_count', 'power_scope')},
                      limitation='Imported device observations; authenticity and measurement independence are not attested by this notebook.')
    else:
        raise ValueError('Unknown evidence kind')
    return sanitize(report)
