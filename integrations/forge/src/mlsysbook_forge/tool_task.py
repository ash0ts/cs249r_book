"""Frozen synthetic tool-selection task, shared by inference, SFT and RLVR."""
from __future__ import annotations

import hashlib
import json

SYSTEM = "Choose exactly one tool for the request. Supply only the requested arguments."
TOOLS = [
    {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False}}}
    for name, description, properties in [
        ("add", "Add two integers", {"a": {"type": "integer"}, "b": {"type": "integer"}}),
        ("multiply", "Multiply two integers", {"a": {"type": "integer"}, "b": {"type": "integer"}}),
        ("unit_name", "Expand a unit symbol", {"symbol": {"type": "string", "enum": ["ms", "W", "J"]}}),
    ]
]


def tasks(split: str) -> list[dict]:
    if split not in {"train", "evaluation"}:
        raise ValueError("Unknown fixed split")
    numbers = range(1, 9) if split == "train" else range(21, 23)
    rows = []
    for n in numbers:
        for name, prompt, args in [
            ("add", f"Add {n} and {n + 3}.", {"a": n, "b": n + 3}),
            ("multiply", f"Multiply {n} by {n + 2}.", {"a": n, "b": n + 2}),
            ("unit_name", f"Record {n}: expand the unit symbol {['ms', 'W', 'J'][n % 3]}.",
             {"symbol": ["ms", "W", "J"][n % 3]}),
        ]:
            rows.append({"id": f"{split}-{name}-{n}", "prompt": prompt,
                         "expected": {"name": name, "arguments": args}})
    return rows


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def execute_tool(name: str, arguments: dict):
    """Allowlisted pure tools; no arbitrary code, network or filesystem access."""
    if name in {"add", "multiply"}:
        if set(arguments) != {"a", "b"} or any(type(v) is not int or abs(v) > 10000 for v in arguments.values()):
            raise ValueError("Expected two bounded integer arguments")
        a, b = arguments["a"], arguments["b"]
        return a + b if name == "add" else a * b
    if name == "unit_name" and set(arguments) == {"symbol"}:
        return {"ms": "millisecond", "W": "watt", "J": "joule"}[arguments["symbol"]]
    raise ValueError("Unknown tool or arguments")


def score(task: dict, output: dict) -> dict:
    available = output.get("status") == "completed"
    selected = output.get("selection")
    exact = available and digest(selected) == digest(task["expected"])
    return {"available": available, "exact_tool_and_arguments": bool(exact)}


def supervised_rows() -> list[dict]:
    return [{"messages": [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": row["prompt"]},
                          {"role": "assistant", "tool_calls": [{"id": "call_0", "type": "function",
                           "function": {"name": row["expected"]["name"],
                                        "arguments": json.dumps(row["expected"]["arguments"], sort_keys=True)}}]}],
             "tools": TOOLS} for row in tasks("train")]
