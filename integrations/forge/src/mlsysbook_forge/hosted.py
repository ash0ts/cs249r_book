"""Explicit, bounded hosted model execution; no SDK imports on lesson import."""
from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path

from .budget import Budget
from .tool_task import SYSTEM, TOOLS, digest, execute_tool, score, tasks

INFERENCE_ENDPOINT = "https://api.inference.wandb.ai/v1"


def validate_config(config: dict):
    allowed = {"endpoint", "project", "model", "source_commit", "max_tokens", "temperature",
               "pricing_source", "pricing_checked", "input_usd_per_million", "output_usd_per_million"}
    if set(config) != allowed:
        raise ValueError("Use only the documented configuration fields; never include credentials")
    if config.get("endpoint") != INFERENCE_ENDPOINT:
        raise ValueError("Use the documented Forge inference endpoint; credentials are never sent elsewhere")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", config.get("project", "")):
        raise ValueError("Explicit entity/project required")
    if not config.get("model") or not re.fullmatch(r"[0-9a-f]{40}", config.get("source_commit", "")):
        raise ValueError("Explicit model and immutable source commit required")
    if config.get("max_tokens") != 256 or config.get("temperature") != 0:
        raise ValueError("This frozen smoke uses max_tokens=256 and temperature=0")
    if not config.get("pricing_source") or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", config.get("pricing_checked", "")):
        raise ValueError("Record the verified current pricing source and date")
    for key in ("input_usd_per_million", "output_usd_per_million"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Explicit verified positive token rates required")


def run_evaluation(config: dict, *, budget_path: Path, output_dir: Path, attempt: str,
                   split: str = "evaluation", trace: bool = True) -> dict:
    """Call only from an explicit user action. Same attempt reuses durable results.

    A failed or uncertain request is retained and never silently resubmitted.
    A new attempt must be an explicit choice and consumes a new reservation.
    """
    validate_config(config)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", attempt):
        raise ValueError("Choose an explicit attempt identifier")
    rows = tasks(split)
    campaign = {"config": config, "dataset_sha256": digest(rows), "split": split, "attempt": attempt}
    action_id = "inference-" + digest(campaign)
    budget = Budget(budget_path, ceiling_usd=100)
    cached = budget.result(action_id)
    if cached:
        return cached
    # A deliberately loose bound: at most 16K input tokens for this <4KiB
    # request plus the API-enforced 256 output-token maximum, per row.
    if any(len(json.dumps({"messages": messages(row), "tools": TOOLS}).encode()) > 4096 for row in rows):
        raise ValueError("Frozen request exceeds the cost-bound assumptions")
    maximum = len(rows) * (16384 * config["input_usd_per_million"] +
                           256 * config["output_usd_per_million"]) / 1e6
    basis = json.dumps({"input_token_allowance_per_request": 16384, "max_tokens": 256,
                        "requests": len(rows), "rates": config}, sort_keys=True)
    if not budget.reserve(action_id, maximum, basis):
        raise RuntimeError("Attempt is incomplete or uncertain. Inspect local evidence before a separately budgeted retry.")
    directory = output_dir / action_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "manifest.json").write_text(json.dumps(campaign, indent=2) + "\n")
    # All service imports and authentication are below the explicit action boundary.
    import wandb
    from openai import OpenAI
    api = wandb.Api(timeout=30)
    entity, project = config["project"].split("/")
    project_info = api._service_api.execute_graphql(
        "query($e:String!,$p:String!){project(name:$p,entityName:$e){access}}", {"e": entity, "p": project})
    if project_info["project"]["access"] != "PRIVATE":
        raise ValueError("Hosted evidence requires a verified private project")
    client = OpenAI(api_key=api.api_key, base_url=config["endpoint"], project=config["project"],
                    timeout=60, max_retries=0)
    catalog = client.models.list().model_dump()
    (directory / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    if config["model"] not in {model["id"] for model in catalog["data"]}:
        raise ValueError("Model is absent from the authenticated service catalog")
    trace_error = None
    weave = None
    if trace:
        try:
            import weave as weave_sdk
            weave_sdk.init(config["project"], settings={"capture_code": False,
                "capture_client_info": False, "capture_system_info": False,
                "implicitly_patch_integrations": False, "print_call_link": False,
                "client_parallelism": 1}, attributes={"source_commit": config["source_commit"],
                "dataset_sha256": digest(rows), "attempt": attempt, "result_kind": "hosted_model_execution"})
            weave = weave_sdk
        except Exception as error:
            trace_error = type(error).__name__
    outputs = {}

    def model_call(task):
        if task["id"] in outputs:
            return outputs[task["id"]]
        record = {"task_id": task["id"], "status": "service_error"}
        try:
            response = client.chat.completions.create(model=config["model"], messages=messages(task),
                tools=TOOLS, tool_choice="required", temperature=0, max_tokens=256)
            # Synthetic prompts and model responses only; no credentials or client objects.
            record.update({"response": response.model_dump(), "status": "completed"})
            calls = response.choices[0].message.tool_calls or []
            if len(calls) == 1:
                call = calls[0].function
                try:
                    selection = {"name": call.name, "arguments": json.loads(call.arguments)}
                    record["selection"] = selection
                    record["tool_result"] = tool_op(**selection)
                except (ValueError, TypeError, KeyError):
                    record["tool_error"] = "invalid selection or arguments"
        except Exception as error:
            record["error_type"] = type(error).__name__
        outputs[task["id"]] = record
        (directory / f"{task['id']}.json").write_text(json.dumps(record, indent=2) + "\n")
        return record

    tool_op = execute_tool
    evaluation_summary = None
    try:
        if weave:
            tool_op = weave.op(name="mlsysbook.execute_tool")(execute_tool)
            model_op = weave.op(name="mlsysbook.tool_selection")(model_call)
            scorer = weave.op(name="mlsysbook.exact_tool_score")(score)
            evaluation = weave.Evaluation(name="mlsysbook-tool-selection-" + attempt,
                dataset=[{"task": row} for row in rows], scorers=[scorer], trials=1,
                metadata=campaign)
            try:
                evaluation_summary = asyncio.run(evaluation.evaluate(model_op))
            except Exception as error:
                trace_error = type(error).__name__
        # A trace/evaluation upload failure does not discard completed calls.
        # In-memory durable task IDs prevent evaluation retries from calling twice.
        for task in rows:
            if task["id"] not in outputs:
                model_call(task)
    finally:
        client.close()
        if weave:
            try:
                weave.finish()
            except Exception as error:
                trace_error = type(error).__name__
    scores = [{"task_id": row["id"], **score(row, outputs[row["id"]])} for row in rows]
    report = {**campaign, "result_kind": "hosted_model_execution", "action_id": action_id,
              "status": "completed", "scores": scores, "total": len(rows),
              "available": sum(s["available"] for s in scores),
              "correct": sum(s["exact_tool_and_arguments"] for s in scores),
              "trace_status": "failed" if trace_error else "submitted" if weave else "disabled",
              "trace_error_type": trace_error, "evaluation_summary": evaluation_summary,
              "maximum_reserved_usd": maximum, "actual_billed_usd": None,
              "outputs": list(outputs.values())}
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    budget.complete(action_id, report)
    return report


def messages(task):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
