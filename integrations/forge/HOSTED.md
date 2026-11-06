# Optional hosted inference and tool evaluation

The `hosted-tool-selection` companion connects the outcome-versus-execution discussion in Volume I, serving and distributed-system tradeoffs in Volume II, and agent evaluation in Volume III to one small actual model execution. Build it with the same immutable packaging command as other companions. Canonical Co-Labs continue to run without an account.

The notebook starts without any remote actions. Supply a private `entity/project`, an explicit model ID from the authenticated service catalog, the documented inference endpoint, current token rates and their verification date. Use **Run bounded hosted evaluation** to execute six synthetic tasks. The input, response, real local tool execution, deterministic score, source revision and token usage are retained locally. Weave receives actual model/tool operations and a native evaluation when tracing succeeds. A separate evidence-publishing button remains optional.

See [the research contract](RESEARCH_CONTRACT.md) for splits, checker semantics, exclusions and cost assumptions. The same training split is used by the optional SFT/RLVR companion. No student reflections are sent.

All campaign allowances belong in one `mlsysbook_forge.budget.Budget` database. Reserve notebook, storage and other service costs before calling `run_evaluation`; its internal reservation covers the six inference requests. Keep the database in private persistent storage. Copies of a notebook do not automatically share their local filesystem: a campaign orchestrator must supply the same ledger or preallocate a bounded allowance per copy. Never reset the ledger or reuse a budget independently in multiple copies.

Repeated clicks, cell execution and reloads reuse the named attempt on the same persistent ledger. Interrupted or uncertain attempts stop for inspection. Choose a new attempt only for an intentional, separately budgeted retry. Credentials come from the existing authenticated SDK environment and never from notebook fields. SDKs load only after the explicit action.

Service catalog and pricing references (check again before spending):

- [Forge inference API](https://docs.coreweave.com/products/inference/serverless/api-reference)
- [Inference and sandbox pricing](https://coreweave.com/forge-pricing/inference-agents)
- [Serverless training API](https://docs.coreweave.com/products/post-training/serverless-training/api-reference)

An upload failure does not erase local evidence. A `submitted` trace status does not prove remote persistence; inspect the project and record a separate private validation receipt. Private notebook, run and trace URLs do not belong in upstream teaching documentation.

The hosted lock pins OpenTelemetry 1.38.0 because Weave 0.53.9 expects its requests-based exporter session. Newer 1.45 exporters remove that attribute and fail trace initialization; local model evidence remains available. Revalidate this constraint when upgrading the SDK.
