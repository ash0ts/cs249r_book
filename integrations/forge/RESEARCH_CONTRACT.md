# Synthetic tool-selection smoke contract

Question: can a selected hosted model choose one of three pure tools with the exact requested arguments, and can its response be reconciled with local evidence and a Weave evaluation?

The unit is one task ID in one named attempt. `src/mlsysbook_forge/tool_task.py` freezes 24 training prompts and six held-out evaluation prompts (two per tool). The unit-symbol vocabulary overlaps deliberately; arithmetic operands and full prompts do not. This is a plumbing smoke, not an estimate of general tool-use ability. SFT and RLVR use only the training split. Evaluation rows never supply training rewards.

Freeze the source commit, dependency lock, authenticated model catalog, model ID, endpoint, rates and date, temperature zero, and maximum 256 completion tokens before any request. The manifest records a SHA-256 of the full selected split, including expected selections. Prompts contain synthetic data only. No student content, secrets or host identifiers belong in the dataset.

A success requires exactly one tool call with the expected name and JSON arguments (including types). A returned answer without a tool call, malformed arguments, an extra tool call, or the wrong arguments is incorrect. The tools perform bounded integer addition, multiplication, or unit-symbol lookup. There is no LLM judge. Tool result and response are retained separately from the deterministic score.

Service failures are unavailable, not model mistakes. Report correct/total and available/total together. Preserve every attempt, usage when supplied, and missing usage as unknown. No automatic request retries. Reusing an attempt returns its durable report or stops on an interrupted/uncertain reservation; it cannot submit the request again. A separately named retry consumes a new allowance.

Only executed models and tools produce Weave operations. Analytical lab comparisons never produce synthetic model traces. Tracing failures preserve local results and are reported separately. Verify remote trace persistence before claiming trace validation; `submitted` alone is insufficient.

Reserve notebook compute, inference, storage, training and retries in one campaign ledger before launch. Current token prices are explicit inputs, not frozen promises. Inference reserves a loose 16,384-token input allowance per fixed request (request JSON is capped at 4 KiB) plus the API-enforced output maximum. Unknown charges retain their full allowance. Client timeout is not proof that remote work stopped. Training must have a service-enforced cost/runtime bound; small batch size and a client polling deadline do not establish one.

Before/after reports pair the same six task IDs, checker and model settings. Show gains, losses, shared successes/failures and unavailable pairs. A single smoke makes no claim that training improves quality. Changing a checker creates a separate study and never silently rescales prior attempts.
