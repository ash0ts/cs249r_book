# Portable Forge notebooks

The builder creates copies from an explicit Git commit. Canonical browser labs
and graded TinyTorch exports stay authoritative. Generated notebooks, extracted
runtime files, outputs and private receipts live under the ignored `.forge/`.

## Build a copy

Install `integrations/forge` locally, then run from the repository root:

```bash
python -m mlsysbook_forge catalog > .forge/coverage.json
python -m mlsysbook_forge build --revision HEAD --lab-id LAB_ID
```

Curriculum layers register supported entries in `catalog/*.json`. Inventory also
lists pending entries, hardware requirements, and exclusions. A registration or
successful build is not proof of hosted validation.

Generated Marimo source embeds a deterministic compressed archive of the tracked
runtime packages and license. On native startup it verifies the archive digest
and extracts to a content-addressed directory. Missing or different lesson
dependencies are restored from the hash-pinned lock to an isolated directory
using uv. The platform environment is not modified. The notebook needs no checkout,
relative wheel server, or access to someone else's private artifacts. Only the
native bootstrap is replaced; lesson calculations and controls are retained.
Generated source is capped at 3500 KiB to leave transport headroom. The artifact
store's larger source limit does not establish that a runtime can launch it.

`notebook-requirements.in` pins the direct runtime dependencies;
`notebook-requirements.lock` records the complete resolved set and wheel hashes.
Create a clean Python 3.13 environment using the lock before validating a build.
PEP 723 metadata declares pinned dependencies. The bootstrap also restores the
lock on hosts that do not consume this metadata. Marimo must match the tested
version because it is already running when the first cell executes. Rebuild and
validate the lock when the hosted runtime changes; do not silently replace a
live server's Marimo package.
Generated hosted copies include the pinned online SDK, but load it only when
online mode is selected. Canonical browser labs require no SDK or credentials.

## Publish through the authenticated editor

1. Select your own entity and a private project. Verify the project visibility.
2. Inspect existing receipts before creating a notebook. Compare the current
   saved source digest with the last receipt. Reuse an identical notebook; if
   someone edited it, preserve those edits and create a copy.
3. Create a notebook in Forge, upload the built source in its editor, and save it.
   Source creation or a transient workspace upload alone is not completion.
4. Verify the committed artifact version contains the expected source digest.
   Execute the notebook, inspect its rendered output, exercise its controls,
   close and reopen it, and repeat a basic execution check.
5. Store the notebook ID/URL, canonical path/commit, content digest, committed
   version, runtime versions, checks and failures in private receipts.
   `mlsysbook_forge.publication.save_receipt` validates the required proof fields.
   Keep private notebook URLs out of upstream documentation.

The evidence panel is disabled by default. A click publishes only a saved
comparison, previews its filtered payload, and reports the delivery receipt.
Changing a slider or opening the notebook does not create a run. Offline evidence
is local to the runtime: download the lesson's evidence before terminating it.
A runtime restart may discard its filesystem; uploaded notebook source persists.

## Validation and costs

Use the shared validation budget for notebook compute, inference, training and
retries. Read current rates and actual allocation/lifetime, reserve a conservative
upper bound before launch, and terminate each runtime after validation. Do not
launch GPU or training jobs without a known bounded cost. Publication receipts
must distinguish source persistence, execution, interactive checks, live tracking,
and model-job validation; one does not establish the others.

## Larger hosted bundles

For Forge, upload the generated `runtime/DIGEST.xz` as `runtime.xz` in a private
`mlsysbook-runtime` artifact, then rebuild with
`--runtime-artifact YOUR_ENTITY/YOUR_PROJECT/YOUR_COLLECTION:v0` (use the actual
committed version). The hosted bootstrap verifies its SHA-256 before extraction.
This explicit build option uses the sandbox's authenticated SDK to retrieve a
durable asset, and keeps notebook source small. Track the asset reference and
digest in the publication receipt. Never use a mutable `latest` alias.
The default embedded build remains available for local use without a service
account. The generated source and private asset references remain outside Git.
