# MLPerf EDU report companions

CPU wheel sources are scoped to the official PyTorch `torch` and `torchvision`
wheel pages through `--find-links`. Other dependencies resolve from PyPI. This
keeps their complete wheel hashes available across supported Python versions
instead of inheriting a partial mirror from the PyTorch index. Generated
PEP 723 metadata preserves these wheel locations.

When refreshing the lock, retain and verify the official PyTorch wheel hashes:
uv's `--find-links` compilation can omit their hash blocks even when
`--generate-hashes` is enabled. The lock completeness test rejects such output.
Hashes remain mandatory during installation on both macOS and Linux.

Four companions cover the implemented Tutorial 01 workflow and the three implemented CLI labs. They invoke the public tutorial smoke or lab CLI, preserving its quality, finite-value and token-parity gates. The original tutorial and CLI files remain intact. The tutorial's proposed future sessions remain roadmap material.

The default runs are CPU functional smokes. Tutorial 01 uses the canonical synthetic time-series `min` profile; the other labs use their explicit `--smoke` flags. These verify functionality and reporting, not forecasting quality, convergence, or public benchmark scores. See the [canonical tutorial rules](../../mlperf-edu/tutorials/README.md).

The optional report adapter runs **after** the measured subprocess exits. It reads JSON reports rather than scraping stdout. When a provenance manifest is supplied it runs `python -m mlperf_edu verify` before producing publication evidence. Verification failure stops that publication path. It preserves the report's quality status even when a process exits successfully. No tracker callbacks are inserted into timed regions.

Original JSON/HTML/CSV reports and `.provd.json` manifests stay in the isolated local workspace. The upload preview contains sanitized report fields, units encoded by the original metric names, version/hardware fingerprints, and original-file SHA-256 digests. It excludes executable paths and free-form software environment values. The sanitized projection is not represented as the byte-identical original report. Provenance verifies content integrity; it does not authenticate the producer.

Each generated source retains platform markers and the official PyTorch CPU index in its dependency metadata. Linux uses CPU wheels; macOS uses the corresponding native wheels. Hosted smoke results remain functional smoke regardless of the machine running them. Readers conducting full device experiments should run the canonical CLI in their own controlled environment, then call `mlsysbook_forge.mlperf.read_report` and publish its returned evidence explicitly through a `ForgeSession` with the reported result category.

Frozen companion copies intentionally do not inherit a parent checkout's Git metadata. The notebook's source commit and bundle hashes establish which canonical sources were packaged; the report's own missing-Git context remains explicit.

The tutorial delegates seed selection to MLPerf EDU. Its companion does not declare a seed it never sets; use the actual `seed` in the verified workload report (including any documented `MLPERF_EDU_SEED` override). CLI lab seeds remain their explicit canonical defaults.
