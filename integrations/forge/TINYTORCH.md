# TinyTorch companions

Build companions in the pinned Marimo 0.25.1 environment. The builder runs
Marimo's safe source formatter before hashing the notebook, so opening it in
the editor does not change cell signatures or its publication digest. This
formatting step does not execute any teaching cells.

The 20 module companions execute `tito dev export --all` and the selected
`tito module test NN` in a new disposable workspace. They preserve the canonical
source, student notebooks, grading cells and module workflow. Exporting source
solutions is a maintainer validation workflow; students continue using their
original graded notebooks. No companion replaces an assignment.

The 13 executable milestone companions cover all seven milestone families.
They use canonical scripts and include tracked dataset samples, datasheets and
licenses. Original RNG behavior is preserved; an unspecified seed is recorded
as null, never invented. Dataset downloads and longer milestones may need local
execution. Most commands have a 180-second limit; the TinyDigits MLP and three language-model
companions allow 600 seconds. All kill their process group on timeout; a timeout
is a failed validation, not a score. Companion stdin is closed and TinyTorch
terminal prompts are disabled. The three language-model companions use the
canonical `--quick` offline dataset mode, preserve its quality gates, and record
that scope explicitly. These four companions use one BLAS/OpenMP thread to avoid
oversubscription in small hosted CPU runtimes; the control is recorded with results.
No full-dataset language-model training claim follows from sample-mode results.

Each run records source revision, command/configuration, return code, units and
elapsed time. Raw training and profiling output remains in local command logs. The
optimization companion extracts allowlisted profiler table values after measurement,
including parameter/FLOP counts, latency and serial rate, with units, line numbers
and a log digest. It does not add work inside the canonical timing loop. Explicit printed
`Epoch / Loss / Train / Test` rows are extracted as numerical evidence with
line numbers, units, a log digest and a printed-precision label. Unrecognized
formats are not guessed; an empty epoch list is not proof that training ran.
Elapsed subprocess time includes interpreter startup and is not benchmark
latency or a valid MLPerf result. Only explicitly selected numerical reports
should be published as training metrics. Publishing is a separate button and
is disabled by default. Repeated execution of the same companion reuses its
local result; interrupted actions require inspection before retrying.

Build with the common notebook command and follow [publication checks](NOTEBOOKS.md).
The companion lock includes TinyTorch's canonical nbdev/jupytext export tools;
all generated notebooks, exported packages and datasets remain outside Git.

The CIFAR-10 companion defaults to the canonical `--test-only` architecture check: a synthetic forward pass with no training and no milestone completion claim. The full training path requires the real ~170 MB CIFAR-10 download and an interactive prompt or explicit `TINYTORCH_AUTO_DOWNLOAD=1`; use the canonical CLI for that separate experiment. The TinyDigits CNN companion retains its actual training metrics.

The embeddings module (11) has a runtime constraint: its full inline notebook
analysis was OOM-killed in a 4 GiB hosted runtime (child exit -9; cgroup OOM
counter incremented). Its separate 37 module and 32 integration tests passed.
Use a local machine with more memory for the complete workflow. The hosted
companion opens and preserves evidence, but its default 4 GiB runtime does not
establish full module completion. No tests, array sizes or workload iterations
are removed to turn this failure into a pass.
