# TinyTorch companions

The 20 module companions execute `tito dev export --all` and the selected
`tito module test NN` in a new disposable workspace. They preserve the canonical
source, student notebooks, grading cells and module workflow. Exporting source
solutions is a maintainer validation workflow; students continue using their
original graded notebooks. No companion replaces an assignment.

The 13 executable milestone companions cover all seven milestone families.
They use canonical scripts and include tracked dataset samples, datasheets and
licenses. Original RNG behavior is preserved; an unspecified seed is recorded
as null, never invented. Dataset downloads and longer milestones may need local
execution. Commands have a 180-second limit and kill their process group on
timeout; a timeout is a failed validation, not a score.

Each run records source revision, command/configuration, return code, units and
elapsed time. Training and profiling output remains in local command logs. Explicit printed
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
