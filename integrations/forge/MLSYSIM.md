# MLSys·im companions

The catalog covers all 17 executable Quarto tutorials and all 11 Python examples at the pinned source revision. The tutorial index is navigation, so it has no execution companion. Build a registered ID with the commands in [NOTEBOOKS.md](NOTEBOOKS.md).

Each companion runs the original source in an isolated workspace after **Run canonical companion**. Quarto executes only `{python}` fences that are not marked `eval: false`; illustrative `python` fences remain examples. Source line numbers are retained for diagnostics. Python examples retain their original `__main__` behavior, including sweeps and comparisons.

A temporary Python observer captures actual MLSys·im `solve` return values and supported arguments, including quantity units. It restores any previous observer afterward and never changes solver arguments or results. Collection is capped at 4,096 observations and reports omitted or unsupported results. These are **analytical simulation** results. Observer/process overhead is present, so elapsed times are not performance measurements, and this does not generate Weave model traces. The public Pipeline callback adapter remains available for applications that already control their pipeline; see [README.md](README.md).

The Hugging Face importer example reads the public model configuration from its canonical URL. It does not download weights or run a model. If that request fails and the original example catches the error, the companion records `no_solver_results` instead of calling it a successful analysis. Full stdout/tracebacks stay in local command logs. Only structured, sanitized evidence is offered for a separate explicit publication action.

The dependency lock and immutable runtime bundle include the framework sources and component licenses. Numerical parity with the unobserved engine and Quarto fence handling are tested. A catalog registration alone does not mean that a hosted copy has been validated; private publication receipts record that separately.
