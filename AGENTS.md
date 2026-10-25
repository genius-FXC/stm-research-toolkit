# Tool routing and scientific invariants

The repository root is the directory containing this file; never assume a particular drive or username. Read `registry/tools.json`, `README.md`, and `docs/ANALYSIS_TEMPLATES.md` first.

- Keep tools independent. Add thin adapters rather than rewriting scientific algorithms.
- Raw input is read-only. Write outputs to an explicitly chosen separate folder; never overwrite experimental files.
- Preserve coordinate conventions, units, normalization, missing values, FFT phase and fitting definitions. Display adjustments do not change source measurements.
- `research_quicklooks` are dataset-specific templates. Inspect their upstream inputs and scientific assumptions before executing; never infer missing regions or fabricate data.
- Kondo requires MATLAB and Optimization Toolbox. Preserve `saveKondoFitResult` normalization. Ternes requires Scilab and explicit fitting configuration; launching its GUI does not perform a fit.
- External executable locations belong in ignored `registry/machine.json`. Credentials belong in ignored local configuration, never source.
- Origin export must be validated by reopening the saved project when making scientific deliverables.
- Unified AI/OAuth is an extension protocol only. Do not claim connection to other chat histories. 2D-lockin has its own optional remote AI feature.
- Run `scripts/doctor.py` and relevant synthetic tests for packaging changes. Do not upload experimental data, sessions, environment folders, or generated results.
