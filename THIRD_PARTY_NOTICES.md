# Third-party components

## Ternes spin-excitation code

`modules/ternes/main/Spin-*.sce`: copyright Markus Ternes, as stated in the retained source headers. GNU GPL version 3 or later. Full GPL text is in `modules/ternes/LICENSE`. The four scientific source files are preserved; they run with a separately installed Scilab. The wrapper is not a replacement for the original model or its documentation.

## Hurwitz/Frota-Fano fitting

`modules/kondo/main-program/hurwitzKondoClass.m` derives from Robert Drost's MATLAB implementation, copyright 2024 Robert Drost, Aalto University. Upstream: [david-jacob/HurwitzFanoFit](https://github.com/david-jacob/HurwitzFanoFit), MATLAB directory, GNU GPL v3. License is included in `modules/kondo/LICENSE`.

This toolkit includes a locally adapted model and GUI/export wrappers, not a claim of original authorship of the fitting theory or class. Relative to upstream, the existing local version includes GUI-related parameter controls, bounds/fixed-parameter handling and fit-result export integration. Publication packaging on 2026-10-25 does not change the model file. The related Kondo integration is distributed under GPL v3. See upstream README for scientific citations and acknowledgement requests.

## Color palettes

`assets/Nanox.cpal` retains the existing Nanox color ramp used by the tools; it is not a copy of the WSxM application. Other palette names identify their source color schemes. No WSxM/Gwyddion application binaries are bundled.

## Installed dependencies and external software

Python, NumPy, SciPy, Matplotlib, Pillow, PySide6/Qt, pyqtgraph, PyOpenGL and optional originpro/OriginExt are installed separately and retain their respective licenses. No copies of the original machine's Python environment or vendor SDK are distributed. MATLAB, Origin and Scilab must be installed separately under their own terms.

No blanket license is asserted for unrelated original components of this repository. Retain individual notices when redistributing licensed components.
