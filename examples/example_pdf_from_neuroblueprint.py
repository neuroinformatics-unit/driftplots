"""Iterate over a NeuroBlueprint project and save a single multi-page drift map PDF.

This example assumes a NeuroBlueprint-style directory layout
(https://neuroblueprint.neuroinformatics.dev) where spike-sorted
data lives under ``derivatives/``:

    my_project/
    └── derivatives/
        ├── sub-001/
        │   ├── ses-001/
        │   │   └── ephys/
        │   │       └── sorting_analyzer.zarr/
        │   └── ses-002/
        │       └── ephys/
        │           └── sorting_analyzer.zarr/
        └── sub-002/
            └── ses-001/
                └── ephys/
                    └── sorting_analyzer.zarr/

Each sorting analyzer is loaded and a matplotlib drift-map figure is
added as a page to a single output PDF.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import spikeinterface as si
from matplotlib.backends.backend_pdf import PdfPages

from driftplots import DriftPlotter

# ---------- configure paths ----------
project_path = Path("/path/to/my_project")
derivatives_path = project_path / "derivatives"
output_path = project_path / "output" / "all_drift_maps.pdf"
output_path.parent.mkdir(parents=True, exist_ok=True)

# ---------- iterate over subjects and sessions ----------
with PdfPages(output_path) as pdf:
    for sub_dir in sorted(derivatives_path.glob("sub-*")):
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            analyzer_path = ses_dir / "ephys" / "sorting_analyzer.zarr"

            analyzer = si.load_sorting_analyzer(analyzer_path)

            plotter = DriftPlotter(analyzer)

            fig = plotter.drift_map_plot_matplotlib(
                add_histogram_plot=True,
                weight_histogram_by_amplitude=True,
            )

            fig.suptitle(f"{sub_dir.name}  /  {ses_dir.name}", fontsize=14)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            print(f"  Added {sub_dir.name}/{ses_dir.name}")

print(f"\nSaved multi-page PDF to {output_path}")
