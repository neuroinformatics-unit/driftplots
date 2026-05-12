# Creating a PDF from an experiment

This example shows how to loop through folders in an experimental project,
adding the driftplot for each session to a PDF file. This allows you to
get a quick overview of the drift within and across sessions in an experiment.

You can extend this example to use the same scaling for the amplitudes
across sessions, using {ref}`the consistent amplitudes example <consistent-amplitudes>`.

```python
from pathlib import Path

import matplotlib.pyplot as plt
import spikeinterface as si
from matplotlib.backends.backend_pdf import PdfPages

from driftplots import DriftPlotter

# Set up paths
project_path = Path("/path/to/my/project")
derivatives_path = project_path / "derivatives"
output_path = project_path / "output" / "all_drift_maps.pdf"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Iterate through every subject and session, adding
# the generated matplotlib plot to the PDF

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
```
