# Creating a PDF from an experiment

The matplotlib viewer

```python
from pathlib import Path

import matplotlib.pyplot as plt
import spikeinterface as si

from driftmap_viewer import DriftMapView

project_path = Path("/path/to/my_project")
pdf_name = f"{sub_dir.name}_{ses_dir.name}_drift_map.pdf"
output_path = project_path / "all_drift_maps.pdf"

with PdfPages(output_path) as pdf:
    for sub_dir in sorted(derivatives_path.glob("sub-*")):
        for ses_dir in sorted(sub_dir.glob("ses-*")):

            analyzer_path = ses_dir / "ephys" / "sorting_analyzer.zarr"

            analyzer = si.load_sorting_analyzer(analyzer_path)

            plotter = DriftMapView(analyzer)

            fig = plotter.drift_map_plot_matplotlib(
                add_histogram_plot=True,
                weight_histogram_by_amplitude=True,
            )

            fig.suptitle(f"{sub_dir.name}  /  {ses_dir.name}", fontsize=14)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            print(f"  Saved {pdf_name}")

print(f"\nAll drift map PDFs saved to {output_path}")
```
