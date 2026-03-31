from pathlib import Path

import spikeinterface as si
from PySide6 import QtWidgets

from driftmap_viewer import DriftMapView
from driftmap_viewer.interactive.multi_session_drift_map import (
    MultiSessionDriftmapWidget,
)

# In interactive mode, we must create a Qt instance before generating any plot.
app = QtWidgets.QApplication([])

# Load the data. In this example we load as a sorting analyzer
# or from the raw kilosort output to demonstrate both methods
data_path = Path(__file__).parent / "example_data"
analyzer = si.load_sorting_analyzer(data_path / "analyzer.zarr")
sorting_output_path = data_path / "sorting" / "sorter_output"

# Create a list of interactive plots, and collect them
# into a single plot using MultiSessionDriftmapWidget
panels = []
for path_or_analyzer in [analyzer, sorting_output_path]:
    plotter = DriftMapView(path_or_analyzer)

    plot = plotter.drift_map_plot_interactive(
        decimate=False,
        exclude_noise=False,
        filter_amplitude_mode="percentile",
        filter_amplitude_values=(1, 99),
        amplitude_scaling="linear",
        n_color_bins=25,
    )

    panels.append(plot)

multi = MultiSessionDriftmapWidget(panels)

# We must start the Qt event loop for the plots to appear
app.exec()

# You will notice the plots and templates look different, even
# though the underlying data is the same. This is because kilosort
# and SpikeInterface use different methods to compute amplitudes, depths and templates
