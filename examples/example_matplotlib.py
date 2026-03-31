import matplotlib.pyplot as plt
from driftmap_viewer import DriftMapView
import spikeinterface as si
from pathlib import Path

# Load the data. In this example we load as a sorting analyzer
# or from the raw kilosort output to demonstrate both methods
data_path = Path(__file__).parent / "example_data"
analyzer = si.load_sorting_analyzer(data_path / "analyzer.zarr")

plotter = DriftMapView(analyzer)

plot = plotter.drift_map_plot_matplotlib(
    decimate=False,
    exclude_noise=False,
    amplitude_scaling=None,
    n_color_bins=25,
    filter_amplitude_mode="percentile",
    filter_amplitude_values=(0, 99),
    add_histogram_plot=True,
    weight_histogram_by_amplitude=True
)

plt.show()
