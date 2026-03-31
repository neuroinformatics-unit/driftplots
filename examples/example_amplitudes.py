import matplotlib.pyplot as plt
from driftmap_viewer import get_amplitudes, DriftMapView
import spikeinterface as si
from pathlib import Path
import numpy as np

# Getting the amplitudes across a set of sorting outputs can be useful to
# compute absolute amplitudes used for filtering spikes based
# on amplitude, or scaling color map values the same across plots.

# Load the data. In this example we load as a sorting analyzer
# or from the raw kilosort output to demonstrate both methods
data_path = Path(__file__).parent / "example_data"
analyzer = si.load_sorting_analyzer(data_path / "analyzer.zarr")
sorting_output_path = data_path / "sorting" / "sorter_output"
breakpoint()
all_spike_amplitudes = get_amplitudes([analyzer, sorting_output_path], concatenate=False)

fig, axes = plt.subplots(1, 2)
for idx, amplitudes in enumerate(all_spike_amplitudes):
    axes[idx].hist(amplitudes, bins=25)
    axes[idx].set_title(f"Session: {idx}")

plt.show()

concat_spike_amplitudes = np.concatenate(all_spike_amplitudes)
min_cutoff, max_cutoff = concat_spike_amplitudes.min(), concat_spike_amplitudes.max()

for path_or_analzyer in [analyzer, sorting_output_path]:
    plotter = DriftMapView(analyzer)

    plot = plotter.drift_map_plot_matplotlib(
        amplitude_scaling=(min_cutoff, max_cutoff),
        n_color_bins=25,
        filter_amplitude_mode=None,
    )

    plt.show()

