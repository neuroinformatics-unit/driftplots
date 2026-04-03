# Multi-widgets for comparing sessions

The main use case of the interactive viewer is to compare multiple sessinos. For well-aligned sessions,
the drift maps should match (e.g. high amplitude near high amplitude) and templates of neurons
are points should look similar (i.e. they are the same neuron, found on separate days).

To do this, you can create multiple driftmaps and then plot them together on the MultiSessionDriftmapWidget()

```python
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
        amplitude_cmap_scaling="linear",
        n_color_bins=25,
    )

    panels.append(plot)

multi = MultiSessionDriftmapWidget(panels)

# We must start the Qt event loop for the plots to appear
app.exec()

# You will notice the plots and templates look different, even
# though the underlying data is the same. This is because kilosort
# and SpikeInterface use different methods to compute amplitudes, depths and templates

```

# Scaling amplitudes

When you are comparing sessions, it can be useful to set the same limits on the mplaitude
of included spikes,a d ensure the color scaling is the same across all plots.
This helps with direct comparison.

To do this, first generate ann pooled amplitudes so you can calculate limits.
This can also be useful to compare amplitudes across sessions to check they
are in a similar range.


```python
# Getting the amplitudes across a set of sorting outputs can be useful to
# compute absolute amplitudes used for filtering spikes based
# on amplitude, or scaling color map values the same across plots.

# Load the data. In this example we load as a sorting analyzer
# or from the raw kilosort output to demonstrate both methods
data_path = Path(__file__).parent / "example_data"
analyzer = si.load_sorting_analyzer(data_path / "analyzer.zarr")
sorting_output_path = data_path / "sorting" / "sorter_output"

all_spike_amplitudes = get_amplitudes(
    [analyzer, sorting_output_path], concatenate=False
)

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
        amplitude_cmap_scaling=(min_cutoff, max_cutoff),
        n_color_bins=25,
        filter_amplitude_mode="absolute",
        filter_amplitude_values=(min_cutoff, max_cutoff),
        exclude_noise="KSLabel",
    )

    plt.show()

```
