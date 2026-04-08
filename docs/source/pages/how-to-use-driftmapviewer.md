# Using `driftplots` 

`driftplots` can be to generate static matplotlib figures or an interactive viewer (using Qt). 
In the interactive viewer, clicking a spike will display the *template* (unscaled, whitened) for 
the cluster to which that spike was assigned.

`driftplots` accepts a path to Kilosort's output or a SpikeInterface SortingAnalyzer as input.

Below we will cover the main ways to use `driftplots`. 
See the [API Reference](/pages/api_index) for a full list of arguments.

## Inputs

Kilosort 1-4 are supported when passing a path to Kilosort output 
(see [here](/pages/how-parameters-are-calculated) for details on how features 
are computed across versions). By default, `spike_clusters.npy` is used to assign the
template for each spike (i.e. this file reflects any changes made in Phy). 
If no changes were made in Phy, the file does not exist and Kilosort's original
clusters are used in instead (`spike_templates.npy`).

If passing a `SortingAnalzyer`, it is expected the required extensions
have already been computed. See 
[this example](https://github.com/neuroinformatics-unit/driftplots/blob/e8ec328e14cc848feca3e7e90604501bb9e343f1/examples/example_data/create_analyzer.py#L1) 
for the required extensions. Note that the number of spikes displayed will depend 
on the argument set for `max_spikes_per_unit` used when computing `"random_spikes"`.

By default, the number of spikes displayed will be decimated to `100,000`.
See [Data Processing](data-processing) for details on how data can be processed prior to display.

::: {warning}
`driftplots` was designed and tested with Neuropixels probes, but it should also work with other probe types.
:::


## Interactive Viewer

The interactive viewer can be used to open a Qt interface that allows you to select
the individual spikes on the driftmap. Once selected, the template for that driftmap will
be displayed on the right-hand side.

```{image} /_static/interactive-single-example.png
   :align: center
   :width: 750px
```

```python
from driftplots import DriftPlotter

plotter = DriftPlotter(
    "/path/to/sorter_output",
)

driftmap = plotter.drift_map_plot_interactive(
    exclude_noise=True
)

driftmap.plot()

```

It is important to remember that the displayed heatmap shows the *template*
associated with that spike i.e. it will appear the same for all spikes
within the same cluster. The template is whitened and is not scaled to each individual spike.
This is due to the difficulty of reconstructing individual waveforms from
kilosort outputs across versions, and because the main purpose of the interactive mode is to check that templates are identifiable
across sessions (rather than inspect the individual, noisy waveforms). 
See [here](/pages/how-parameters-are-calculated) for details).

### Interactive Viewer with Multiple Plots

The `MultiSessionDriftmapWidget` can be used to display 
multiple interactive plots at once. In this mode, the y-axis
will be aligned when zooming. See [below](amplitudes) details
on how to ensure amplitude scaling is consistent.

```python
from pathlib import Path
import spikeinterface as si
from driftplots import DriftPlotter, MultiSessionDriftmapWidget

# Load the data. In this example we load as a sorting analyzer
# or from the raw kilosort output to demonstrate both methods
data_path = Path(__file__).parent / "example_data"
analyzer = si.load_sorting_analyzer(data_path / "analyzer.zarr")
sorting_output_path = data_path / "sorting" / "sorter_output"

# Create a list of interactive plots, and collect them
# into a single plot using MultiSessionDriftmapWidget
panels = []
for path_or_analyzer in [analyzer, sorting_output_path]:
    plotter = DriftPlotter(path_or_analyzer)

    plot = plotter.drift_map_plot_interactive()

    panels.append(plot)

multi = MultiSessionDriftmapWidget(panels)

multi.plot()
```

## Matplotlib mode

Matplotlib mode will return a static Matplotlib figure. This function
takes all the same arguments as {py:meth}`~driftplots.DriftPlotter.drift_map_plot_interactive`
but additionally can be used to plot a 1D activity histogram next to the driftmap.

```{image} /_static/matplotib-example.png
   :align: center
   :width: 750px
```

See [this example](/pages/examples/creating-pdf) for how to stitch matplotlib figures together across 
a project to quickly assess recording quality and stability.

```python
import matplotlib.pyplot as plt
import spikeinterface as si

from driftplots import DriftPlotter

analyzer = si.load_sorting_analyzer("/path/to/analyzer.zarr")

plotter = DriftPlotter(analyzer)

fig = plotter.drift_map_plot_matplotlib(
    exclude_noise="KSLabel",
    add_histogram_plot=True,
    weight_histogram_by_amplitude=True,
)

plt.show()

```

(amplitudes)=
## Aligning amplitudes across sessions

`driftplots` provides options for excluding spikes based on their
amplitudes and setting the color scaling for the amplitudes on the scatter plot.
This can be useful when a small number of high or low amplitude spikes dominate.

When comparing multiple sessions, it is useful to use the same amplitude cutoff
for discarding spikes and color scaling. The {py:func}`~driftplots.get_amplitudes` function can be used 
to pool amplitudes from across sessions, allowing for cutoffs to be calculated
and applied to all plots.

```python
import numpy as np

analyzer = si.load_sorting_analyzer("/path/to/an/analyzer.zarr")

SORTING_SESSIONS = [
    "/path/to/a/sorting",
    analyzer
]

all_spike_amplitudes = get_amplitudes(
    SORTING_SESSIONS, exclude_noise=False, concatenate=False
)

min_cutoff, max_cutoff = np.percentile(concat_spike_amplitudes, (0, 95))

app = QtWidgets.QApplication([])

panels = []
for path_or_analyzer in SORTING_SESSIONS:
    plotter = DriftMapView(path_or_analyzer)

    plot = plotter.drift_map_plot_interactive(
        filter_amplitude_mode="absolute",
        filter_amplitude_values=(min_cutoff, max_cutoff),
        amplitude_cmap_scaling=(min_cutoff, max_cutoff),
        n_color_bins=25,
    )

    panels.append(plot)

multi = MultiSessionDriftmapWidget(panels)

app.exec()

```

(data-processing)=
## Data processing before plotting

When a sorting analyzer is used to create the driftmap, only the number
of spikes used to compute the analyzer are used. This is defined by the
XXX argument on XXX.

When loading kilosort output, every detected spike is loaded from disk.

These spikes may include noise spikes, spuriously low or high amplitudes spikes,
or be so numerous as to slow down the plot generation. These can be dealt with
with hte `exclude_noise`, `` and ``, and `` parametrs repescetively. See the
API documentation for details.

The order in which these are applied can be found in the source code,
detailed here.

Exclude noise and analyzer

