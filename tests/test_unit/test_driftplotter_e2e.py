"""End-to-end tests that exercise the public DriftPlotter API."""
from pathlib import Path

import matplotlib
import pytest
from matplotlib.figure import Figure

from driftplots.driftplotter import DriftPlotter
from driftplots.interactive.driftmap_plot_widget import DriftmapPlotWidget

matplotlib.use("Agg")

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

SORTER_OUTPUT = Path(__file__).parent.parent.parent / "examples" / "example_data" / "sorting" / "sorter_output"


@pytest.fixture(scope="module")
def plotter():
    return DriftPlotter(SORTER_OUTPUT)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:

    def test_arrays_populated(self, plotter):
        assert plotter.data_loader._spike_times.size > 0

    def test_templates_shape(self, plotter):
        dl = plotter.data_loader
        assert dl.templates.ndim == 3


# ---------------------------------------------------------------------------
# Matplotlib path
# ---------------------------------------------------------------------------
class TestMatplotlib:

    def test_returns_figure(self, plotter):
        fig = plotter.drift_map_plot_matplotlib()
        assert isinstance(fig, Figure)

    def test_with_histogram(self, plotter):
        fig = plotter.drift_map_plot_matplotlib(add_histogram_plot=True)
        assert len(fig.axes) == 2

    def test_with_options(self, plotter):
        fig = plotter.drift_map_plot_matplotlib(
            decimate=2,
            exclude_noise=False,
            amplitude_cmap_scaling="log2",
            n_color_bins=10,
            point_size=3.0,
            title="test title",
        )
        assert isinstance(fig, Figure)
        assert fig.texts[0].get_text() == "test title"  # suptitle


# ---------------------------------------------------------------------------
# Interactive path
# ---------------------------------------------------------------------------
class TestInteractive:

    def test_returns_widget(self, plotter):
        widget = plotter.drift_map_plot_interactive()
        assert isinstance(widget, DriftmapPlotWidget)

    def test_with_options(self, plotter):
        widget = plotter.drift_map_plot_interactive(
            decimate=2,
            exclude_noise=False,
            amplitude_cmap_scaling="log10",
            n_color_bins=15,
            point_size=4.0,
            title="interactive test",
        )
        assert isinstance(widget, DriftmapPlotWidget)
