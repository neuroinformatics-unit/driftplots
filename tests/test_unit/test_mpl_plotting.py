import inspect

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from driftplots import DriftPlotter
from driftplots.mpl_plotting import plot_matplotlib

matplotlib.use("Agg")


class TestMatplotlibScatter:
    """The scatter plot axes should faithfully reflect the spike data."""

    def test_returns_figure(self, synthetic_model):
        fig = plot_matplotlib(
            synthetic_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=False, weight_histogram_by_amplitude=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_scatter_has_correct_number_of_points(self, synthetic_model):
        fig = plot_matplotlib(
            synthetic_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=False, weight_histogram_by_amplitude=False,
        )
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()
        assert len(offsets) == 50
        plt.close(fig)

    def test_scatter_xy_data_matches(self, synthetic_model, synthetic_data):
        fig = plot_matplotlib(
            synthetic_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=False, weight_histogram_by_amplitude=False,
        )
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()

        np.testing.assert_array_equal(offsets[:, 0], synthetic_data["spike_times"])
        np.testing.assert_array_equal(offsets[:, 1], synthetic_data["spike_depths"])
        plt.close(fig)

    def test_single_axis_without_histogram(self, synthetic_model):
        fig = plot_matplotlib(
            synthetic_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=False, weight_histogram_by_amplitude=False,
        )
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_title(self, synthetic_model):
        fig = plot_matplotlib(
            synthetic_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=False, weight_histogram_by_amplitude=False,
            title="My Title",
        )
        assert fig._suptitle.get_text() == "My Title"
        plt.close(fig)


class TestMatplotlibHistogram:
    """When add_histogram_plot=True, a histogram axis should be added."""

    def test_two_axes_with_histogram(self, synthetic_model):
        fig = plot_matplotlib(
            synthetic_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=True, weight_histogram_by_amplitude=False,
        )
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_histogram_has_line(self, synthetic_model):
        fig = plot_matplotlib(
            synthetic_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=True, weight_histogram_by_amplitude=False,
        )
        hist_ax = fig.axes[0]
        assert len(hist_ax.lines) == 1
        plt.close(fig)

    def test_histogram_count_data(self, synthetic_model):
        """Histogram line x-data should match computed bin counts."""
        fig = plot_matplotlib(
            synthetic_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=True, weight_histogram_by_amplitude=False,
        )
        hist_ax = fig.axes[0]
        line = hist_ax.lines[0]

        bin_centers, counts = synthetic_model.compute_activity_histogram(
            weight_histogram_by_amplitude=False,
        )
        np.testing.assert_array_equal(line.get_xdata(), counts)
        np.testing.assert_array_equal(line.get_ydata(), bin_centers)
        assert line.get_xdata().sum() == 50
        plt.close(fig)

    def test_histogram_weighted_data(self, synthetic_model):
        """Weighted histogram values should differ from unweighted counts."""
        fig = plot_matplotlib(
            synthetic_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=True, weight_histogram_by_amplitude=True,
        )
        hist_ax = fig.axes[0]
        line = hist_ax.lines[0]

        bin_centers, weighted = synthetic_model.compute_activity_histogram(
            weight_histogram_by_amplitude=True,
        )
        np.testing.assert_array_equal(line.get_xdata(), weighted)
        np.testing.assert_array_equal(line.get_ydata(), bin_centers)
        assert np.all(line.get_xdata() >= 0)
        plt.close(fig)


# ---------------------------------------------------------------------------
# DriftPlotter API signature parity
# ---------------------------------------------------------------------------
MATPLOTLIB_ONLY_PARAMS = ["self", "add_histogram_plot", "weight_histogram_by_amplitude"]


def test_interactive_and_matplotlib_share_signature():
    """Shared parameters between interactive and matplotlib methods
    should have the same name, default and type annotation.
    """
    interactive = inspect.signature(DriftPlotter.drift_map_plot_interactive)
    matplotlib = inspect.signature(DriftPlotter.drift_map_plot_matplotlib)

    interactive_params = {k: v for k, v in interactive.parameters.items() if k != "self"}
    matplotlib_params = {k: v for k, v in matplotlib.parameters.items() if k not in MATPLOTLIB_ONLY_PARAMS}

    assert len(interactive_params) == len(matplotlib_params)

    for name, param in interactive_params.items():
        assert param == matplotlib_params[name]
