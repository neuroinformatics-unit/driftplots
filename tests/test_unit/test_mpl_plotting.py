from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from driftplots.data_loader import DataLoader
from driftplots.mpl_plotting import plot_matplotlib

matplotlib.use("Agg")

SORTER_OUTPUT = Path(__file__).parent.parent.parent / "examples" / "example_data" / "sorting" / "sorter_output"


@pytest.fixture()
def data_model():
    loader = DataLoader(SORTER_OUTPUT)
    return loader.get_processed_data(
        exclude_noise=False, decimate=False,
        filter_amplitude_mode=None, filter_amplitude_values=(),
    )


class TestMatplotlibScatter:
    """The scatter plot axes should faithfully reflect the spike data."""

    def test_returns_figure(self, data_model):
        fig = plot_matplotlib(
            data_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=False, weight_histogram_by_amplitude=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_scatter_has_correct_number_of_points(self, data_model):
        fig = plot_matplotlib(
            data_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=False, weight_histogram_by_amplitude=False,
        )
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()
        assert len(offsets) == data_model.spike_times.size
        plt.close(fig)

    def test_scatter_xy_data_matches(self, data_model):
        fig = plot_matplotlib(
            data_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=False, weight_histogram_by_amplitude=False,
        )
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()

        np.testing.assert_array_equal(offsets[:, 0], data_model.spike_times.ravel())
        np.testing.assert_array_equal(offsets[:, 1], data_model.spike_depths.ravel())
        plt.close(fig)

    def test_single_axis_without_histogram(self, data_model):
        fig = plot_matplotlib(
            data_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=False, weight_histogram_by_amplitude=False,
        )
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_title(self, data_model):
        fig = plot_matplotlib(
            data_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=False, weight_histogram_by_amplitude=False,
            title="My Title",
        )
        assert fig._suptitle.get_text() == "My Title"
        plt.close(fig)


class TestMatplotlibHistogram:
    """When add_histogram_plot=True, a histogram axis should be added."""

    def test_two_axes_with_histogram(self, data_model):
        fig = plot_matplotlib(
            data_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=True, weight_histogram_by_amplitude=False,
        )
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_histogram_has_line(self, data_model):
        fig = plot_matplotlib(
            data_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=True, weight_histogram_by_amplitude=False,
        )
        hist_ax = fig.axes[0]
        assert len(hist_ax.lines) == 1
        plt.close(fig)

    def test_histogram_count_data(self, data_model):
        """Histogram line x-data should match computed bin counts."""
        fig = plot_matplotlib(
            data_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=True, weight_histogram_by_amplitude=False,
        )
        hist_ax = fig.axes[0]
        line = hist_ax.lines[0]

        bin_centers, counts = data_model.compute_activity_histogram(
            weight_histogram_by_amplitude=False,
        )
        np.testing.assert_array_equal(line.get_xdata(), counts)
        np.testing.assert_array_equal(line.get_ydata(), bin_centers)
        plt.close(fig)

    def test_histogram_weighted_data(self, data_model):
        """Weighted histogram values should differ from unweighted counts."""
        fig = plot_matplotlib(
            data_model, amplitude_cmap_scaling="linear",
            n_color_bins=20, point_size=5.0,
            add_histogram_plot=True, weight_histogram_by_amplitude=True,
        )
        hist_ax = fig.axes[0]
        line = hist_ax.lines[0]

        bin_centers, weighted = data_model.compute_activity_histogram(
            weight_histogram_by_amplitude=True,
        )
        np.testing.assert_array_equal(line.get_xdata(), weighted)
        np.testing.assert_array_equal(line.get_ydata(), bin_centers)
        plt.close(fig)
