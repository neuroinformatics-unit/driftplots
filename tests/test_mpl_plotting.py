import inspect

import matplotlib
import numpy as np
import pytest

from driftplots import DriftPlotter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestMatplotlibScatter:
    """The scatter plot axes should faithfully reflect the spike data."""

    def test_scatter_xy_matches_synthetic(self, synthetic_ks4_output, synthetic_data):
        plotter = DriftPlotter(synthetic_ks4_output)
        fig = plotter.drift_map_plot_matplotlib(
            decimate=False,
            exclude_noise=False,
        )
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()
        np.testing.assert_array_equal(offsets[:, 0], synthetic_data["spike_times"])
        np.testing.assert_array_almost_equal(
            offsets[:, 1], synthetic_data["spike_depths"]
        )
        plt.close(fig)

    @pytest.mark.parametrize(
        "scaling, n_bins",
        [
            ("linear", 20),
            ("log2", 20),
            ("log10", 20),
            ((5.0, 15.0), 20),
            ("linear", 10),
        ],
    )
    def test_scatter_colors_for_all_scaling_modes(
        self, synthetic_ks4_output, scaling, n_bins
    ):
        plotter = DriftPlotter(synthetic_ks4_output)
        fig = plotter.drift_map_plot_matplotlib(
            decimate=False,
            exclude_noise=False,
            amplitude_cmap_scaling=scaling,
            n_color_bins=n_bins,
        )
        ax = fig.axes[0]
        facecolors = ax.collections[0].get_facecolors()

        processed = plotter.data_loader.get_processed_data(
            exclude_noise=False,
            decimate=False,
            filter_amplitude_mode=None,
            filter_amplitude_values=(),
        )
        expected = processed.compute_amplitude_colors(
            scaling, n_bins, unit_normalise=True
        )
        np.testing.assert_array_almost_equal(facecolors, expected)
        plt.close(fig)

    def test_custom_point_size(self, synthetic_ks4_output):
        plotter = DriftPlotter(synthetic_ks4_output)
        fig = plotter.drift_map_plot_matplotlib(
            decimate=False,
            exclude_noise=False,
            point_size=42.0,
        )
        ax = fig.axes[0]
        sizes = ax.collections[0].get_sizes()
        assert np.all(sizes == 42.0)
        plt.close(fig)

    def test_single_axis_without_histogram(self, synthetic_ks4_output):
        plotter = DriftPlotter(synthetic_ks4_output)
        fig = plotter.drift_map_plot_matplotlib(
            decimate=False,
            exclude_noise=False,
            add_histogram_plot=False,
        )
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_title(self, synthetic_ks4_output):
        plotter = DriftPlotter(synthetic_ks4_output)
        fig = plotter.drift_map_plot_matplotlib(
            decimate=False,
            exclude_noise=False,
            title="My Title",
        )
        assert fig._suptitle.get_text() == "My Title"
        plt.close(fig)

    def test_no_title(self, synthetic_ks4_output):
        plotter = DriftPlotter(synthetic_ks4_output)
        fig = plotter.drift_map_plot_matplotlib(
            decimate=False,
            exclude_noise=False,
        )
        assert fig._suptitle is None
        plt.close(fig)


class TestMatplotlibHistogram:
    """When add_histogram_plot=True, a histogram axis should be added."""

    def test_histogram_counts_match_synthetic(
        self, synthetic_ks4_output, synthetic_data
    ):
        """Unweighted histogram — both processed data and plot line should
        match an independent histogram computed from synthetic depths."""
        plotter = DriftPlotter(synthetic_ks4_output)
        fig = plotter.drift_map_plot_matplotlib(
            decimate=False,
            exclude_noise=False,
            add_histogram_plot=True,
            weight_histogram_by_amplitude=False,
        )
        processed = plotter.data_loader.get_processed_data(
            exclude_noise=False,
            decimate=False,
            filter_amplitude_mode=None,
            filter_amplitude_values=(),
        )

        # Independently compute histogram from synthetic depths
        spike_depths = synthetic_data["spike_depths"]
        bin_um = 2
        bins = np.arange(
            spike_depths.min() - bin_um, spike_depths.max() + bin_um, bin_um
        )
        expected_counts, expected_edges = np.histogram(spike_depths, bins=bins)
        expected_centers = (expected_edges[:-1] + expected_edges[1:]) / 2

        # Check processed data matches synthetic
        bin_centers, counts = processed.compute_activity_histogram(False)
        assert counts.sum() == synthetic_data["spike_times"].size
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_almost_equal(bin_centers, expected_centers)

        # Check plot line matches synthetic
        line = fig.axes[0].lines[0]
        np.testing.assert_array_almost_equal(line.get_xdata(), expected_counts)
        np.testing.assert_array_almost_equal(line.get_ydata(), expected_centers)
        plt.close(fig)

    def test_histogram_weighted_matches_synthetic(
        self, synthetic_ks4_output, synthetic_data
    ):
        """Weighted histogram — both processed data and plot line should
        match an independent computation from synthetic depths + amplitudes."""
        plotter = DriftPlotter(synthetic_ks4_output)
        fig = plotter.drift_map_plot_matplotlib(
            decimate=False,
            exclude_noise=False,
            add_histogram_plot=True,
            weight_histogram_by_amplitude=True,
        )
        processed = plotter.data_loader.get_processed_data(
            exclude_noise=False,
            decimate=False,
            filter_amplitude_mode=None,
            filter_amplitude_values=(),
        )

        # Independently compute weighted histogram
        spike_depths = synthetic_data["spike_depths"]
        spike_amplitudes = processed.spike_amplitudes
        bin_um = 2
        bins = np.arange(
            spike_depths.min() - bin_um, spike_depths.max() + bin_um, bin_um
        )
        expected_centers = (bins[:-1] + bins[1:]) / 2

        scaled = (spike_amplitudes - spike_amplitudes.min()) / np.ptp(spike_amplitudes)
        bin_indices = np.digitize(spike_depths, bins, right=True) - 1
        expected_values = np.zeros(bin_indices.max() + 1, dtype=np.float64)
        np.add.at(expected_values, bin_indices, scaled)

        # Check processed data matches synthetic
        bin_centers, values = processed.compute_activity_histogram(True)
        np.testing.assert_array_almost_equal(values, expected_values)
        np.testing.assert_array_almost_equal(bin_centers, expected_centers)

        # Check plot line matches synthetic
        line = fig.axes[0].lines[0]
        np.testing.assert_array_almost_equal(line.get_xdata(), expected_values)
        np.testing.assert_array_almost_equal(line.get_ydata(), expected_centers)
        plt.close(fig)


# ---------------------------------------------------------------------------
# DriftPlotter API signature parity
# ---------------------------------------------------------------------------


def test_interactive_and_matplotlib_share_signature():
    """Shared parameters between interactive and matplotlib methods
    should have the same name, default and type annotation.
    """
    matplotlib_only_params = [
        "self",
        "add_histogram_plot",
        "weight_histogram_by_amplitude",
    ]

    interactive = inspect.signature(DriftPlotter.drift_map_plot_interactive)
    matplotlib = inspect.signature(DriftPlotter.drift_map_plot_matplotlib)

    interactive_params = {
        k: v for k, v in interactive.parameters.items() if k != "self"
    }
    matplotlib_params = {
        k: v
        for k, v in matplotlib.parameters.items()
        if k not in matplotlib_only_params
    }

    assert len(interactive_params) == len(matplotlib_params)

    for name, param in interactive_params.items():
        assert param == matplotlib_params[name]
