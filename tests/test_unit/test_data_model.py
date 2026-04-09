from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from driftplots.data_loader import DataLoader
from driftplots.data_model import DataModel

matplotlib.use("Agg")

SORTER_OUTPUT = Path(__file__).parent.parent.parent / "examples" / "example_data" / "sorting" / "sorter_output"


@pytest.fixture()
def data_model():
    loader = DataLoader(SORTER_OUTPUT)
    return loader.get_processed_data(
        exclude_noise=False, decimate=False,
        filter_amplitude_mode=None, filter_amplitude_values=(),
    )


# ---------------------------------------------------------------------------
# compute_amplitude_colors
# ---------------------------------------------------------------------------
class TestAmplitudeColors:

    def test_output_shape(self, data_model):
        colors = data_model.compute_amplitude_colors("linear", 20)
        assert colors.shape == (data_model.spike_times.size, 4)

    def test_uint8_by_default(self, data_model):
        colors = data_model.compute_amplitude_colors("linear", 20)
        assert colors.dtype == np.uint8

    def test_unit_normalise(self, data_model):
        colors = data_model.compute_amplitude_colors("linear", 20, unit_normalise=True)
        assert colors.dtype == np.float64
        assert np.all(colors >= 0) and np.all(colors <= 1)

    def test_log2_scaling(self, data_model):
        colors_lin = data_model.compute_amplitude_colors("linear", 20)
        colors_log = data_model.compute_amplitude_colors("log2", 20)
        # Different scaling produces different colour assignments
        assert not np.array_equal(colors_lin, colors_log)

    def test_log10_scaling(self, data_model):
        colors_lin = data_model.compute_amplitude_colors("linear", 20)
        colors_log = data_model.compute_amplitude_colors("log10", 20)
        assert not np.array_equal(colors_lin, colors_log)

    def test_explicit_range_tuple(self, data_model):
        amps = np.abs(data_model.spike_amplitudes)
        low, high = np.percentile(amps, (25, 75))
        colors = data_model.compute_amplitude_colors((low, high), 20)
        assert colors.shape == (data_model.spike_times.size, 4)

    def test_n_bins_changes_output(self, data_model):
        colors_5 = data_model.compute_amplitude_colors("linear", 5)
        colors_50 = data_model.compute_amplitude_colors("linear", 50)
        unique_5 = np.unique(colors_5, axis=0).shape[0]
        unique_50 = np.unique(colors_50, axis=0).shape[0]
        assert unique_50 >= unique_5


# ---------------------------------------------------------------------------
# compute_activity_histogram
# ---------------------------------------------------------------------------
class TestActivityHistogram:

    def test_returns_two_arrays(self, data_model):
        bin_centers, values = data_model.compute_activity_histogram(False)
        assert isinstance(bin_centers, np.ndarray)
        assert isinstance(values, np.ndarray)

    def test_bin_centers_and_values_same_length(self, data_model):
        bin_centers, values = data_model.compute_activity_histogram(False)
        assert bin_centers.size == values.size

    def test_counts_sum_to_num_spikes(self, data_model):
        _, counts = data_model.compute_activity_histogram(False)
        assert counts.sum() == data_model.spike_times.size

    def test_weighted_differs_from_counts(self, data_model):
        _, counts = data_model.compute_activity_histogram(False)
        _, weighted = data_model.compute_activity_histogram(True)
        assert not np.array_equal(counts, weighted)

    def test_weighted_values_non_negative(self, data_model):
        _, weighted = data_model.compute_activity_histogram(True)
        assert np.all(weighted >= 0)


# ---------------------------------------------------------------------------
# get_template_heatmap
# ---------------------------------------------------------------------------
class TestTemplateHeatmap:

    def test_heatmap_returns_2d(self, data_model):
        template = data_model.get_template_heatmap(0, "heatmap")
        assert template.ndim == 2

    def test_heatmap_all_channels_returns_2d(self, data_model):
        template = data_model.get_template_heatmap(0, "heatmap_all_channels")
        assert template.ndim == 2

    def test_all_channels_wider_or_equal(self, data_model):
        """'all channels' mode should show at least as many channels."""
        t_signal = data_model.get_template_heatmap(0, "heatmap")
        t_all = data_model.get_template_heatmap(0, "heatmap_all_channels")
        assert t_all.shape[1] >= t_signal.shape[1]

    def test_same_num_samples(self, data_model):
        t_signal = data_model.get_template_heatmap(0, "heatmap")
        t_all = data_model.get_template_heatmap(0, "heatmap_all_channels")
        assert t_signal.shape[0] == t_all.shape[0]

    def test_different_spikes_same_cluster_same_template(self, data_model):
        """Spikes in the same cluster should return the same template."""
        cluster_0_indices = np.where(data_model.spike_clusters == data_model.spike_clusters[0])[0]
        if len(cluster_0_indices) < 2:
            pytest.skip("Need at least 2 spikes in the same cluster")
        t0 = data_model.get_template_heatmap(cluster_0_indices[0], "heatmap")
        t1 = data_model.get_template_heatmap(cluster_0_indices[1], "heatmap")
        np.testing.assert_array_equal(t0, t1)


# ---------------------------------------------------------------------------
# get_scatter_data / get_template_id
# ---------------------------------------------------------------------------
class TestScatterAndTemplateId:

    def test_scatter_data_returns_three(self, data_model):
        times, depths, amps = data_model.get_scatter_data()
        assert times is data_model.spike_times
        assert depths is data_model.spike_depths
        assert amps is data_model.spike_amplitudes

    def test_template_id_matches_clusters(self, data_model):
        for i in [0, 10, data_model.spike_times.size - 1]:
            assert data_model.get_template_id(i) == data_model.spike_clusters[i]
