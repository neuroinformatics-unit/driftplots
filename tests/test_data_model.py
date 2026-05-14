"""Tests for DataModel methods against synthetic ground truth."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from driftplots.driftplotter import DriftPlotter


def _expected_colors(spike_amplitudes, scaling, n_bins, unit_normalise):
    """Independently replicate compute_amplitude_colors logic."""
    amp_values = np.abs(spike_amplitudes)

    if scaling == "log2":
        amp_values = np.log2(np.maximum(amp_values, np.finfo(float).eps))
    elif scaling == "log10":
        amp_values = np.log10(np.maximum(amp_values, np.finfo(float).eps))

    amp_min, amp_max = amp_values.min(), amp_values.max()

    color_bins = np.linspace(amp_min, amp_max, n_bins + 1)
    gray_colors = plt.get_cmap("gray")(np.linspace(0, 1, n_bins))[::-1]
    bin_indices = np.clip(
        np.searchsorted(color_bins, amp_values, side="right") - 1,
        0,
        n_bins - 1,
    )
    colors = gray_colors[bin_indices]

    if not unit_normalise:
        colors *= 255
        colors = colors.astype(np.uint8)

    return colors


SCALINGS = ["linear", "log2", "log10"]
N_BINS = 20


class TestComputeAmplitudeColors:
    """compute_amplitude_colors should map amplitudes to grey-scale RGBA
    values that match an independent reimplementation of the algorithm."""

    @pytest.fixture()
    def processed(self, synthetic_ks4_output):
        plotter = DriftPlotter(synthetic_ks4_output)
        return plotter._data_loader.get_processed_data(
            good_units_only=False,
            decimate=False,
            filter_amplitude_mode=None,
            filter_amplitude_values=(),
            verbose=False,
        )

    @pytest.mark.parametrize("scaling", SCALINGS)
    def test_colors_match_ground_truth_uint8(self, processed, scaling):
        """Default (unit_normalise=False) returns uint8 RGBA matching
        an independent computation from the synthetic amplitudes."""
        result = processed.compute_amplitude_colors(scaling, N_BINS)
        expected = _expected_colors(processed.spike_amplitudes, scaling, N_BINS, False)

        assert result.dtype == np.uint8
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("scaling", SCALINGS)
    def test_colors_match_ground_truth_unit_normalised(self, processed, scaling):
        """unit_normalise=True returns float RGBA in [0, 1]."""
        result = processed.compute_amplitude_colors(
            scaling, N_BINS, unit_normalise=True
        )
        expected = _expected_colors(processed.spike_amplitudes, scaling, N_BINS, True)

        assert result.dtype == np.float64
        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result, expected)

    def test_tuple_scaling_fixes_range(self, processed):
        """A (min, max) tuple should override the data-derived range."""
        fixed_range = (0.5, 1.5)
        result = processed.compute_amplitude_colors(fixed_range, N_BINS)

        # Independently compute with fixed range
        amp_values = np.abs(processed.spike_amplitudes)
        color_bins = np.linspace(0.5, 1.5, N_BINS + 1)
        gray_colors = plt.get_cmap("gray")(np.linspace(0, 1, N_BINS))[::-1]
        bin_indices = np.clip(
            np.searchsorted(color_bins, amp_values, side="right") - 1,
            0,
            N_BINS - 1,
        )
        expected = (gray_colors[bin_indices] * 255).astype(np.uint8)

        np.testing.assert_array_equal(result, expected)

    def test_different_scalings_produce_different_colors(self, processed):
        """Confidence check: different scaling modes should not all give
        identical colours (unless amplitudes are degenerate)."""
        results = {
            s: processed.compute_amplitude_colors(s, N_BINS, unit_normalise=True)
            for s in SCALINGS
        }
        assert not np.array_equal(results["linear"], results["log2"])
        assert not np.array_equal(results["linear"], results["log10"])
