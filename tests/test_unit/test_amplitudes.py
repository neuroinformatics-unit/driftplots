from pathlib import Path

import numpy as np
import pytest

from driftplots.amplitudes import get_amplitudes

SORTER_OUTPUT = Path(__file__).parent.parent.parent / "examples" / "example_data" / "sorting" / "sorter_output"


class TestGetAmplitudes:
    """Test the get_amplitudes utility function."""

    def test_returns_list_by_default(self):
        result = get_amplitudes([SORTER_OUTPUT])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)

    def test_concatenate_returns_array(self):
        result = get_amplitudes([SORTER_OUTPUT], concatenate=True)
        assert isinstance(result, np.ndarray)

    def test_multiple_sessions_list(self):
        result = get_amplitudes([SORTER_OUTPUT, SORTER_OUTPUT])
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], result[1])

    def test_multiple_sessions_concatenated(self):
        single = get_amplitudes([SORTER_OUTPUT], concatenate=True)
        double = get_amplitudes([SORTER_OUTPUT, SORTER_OUTPUT], concatenate=True)
        assert double.size == single.size * 2

    def test_amplitudes_match_loader(self):
        """Values should match what DataLoader returns."""
        from driftplots.data_loader import DataLoader

        loader = DataLoader(SORTER_OUTPUT)
        data = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        result = get_amplitudes([SORTER_OUTPUT], concatenate=True)
        np.testing.assert_array_equal(result, data.spike_amplitudes)
