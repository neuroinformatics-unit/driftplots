import numpy as np

from driftplots.amplitudes import get_amplitudes


class TestGetAmplitudes:
    """Test get_amplitudes against two synthetic KS4 sessions."""

    def test_single_session_returns_list(self, synthetic_ks4_output, synthetic_data):
        result = get_amplitudes([synthetic_ks4_output])
        expected = synthetic_data["spike_amplitudes"]
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_single_session_values_match_expected(
        self, synthetic_ks4_output, synthetic_data
    ):
        """Single-session amplitudes must match ground-truth spike amplitudes."""
        result = get_amplitudes([synthetic_ks4_output], concatenate=True)
        expected = synthetic_data["spike_amplitudes"]
        assert result.size == synthetic_data["spike_times"].size

        np.testing.assert_array_almost_equal(result, expected)

    def test_two_sessions_list(
        self, synthetic_ks4_output, synthetic_ks4_output_second, synthetic_data
    ):
        """Two sessions should return a list of two distinct arrays."""
        result = get_amplitudes([synthetic_ks4_output, synthetic_ks4_output_second])
        assert len(result) == 2

        expected_1 = synthetic_data["spike_amplitudes"]
        expected_2 = synthetic_data["spike_amplitudes_second"]

        np.testing.assert_array_almost_equal(result[0], expected_1)
        np.testing.assert_array_almost_equal(result[1], expected_2)
        # The two sessions have different amplitude jitter — values must differ
        assert not np.array_equal(result[0], result[1])

    def test_two_sessions_concatenated(
        self, synthetic_ks4_output, synthetic_ks4_output_second, synthetic_data
    ):
        """Concatenated output should be [session1, session2] in order."""
        result = get_amplitudes(
            [synthetic_ks4_output, synthetic_ks4_output_second],
            concatenate=True,
        )
        expected_1 = synthetic_data["spike_amplitudes"]
        expected_2 = synthetic_data["spike_amplitudes_second"]

        assert isinstance(result, np.ndarray)
        n = synthetic_data["spike_times"].size
        assert result.size == n * 2
        np.testing.assert_array_almost_equal(result[:n], expected_1)
        np.testing.assert_array_almost_equal(result[n:], expected_2)

    def test_exclude_noise(self, synthetic_ks4_output, synthetic_ks4_output_second):
        """exclude_noise=True should remove noise spikes from both sessions."""
        result = get_amplitudes(
            [synthetic_ks4_output, synthetic_ks4_output_second],
            exclude_noise=True,
            concatenate=True,
        )
        full = get_amplitudes(
            [synthetic_ks4_output, synthetic_ks4_output_second],
            exclude_noise=False,
            concatenate=True,
        )
        assert result.size < full.size
