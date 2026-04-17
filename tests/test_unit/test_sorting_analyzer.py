from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si

from driftplots.data_loader import DataLoader

ANALYZER_PATH = Path(__file__).parent.parent.parent / "examples" / "example_data" / "analyzer.zarr"

pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning",
)


class TestFromSortingAnalyzer:
    """DataLoader should correctly load arrays from a SortingAnalyzer."""

    def test_loaded_arrays_match_analyzer_extensions_and_are_read_only(self):
        analyzer = si.load_sorting_analyzer(ANALYZER_PATH)
        loader = DataLoader(analyzer)

        # Expected values from the analyzer extensions
        random_spike_indices = analyzer.get_extension("random_spikes").data["random_spikes_indices"]
        spike_vector = analyzer.sorting.to_spike_vector()

        expected_times = spike_vector["sample_index"][random_spike_indices] / analyzer.sorting.get_sampling_frequency()
        np.testing.assert_array_equal(loader._spike_times, expected_times)

        expected_amplitudes = np.abs(analyzer.get_extension("spike_amplitudes").data["amplitudes"])
        np.testing.assert_array_equal(loader._spike_amplitudes, expected_amplitudes)

        expected_depths = analyzer.get_extension("spike_locations").data["spike_locations"]["y"]
        np.testing.assert_array_equal(loader._spike_depths, expected_depths)

        expected_templates = spike_vector["unit_index"][random_spike_indices]
        np.testing.assert_array_equal(loader._spike_templates, expected_templates)

        expected_waveforms = analyzer.get_extension("templates").data["average"]
        np.testing.assert_array_equal(loader.templates, expected_waveforms)

        np.testing.assert_array_equal(loader.channel_locations, analyzer.get_channel_locations())

        # All arrays should be read-only
        for arr in (loader._spike_times, loader._spike_amplitudes,
                loader._spike_depths, loader._spike_templates,
                    loader.templates, loader.channel_locations):
            assert not arr.flags.writeable
