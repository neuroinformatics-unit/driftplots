from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si

from driftplots.data_loader import DataLoader

ANALYZER_PATH = (
    Path(__file__).parent.parent / "examples" / "example_data" / "analyzer.zarr"
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning",
)


class TestFromSortingAnalyzer:
    """DataLoader should correctly load arrays from a SortingAnalyzer."""

    def test_loaded_arrays_match_analyzer_extensions(self):
        analyzer = si.load_sorting_analyzer(ANALYZER_PATH)
        loader = DataLoader(analyzer, verbose=False)

        spike_vector = analyzer.sorting.to_spike_vector()

        expected_times = (
            spike_vector["sample_index"] / analyzer.sorting.get_sampling_frequency()
        )
        np.testing.assert_array_equal(loader._spike_times, expected_times)

        expected_amplitudes = np.abs(
            analyzer.get_extension("spike_amplitudes").data["amplitudes"]
        )
        np.testing.assert_array_equal(loader._spike_amplitudes, expected_amplitudes)

        expected_depths = analyzer.get_extension("spike_locations").data[
            "spike_locations"
        ]["y"]
        np.testing.assert_array_equal(loader._spike_depths, expected_depths)

        expected_templates = spike_vector["unit_index"]
        np.testing.assert_array_equal(loader._spike_templates, expected_templates)

        expected_waveforms = analyzer.get_extension("templates").data["average"]
        np.testing.assert_array_equal(loader.templates, expected_waveforms)

        np.testing.assert_array_equal(
            loader.channel_locations, analyzer.get_channel_locations()
        )

        # All arrays should be read-only
        for arr in (
            loader._spike_times,
            loader._spike_amplitudes,
            loader._spike_depths,
            loader._spike_templates,
            loader.templates,
            loader.channel_locations,
        ):
            assert not arr.flags.writeable

    def test_loaded_spikes_ignore_random_spikes_subset(self):
        analyzer = si.load_sorting_analyzer(ANALYZER_PATH).copy()
        spike_vector = analyzer.sorting.to_spike_vector()

        random_spikes = analyzer.get_extension("random_spikes")
        random_spikes.data["random_spikes_indices"] = random_spikes.data[
            "random_spikes_indices"
        ][::2]
        assert random_spikes.data["random_spikes_indices"].size < spike_vector.size

        loader = DataLoader(analyzer, verbose=False)

        expected_times = (
            spike_vector["sample_index"] / analyzer.sorting.get_sampling_frequency()
        )
        np.testing.assert_array_equal(loader._spike_times, expected_times)
        np.testing.assert_array_equal(
            loader._spike_templates, spike_vector["unit_index"]
        )
        assert loader._spike_times.size == spike_vector.size

    def test_good_units_only_keeps_kslabel_good_units(self):
        analyzer = si.load_sorting_analyzer(ANALYZER_PATH)
        loader = DataLoader(analyzer, verbose=False)

        result = loader.get_processed_data(
            good_units_only="KSLabel",
            decimate=False,
            filter_amplitude_mode=None,
            filter_amplitude_values=(),
            verbose=False,
        )

        labels = analyzer.sorting.get_property("KSLabel")
        expected_keep = labels[loader._spike_templates] == "good"

        np.testing.assert_array_equal(
            result.spike_times, loader._spike_times[expected_keep]
        )
        np.testing.assert_array_equal(
            result.spike_amplitudes, loader._spike_amplitudes[expected_keep]
        )
        np.testing.assert_array_equal(
            result.spike_depths, loader._spike_depths[expected_keep]
        )
        assert (labels[result.spike_templates] == "good").all()

    def test_good_units_only_true_raises_for_sorting_analyzer(self):
        analyzer = si.load_sorting_analyzer(ANALYZER_PATH)
        loader = DataLoader(analyzer, verbose=False)

        with pytest.raises(ValueError, match="must be a string"):
            loader.get_processed_data(
                good_units_only=True,
                decimate=False,
                filter_amplitude_mode=None,
                filter_amplitude_values=(),
                verbose=False,
            )
