from __future__ import annotations

import warnings

import numpy as np
import spikeinterface as si


def get_sorting_analyzer(
    analyzer: si.SortingAnalyzer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the required data from the sorting analyzer. Note that this
    will not get all detected spikes, but rather the number of spikes
    specified when creating the analyzer, `max_spikes_per_unit`.
    """
    random_spike_indices = analyzer.get_extension("random_spikes").data[
        "random_spikes_indices"
    ]
    spike_vector = analyzer.sorting.to_spike_vector()
    spike_times = (
        spike_vector["sample_index"][random_spike_indices]
        / analyzer.sorting.get_sampling_frequency()
    )
    spike_amplitudes = np.abs(
        analyzer.get_extension("spike_amplitudes").data["amplitudes"]
    )

    spike_depths = analyzer.get_extension("spike_locations").data["spike_locations"][
        "y"
    ]
    spike_templates = spike_vector["unit_index"][random_spike_indices]

    # Get the templates, assume only one method was used. If multiple
    # methods were used, use the first and throw a warning. If people
    # want this exposed, it can be exposed, but at the moment seems too particular.
    templates_dict = analyzer.get_extension("templates").data
    all_template_keys = templates_dict.keys()
    template_key = list(all_template_keys)[0]

    if len(all_template_keys) != 1:
        warnings.warn(
            f"Multiple template calculation methods detected. Using {template_key}"
        )

    templates = analyzer.get_extension("templates").data[template_key]
    channel_locations = analyzer.get_channel_locations()

    return (
        spike_times,
        spike_amplitudes,
        spike_depths,
        spike_templates,
        templates,
        channel_locations,
    )


def get_noise_mask(
    exclude_noise: bool | str,
    spike_templates: np.ndarray,
    analyzer: si.SortingAnalyzer,
) -> np.ndarray:
    """ """
    if exclude_noise is True:
        raise ValueError(
            f"When using SortingAnalyzer, `exclude_noise` must be a string of the "
            f"name of the labels to use, passed to `analyzer.get_sorting_property()."
            f"Properties on this analyzer are: {analyzer.sorting.get_property_keys()}"
        )

    assert isinstance(exclude_noise, str), "`exclude_noise` must be a string"
    labels = analyzer.get_sorting_property(exclude_noise)

    if labels is None:
        raise ValueError(
            f"The analyzer does not contain a sorting property called: {exclude_noise}"
        )

    noise_mask = (labels == "noise")[spike_templates]

    return noise_mask
