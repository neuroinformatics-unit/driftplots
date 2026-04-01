import numpy as np


def get_amplitudes(analyzer, absolute=True):
    amplitudes = analyzer.get_extension("spike_amplitudes").data["amplitudes"]
    if absolute:
        amplitudes = np.abs(amplitudes)
    return amplitudes


def get_sorting_analyzer(analyzer):
    """"""
    random_spike_indices = analyzer.get_extension("random_spikes").data[
        "random_spikes_indices"
    ]
    spike_vector = analyzer.sorting.to_spike_vector()
    spike_times = (
        spike_vector["sample_index"][random_spike_indices]
        / analyzer.sorting.get_sampling_frequency()
    )
    spike_amplitudes = get_amplitudes(analyzer)
    spike_depths = analyzer.get_extension("spike_locations").data["spike_locations"][
        "y"
    ]
    spike_templates = spike_vector["unit_index"][random_spike_indices]

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


#  TODO: USE SPIKE CLUSTERS
#  PASS SPIKE CLUSTERS TO GET_NOISE_MASK


def get_noise_mask(
    exclude_noise: str, spike_templates: np.ndarray, analyzer: si.SortingAnalyzer
) -> np.ndarray[np.bool]:
    """ """
    if exclude_noise is True:
        raise ValueError(
            f"When using SortingAnalyzer, `exclude_noise` must be a string of the "
            f"name of the labels to use, passed to `analyzer.get_sorting_property()."
            f"Properties on this analyzer are: {analyzer.sorting.get_property_keys()}"
        )

    assert isinstance(exclude_noise, str), "`exclude_noise` must be a string"
    labels = analyzer.get_sorting_property(exclude_noise)
    noise_mask = (labels == "noise")[spike_templates]  # TODO: make sure to test this

    return noise_mask
