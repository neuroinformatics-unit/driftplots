import numpy as np

def get_sorting_analyzer(analyzer):
    """"""
    random_spike_indices = analyzer.get_extension("random_spikes").data["random_spikes_indices"]
    spike_vector = analyzer.sorting.to_spike_vector()
    spike_times = spike_vector["sample_index"][random_spike_indices] / analyzer.sorting.get_sampling_frequency()
    spike_amplitudes = np.abs(analyzer.get_extension("spike_amplitudes").data["amplitudes"])  # TODO: THIS!
    spike_depths = analyzer.get_extension("spike_locations").data["spike_locations"]["y"]
    spike_templates = spike_vector["unit_index"][random_spike_indices]

    templates_dict = analyzer.get_extension("templates").data
    all_template_keys = templates_dict.keys()
    template_key = list(all_template_keys)[0]

    if len(all_template_keys) != 1:
        warnings.warn(f"Multiple template calculation methods detected. Using {template_key}")

    templates = analyzer.get_extension("templates").data[template_key]
    channel_locations = analyzer.get_channel_locations()

    return spike_times, spike_amplitudes, spike_depths, spike_templates, templates, channel_locations
