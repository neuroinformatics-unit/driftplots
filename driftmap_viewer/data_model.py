import warnings

import matplotlib.pyplot as plt
import numpy as np


class DataModel:
    def __init__(
        self,
        sorter,
        spike_times,
        spike_amplitudes,
        spike_depths,
        spike_templates,
        templates,
        channel_locations,
    ):
        self.sorter = sorter
        self.spike_times = spike_times
        self.spike_depths = spike_depths
        self.spike_amplitudes = spike_amplitudes
        self.spike_templates = spike_templates
        self.templates = templates
        self.channel_locations = channel_locations

    def get_scatter_data(self):
        return self.spike_times, self.spike_depths, self.spike_amplitudes

    def get_template_id(self, spike_idx):
        return self.spike_templates[spike_idx]

    def get_template_heatmap(self, spike_index, view_mode):
        """ """
        # Extract the template for this spike
        template_idx = self.spike_templates[spike_index]
        template = self.templates[template_idx, :, :]
        mid_idx = int(template.shape[0] / 2)

        # Next we need to find the shank the template is on. For KS,
        # signal can also be found on other shanks but this confuses the
        # visualisation
        # Find the channel with maximum signal
        max_chan_idx = np.argmax(np.max(np.abs(template), axis=0))
        max_signal_x_loc = self.channel_locations[max_chan_idx, 0]

        # Find other channels in the shank column. Because we are working on
        # KS outputs, we have no knowledge of the probe, so we have to guess.
        # Based on the column vs. shank space for a number of popular probes
        # (NP1: 1 shank, 70um across, NP2: 250um between shank, shank width ~70um,
        # Cambridge Neurotech: shank widths ~80 µm, shank spacing ~200um+,
        # NeuroNexus: does have some shank widths at 100-120um)), in which
        # this will fail. The simplest solution is to document and
        # down the line expose this parameter.
        COL_CUTOFF_UM = 125

        chan_x_locs = np.unique(self.channel_locations[:, 0])

        chan_x_spacings = np.diff(chan_x_locs)
        if np.any(
            np.logical_and(chan_x_spacings > COL_CUTOFF_UM, chan_x_spacings < 150)
        ):
            warnings.warns(
                f"The spacings between x-locations: {chan_x_spacings} makes it difficult to distinguish"
                f"between channel and shank spacing. The cutoff is {COL_CUTOFF_UM}, less than"
                f"this is assumed to be two columns of channels on the same shank."
            )

        valid_pos = chan_x_locs[np.abs(chan_x_locs - max_signal_x_loc) < COL_CUTOFF_UM]

        shank_select = np.zeros(self.channel_locations.shape[0], dtype=bool)
        for pos in valid_pos:
            shank_select = np.logical_or(
                shank_select, self.channel_locations[:, 0] == pos
            )

        # Often the contact positions are not organised contiguous
        # along the y-dimension and need resorting
        sort_idx = np.argsort(self.channel_locations[shank_select, 1], axis=0)

        # Select the shank of interest ordered by depth
        template = template[:, shank_select]
        template = template[:, sort_idx]

        # Either display only the channels with signal on, or all channels but
        # non-signal channels are empty. Using the threshold ==0 works well for
        # SI analyzer and whitened KS templates, less well for un-whitened KS
        # templates which have nonzero signal on all channel, but for which no
        # clear threshold exists.
        if view_mode == "heatmap_all_channels":
            template = template.copy()
            template[:, template[mid_idx, :] == 0] = np.nan
        else:
            contains_data_idx = np.where(template[mid_idx, :] != 0)[0]
            template = template[:, contains_data_idx]

        return template

    # TODO: CHECK THIS
    def compute_amplitude_colors(
        self, amplitude_scaling, n_color_bins, unit_normalise=False
    ):
        """Map spike amplitudes to RGBA colours via grey-scale binning.

        Parameters
        ----------
        amplitude_scaling : {"linear", "log2", "log10"} | tuple
            Scaling mode.  A 2-tuple ``(min, max)`` fixes the colour
            range explicitly.
        n_color_bins : int
            Number of grey-scale bins.

        Returns
        -------
        np.ndarray
            (num_spikes, 4) uint8 RGBA values
        """
        amp_values = np.abs(self.spike_amplitudes)

        if isinstance(amplitude_scaling, tuple):
            amp_min, amp_max = amplitude_scaling
        else:
            if amplitude_scaling == "log2":
                amp_values = np.log2(np.maximum(amp_values, np.finfo(float).eps))

            elif amplitude_scaling == "log10":
                amp_values = np.log10(np.maximum(amp_values, np.finfo(float).eps))

            amp_min, amp_max = amp_values.min(), amp_values.max()

        color_bins = np.linspace(amp_min, amp_max, n_color_bins)
        gray_colors = plt.get_cmap("gray")(np.linspace(0, 1, n_color_bins))[::-1]
        bin_indices = np.clip(
            np.searchsorted(color_bins, amp_values, side="right") - 1,
            0,
            n_color_bins - 2,
        )

        colors = gray_colors[bin_indices]

        if not unit_normalise:
            colors *= 255
            colors = colors.astype(np.uint8)

        return colors

    def compute_activity_histogram(
        self, weight_histogram_by_amplitude: bool
    ) -> tuple[np.ndarray, ...]:
        """
        Compute the activity histogram for the kilosort drift map's left-side plot.

        Parameters
        ----------
        weight_histogram_by_amplitude : bool
            If `True`, the spike amplitudes are taken into consideration when generating the
            histogram. The amplitudes are scaled to the range [0, 1] then summed for each bin,
            to generate the histogram values. If `False`, counts (i.e. num spikes per bin)
            are used.

        Returns
        -------
        bin_centers : np.ndarray
            The spatial bin centers (probe depth) for the histogram.
        values : np.ndarray
            The histogram values. If `weight_histogram_by_amplitude` is `False`, these
            values represent are counts, otherwise they are counts weighted by amplitude.
        """
        # `spike amplitudes should be high precision as many values are summed.
        spike_amplitudes = self.spike_amplitudes.astype(np.float64)

        bin_um = 2
        bins = np.arange(
            self.spike_depths.min() - bin_um, self.spike_depths.max() + bin_um, bin_um
        )
        values, bins = np.histogram(self.spike_depths, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if weight_histogram_by_amplitude:
            bin_indices = np.digitize(self.spike_depths, bins, right=True) - 1
            values = np.zeros(bin_indices.max() + 1, dtype=np.float64)
            scaled_spike_amplitudes = (
                spike_amplitudes - spike_amplitudes.min()
            ) / np.ptp(spike_amplitudes)
            np.add.at(values, bin_indices, scaled_spike_amplitudes)

        return bin_centers, values
