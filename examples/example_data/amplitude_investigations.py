from pathlib import Path

import spikeinterface as si
import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_prepro
from spikeinterface.sorters import run_sorter

SAVE_SORTING = False

base_path = Path(__file__).parent
raw_data = base_path / "recording"

rec = si_extractors.read_spikeglx(raw_data, stream_name="imec0.ap")

rec = si_prepro.phase_shift(rec)
rec = si_prepro.bandpass_filter(rec, freq_min=300, freq_max=6000)
rec = si_prepro.common_reference(rec, operator="median")

if SAVE_SORTING:
    #  out_path = base_path / "sorting"
    # out_path.mkdir()
    sort = run_sorter("kilosort4", rec, folder=base_path / "sorting")
else:
    sort = si_extractors.read_kilosort(base_path / "sorting" / "sorter_output")

# Analyzer Method

if True:
    analyzer = si.create_sorting_analyzer(sort, rec)
    analyzer.compute(
        "random_spikes",
        method="uniform",
        # This determines the number of spikes that
        # will appear on the SI drift plot
        max_spikes_per_unit=1_000_000,
    )
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
    analyzer.compute("templates", operators=["average"])
    analyzer.compute("spike_amplitudes")
    analyzer.compute("spike_locations")
    analyzer.save_as(folder=base_path / "analyzer.zarr", format="zarr")

analyzer = si.load_sorting_analyzer(folder=base_path / "analyzer.zarr", format="zarr")

# SI
import numpy as np

amplitudes_si = np.abs(analyzer.get_extension("spike_amplitudes").data["amplitudes"])

# amplitudes.npy
sorter_output = Path(
    r"C:\Users\Jzimi\git-repos\driftplots\examples\example_data\sorting\sorter_output"
)
amplitudes_npy = np.load(
    r"C:\Users\Jzimi\git-repos\driftplots\examples\example_data\sorting\sorter_output\amplitudes.npy"
)

# Kilosort method from XXX
amplitudes = np.load(sorter_output / "amplitudes.npy")
templates = np.load(sorter_output / "templates.npy")  # rename unwhiten?
spike_templates = np.load(sorter_output / "spike_templates.npy")
template_ptp = np.max(templates, axis=1) - np.min(templates, axis=1)
template_max_peaks = np.max(template_ptp, axis=1)
kilosort_method_1 = template_max_peaks[spike_templates] * amplitudes

# Kilosort reconstruction of waveforms (Chris Halcrow)
ops = np.load(
    r"C:\Users\Jzimi\git-repos\driftplots\examples\example_data\sorting\sorter_output\ops.npy",
    allow_pickle=True,
)
wPCA = ops.tolist()["wPCA"]
pc_inds = np.load(sorter_output / "pc_feature_ind.npy")
pcs = np.load(Path(sorter_output) / "pc_features.npy")
wh_inv = np.load(sorter_output / "whitening_mat_inv.npy")
spike_templates = np.load(sorter_output / "spike_templates.npy")

whitened_waveforms = np.einsum("ji,ajk->aik", wPCA, pcs)

unwhite_waveforms = np.zeros_like(whitened_waveforms)
for spike_idx in range(whitened_waveforms.shape[0]):
    spike_chans = pc_inds[spike_templates[spike_idx]]
    white_max_for_spike = wh_inv[spike_chans][:, spike_chans]
    unwhite_waveforms[spike_idx, :, :] = (
        whitened_waveforms[spike_idx, :, :] @ white_max_for_spike
    )

if False:
    import matplotlib.pyplot as plt

    for i in range(100):
        plt.imshow(whitened_waveforms[i, :, :].T)
        plt.show()

        plt.imshow(unwhite_waveforms[i, :, :].T)
        plt.show()

kilosort_method_wavs = np.max(np.max(unwhite_waveforms, axis=1), axis=1)

# compare


import matplotlib.pyplot as plt
import numpy as np

# Flatten in case some arrays are shaped (n, 1)
amps = {
    "kilosort_method_wavs": np.asarray(kilosort_method_wavs).ravel(),
    "kilosort_method_1": np.asarray(kilosort_method_1).ravel(),
    "amplitudes_npy": np.asarray(amplitudes_npy).ravel(),
    "amplitudes_si": np.asarray(amplitudes_si).ravel(),
}

# Drop NaN/inf just in case
amps = {k: v[np.isfinite(v)] for k, v in amps.items()}

# Shared bin edges so the histograms are directly comparable
all_vals = np.concatenate(list(amps.values()))
bins = np.histogram_bin_edges(all_vals, bins=200)

# Overlaid histograms
plt.figure(figsize=(10, 6))
for name, values in amps.items():
    plt.hist(values, bins=bins, histtype="step", linewidth=2, density=True, label=name)

plt.xlabel("Amplitude")
plt.ylabel("Density")
plt.title("Amplitude histograms")
plt.legend()
plt.tight_layout()
plt.show()

spike_idx = 100
spike_vector = analyzer.sorting.to_spike_vector()

peak_idx = spike_vector["sample_index"][spike_idx]

traces = analyzer.recording.get_traces(
    start_frame=peak_idx - 30, end_frame=peak_idx + 30
)

reorder_idx = si.order_channels_by_depth(analyzer.recording)[0]
spike_y = analyzer.get_extension("spike_locations").data["spike_locations"]["y"][
    spike_idx
]
spike_x = analyzer.get_extension("spike_locations").data["spike_locations"]["x"][
    spike_idx
]

locs = analyzer.recording.get_channel_locations()[reorder_idx]

# Compute distances to the spike position
distances = np.sqrt((locs[:, 0] - spike_x) ** 2 + (locs[:, 1] - spike_y) ** 2)

# Find the index of the closest channel
spike_chan_idx = np.argmin(distances)

breakpoint()

reorder_traces = traces[:, reorder_idx]

plt.plot(reorder_traces[:, spike_chan_idx].T)
plt.show()

np.max(reorder_traces[:, spike_chan_idx])

kilosort_method_wavs[spike_idx]
kilosort_method_1[spike_idx]
amplitudes_si[spike_idx]

# I feel like this is wrong...
breakpoint()
