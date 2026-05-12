"""
The recording is available for download at:
https://github.com/neuroinformatics-unit/driftplots/tree/main/examples/example_data/recording
"""

from pathlib import Path

import spikeinterface as si
import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_prepro
from spikeinterface.sorters import run_sorter

SAVE_SORTING = True
RECORDING_URL = (
    "https://github.com/neuroinformatics-unit/driftplots/tree/main/"
    "examples/example_data/recording"
)

base_path = Path(__file__).parent
raw_data = base_path / "recording"

if not raw_data.exists():
    raise FileNotFoundError(
        f"The `recording` is not provided with the repository. "
        f"Please download from {RECORDING_URL} "
        f"and save in {raw_data}"
    )

rec = si_extractors.read_spikeglx(raw_data, stream_name="imec0.ap")

rec = si_prepro.phase_shift(rec)
rec = si_prepro.bandpass_filter(rec, freq_min=300, freq_max=6000)
rec = si_prepro.common_reference(rec, operator="median")

if SAVE_SORTING:
    out_path = base_path / "sorting"
    out_path.mkdir()
    sort = run_sorter("kilosort4", rec, folder=out_path)
else:
    sort = si_extractors.read_kilosort(
        base_path / "sorting" / "kilosort4_output" / "sorter_output"
    )

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
