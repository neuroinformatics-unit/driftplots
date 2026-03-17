from driftmapviewer_new import get_drift_map_plot, _plot_kilosort_drift_map_raster
import matplotlib.pyplot as plt
import kilosort1_3
import kilosort_4
import helpers
from pathlib import Path
import numpy as np
from driftmap_plot_widget import DriftmapPlotWidget
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import matplotlib.pyplot as plt

# TODO
# ----
# - review KS4 and KS3 method, in particular the scaling and whitening. Check ks4 vs. ks25
#
#
#
# Don't forget to write in docs the caveats on the extraction



# TODO Extra - look into:
# TODO idea: memmap the npy files and decimate ON LOAD
# TODO: KS can wrap template channels around probe boundaries (e.g. channels
#       [0,1,2,380,381,382] for a template near the top). We unwrap these in
#       _get_nonzero_channel_indices. Post about this on the Kilosort GitHub
#       and consider adding an option to disable the unwrap.

# This makes the assumption that there will never be different .csv and .tsv files
# in the same sorter output (this should never happen, there will never even be two).
# Though can be saved as .tsv, it seems the .csv is also tab formatted as far as pandas is concerned.

# TODO: this is super weird, can be improved?
#  if log_transform_amplitudes:
#     spike_amplitudes = np.log(spike_amplitudes)  # TODO: give optional (None, 2 or 10)




# TODO: dont use gain, instead set clim
# TODO: removed localised peaks
# TODO: removed drift event and boundaries

# TODO: it would be really cool and useful to hover over
# the plot and see the template waveform

# TODO idea: memmap the npy files and decimate ON LOAD

# TODO: KS can wrap template channels around probe boundaries (e.g. channels
#       [0,1,2,380,381,382] for a template near the top). We unwrap these in
#       _get_nonzero_channel_indices. Post about this on the Kilosort GitHub
#       and consider adding an option to disable the unwrap.

from PySide6 import QtWidgets
from driftmap_view import DriftMapView

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

FILES = [
    r"X:\aeon\dj_store\ephys-processed\social-ephys0.1-aeon3\ephys_blocks\2024-06-04T11-00-00_2024-06-04T14-00-00\0-95\kilosort4_400\spike_sorting\sorter_output",
    r"X:\aeon\dj_store\ephys-processed\social-ephys0.1-aeon3\ephys_blocks\2024-06-04T13-00-00_2024-06-04T16-00-00\0-95\kilosort4_400\spike_sorting\sorter_output"
]

pooled_amplitudes = helpers.get_pooled_amplitudes(
    FILES
)

min_, max_ = np.percentile(pooled_amplitudes, (97, 99))  # TODO: add exclude noise


panels = []
for file in FILES: #[
 #   r"Y:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\joe\scratch\derivatives\BG_046_26062025\shank_1\sorting\no_motion\sorter_output",
  #  r"Y:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\joe\scratch\derivatives\BG_046_27062025\shank_1\sorting\motion\sorter_output",
   # r"X:\aeon\dj_store\ephys-processed\social-ephys0.1-aeon3\ephys_blocks\2024-06-04T11-00-00_2024-06-04T14-00-00\0-95\kilosort4_400\spike_sorting\sorter_output",
   # r"X:\aeon\dj_store\ephys-processed\social-ephys0.1-aeon3\ephys_blocks\2024-06-04T13-00-00_2024-06-04T16-00-00\0-95\kilosort4_400\spike_sorting\sorter_output",
  #  r"X:\aeon\dj_store\ephys-processed\social-ephys0.1-aeon3\ephys_blocks\2024-06-04T15-00-00_2024-06-04T18-00-00\0-95\kilosort4_400\spike_sorting\sorter_output"


#]:
    plotter = DriftMapView(
        file
    )

    fig = plotter.drift_map_plot_interactive(
        decimate=10,
        exclude_noise=True,
        amplitude_scaling=(min_, max_),
        n_color_bins=25,
        filter_amplitude_mode="absolute",  # "percentile",
        filter_amplitude_values=(min_, max_)
    )

    panels.append(fig)

from multi_session_drift_map import MultiSessionDriftmapWidget
multi = MultiSessionDriftmapWidget(panels)

app.exec()
