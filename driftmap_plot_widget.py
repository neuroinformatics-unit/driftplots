import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import matplotlib.pyplot as plt

# 1) test out with real data, think with ideas
# 2) handle row / column for lots of data
# 3) fix y-lim scaling when your switch sides
# 4) see if the plots can be easily saved, it might be beter just to use matplotlib



pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOption("antialias", True)


class DriftmapPlotWidget(QtWidgets.QWidget):
    def __init__(self, spike_times, spike_amplitudes, spike_depths,
                 spike_templates, templates, log_transform_amplitudes):
        super().__init__()

        self.spike_times = spike_times
        self.spike_amplitudes = spike_amplitudes
        self.spike_depths = spike_depths
        self.spike_templates = spike_templates
        self.templates = templates
        self.log_transform_amplitudes = log_transform_amplitudes

        self.cfgs = {
            "right_panel_view_mode": "max_waveform",
            "left_panel_y_axis": {
                "on": False,
                "y_max": 200,
                "y_min": -200,
            },
        }

        self.selected_spot = None

        self.resize(1400, 820)

        # Core layout
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer_layout.addWidget(splitter, stretch=1)

        win_left = pg.GraphicsLayoutWidget()
        splitter.addWidget(win_left)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        win_right = pg.GraphicsLayoutWidget()
        right_layout.addWidget(win_right, stretch=1)
        splitter.addWidget(right_widget)
        total = splitter.width()
        splitter.setSizes([int(total * 0.75), int(total * 0.25)])

        # Controls Bar — below the full splitter
        # --------------------------------------------------------------------------------------------------------------
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(6, 6, 6, 6)
        controls_layout.setSpacing(6)

        # --- Radio buttons ---
        radio_row = QtWidgets.QWidget()
        radio_layout = QtWidgets.QHBoxLayout(radio_row)
        radio_layout.setContentsMargins(0, 0, 0, 0)
        radio_layout.setSpacing(12)

        self.radio_max_wf = QtWidgets.QRadioButton("Max waveform")
        self.radio_heatmap = QtWidgets.QRadioButton("Heatmap")
        self.radio_heatmap_all = QtWidgets.QRadioButton("Heatmap (all channels)")
        self.radio_max_wf.setChecked(True)

        self._view_radio_group = QtWidgets.QButtonGroup(self)
        self._view_radio_group.addButton(self.radio_max_wf, 0)
        self._view_radio_group.addButton(self.radio_heatmap, 1)
        self._view_radio_group.addButton(self.radio_heatmap_all, 2)

        radio_layout.addWidget(self.radio_max_wf)
        radio_layout.addWidget(self.radio_heatmap)
        radio_layout.addWidget(self.radio_heatmap_all)
        radio_layout.addStretch()
        controls_layout.addWidget(radio_row)

        # Page 0 — Max waveform
        max_wf_page = QtWidgets.QWidget()
        max_wf_layout = QtWidgets.QHBoxLayout(max_wf_page)
        max_wf_layout.setContentsMargins(0, 0, 0, 0)
        max_wf_layout.setSpacing(8)

        fix_ylim_cb = QtWidgets.QCheckBox("Fix y-limits")
        self.ymin_spin = QtWidgets.QDoubleSpinBox()
        self.ymax_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.ymin_spin, self.ymax_spin):
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(1)
            spin.setFixedWidth(90)
            spin.setMinimumWidth(100)
            spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setEnabled(self.cfgs["left_panel_y_axis"]["on"])
        self.ymin_spin.setValue(self.cfgs["left_panel_y_axis"]["y_min"])
        self.ymax_spin.setValue(self.cfgs["left_panel_y_axis"]["y_max"])

        max_wf_layout.addWidget(fix_ylim_cb)
        max_wf_layout.addWidget(QtWidgets.QLabel("Y min:"))
        max_wf_layout.addWidget(self.ymin_spin)
        max_wf_layout.addWidget(QtWidgets.QLabel("Y max:"))
        max_wf_layout.addWidget(self.ymax_spin)
        max_wf_layout.addStretch()

        controls_layout.addWidget(max_wf_page)

        # Add controls below the splitter, spanning full width
        outer_layout.addWidget(controls_widget)

        # Scatter Plot
        # --------------------------------------------------------------------------------------------------------------

        self.p_scatter = win_left.addPlot(row=0, col=0)
        self.p_scatter.setLabel("bottom", "Time (s)")
        self.p_scatter.setLabel("left", "Depth (µm)")
        self.p_scatter.showGrid(x=False, y=False)

        # set amplitude colors
        n_color_bins = 20
        amp_min, amp_max = (
        spike_amplitudes.min(),
        spike_amplitudes.max(),
    )
        assert amp_min >= 0
        assert amp_max >= 0

        if self.log_transform_amplitudes:
            amp_min = np.log(amp_min)
            amp_max = np.log(amp_max)
            spike_amplitudes = np.log(spike_amplitudes)

        color_bins = np.linspace(amp_min, amp_max, n_color_bins)
        gray_colors = plt.get_cmap("gray")(np.linspace(0, 1, n_color_bins))[::-1]
        bin_indices = np.clip(np.searchsorted(color_bins, spike_amplitudes, side="right") - 1, 0, n_color_bins - 2)
        rgba_float = gray_colors[bin_indices]

        # set axis limits
        x_pad = (spike_times.max() - spike_times.min()) * 0.025
        y_pad = (spike_depths.max() - spike_depths.min()) * 0.05
        self.p_scatter.getViewBox().setLimits(
            xMin=spike_times.min() - x_pad,
            xMax=spike_times.max() + x_pad,
            yMin=spike_depths.min() - y_pad,
            yMax=spike_depths.max() + y_pad,
        )

        # create plot
        point_size = 5.0
        self.scatter = pg.ScatterPlotItem(
            spike_times, spike_depths,
            pxMode=True, size=point_size, hoverable=True, antialias=True, data=np.arange(spike_times.size), brush=rgba_float*255, pen=None
        )

        # Template Plot
        # --------------------------------------------------------------------------------------------------------------

        self.panel_plot = win_right.addPlot(row=0, col=0)
        self.panel_plot.setLabel("bottom", "sample")
        self.panel_plot.setLabel("left", "amplitude")
        self.panel_plot.showGrid(x=False, y=False)

        # Tidying Up
        # --------------------------------------------------------------------------------------------------------------

        self.p_scatter.addItem(self.scatter)

        # Connections
        # --------------------------------------------------------------------------------------------------------------

        self.ymin_spin.valueChanged.connect(self.handle_y_spinbox_min)
        self.ymax_spin.valueChanged.connect(self.handle_y_spinbox_max)
        fix_ylim_cb.toggled.connect(self.handle_fix_ylim_cb)
        self.scatter.sigClicked.connect(self.handle_click)
        self._view_radio_group.idToggled.connect(self.handle_view_radio_toggled)

        self.show()

    def handle_view_radio_toggled(self, button_id, checked):
        if not checked:
            return

        mode_map = {0: "max_waveform", 1: "heatmap", 2: "heatmap_all_channels"}
        self.cfgs["right_panel_view_mode"] = mode_map[button_id]

        if self.selected_spot is not None:
            self.set_y_limit()
            self.update_panel(int(self.selected_spot.data()))

    def handle_y_spinbox_min(self, value):
        self.panel_plot.setYRange(
            value,
            self.cfgs["left_panel_y_axis"]["y_max"],
            padding=0
        )

    def handle_y_spinbox_max(self, value):
        self.panel_plot.setYRange(
            self.cfgs["left_panel_y_axis"]["y_min"],
            value,
            padding=0
        )

    def handle_fix_ylim_cb(self, active):
        self.ymin_spin.setEnabled(active)
        self.ymax_spin.setEnabled(active)
        self.cfgs["left_panel_y_axis"]["on"] = active
        self.set_y_limit()
        if self.selected_spot is None:
            return
        self.update_panel(int(self.selected_spot.data()))

    def set_y_limit(self):
        if self.cfgs["right_panel_view_mode"] == "max_waveform":
            if self.cfgs["left_panel_y_axis"]["on"]:
                self.panel_plot.setYRange(self.ymin_spin.value(), self.ymax_spin.value(), padding=0)
            else:
                self.panel_plot.enableAutoRange(axis='y')
        else:
            self.panel_plot.enableAutoRange()

    def handle_click(self, _, points, __):
        if points is None or len(points) <= 0:
            return

        spot = points[0]

        if self.selected_spot is not None:
            self.selected_spot.setPen(pg.mkPen(None))

        spot.setPen(pg.mkPen('r', width=2))
        self.selected_spot = spot

        idx = int(spot.data())
        self.update_panel(idx)

    def update_panel(self, spike_idx):
        template_id = int(self.spike_templates[spike_idx])
        self.panel_plot.setTitle(f"Template {template_id}")

        if self.cfgs["right_panel_view_mode"] == "max_waveform":
            self._draw_max_waveform_on_panel(spike_idx)
        else:
            self._draw_template_heatmap_on_panel(spike_idx)

    def _draw_max_waveform_on_panel(self, spike_index):
        template_waveform = self.get_max_waveform_data(spike_index)
        n_samples = template_waveform.size

        pen = pg.mkPen("k", width=2.5)
        self.panel_plot.clear()
        self.panel_plot.plot(np.arange(n_samples), template_waveform, pen=pen)

        self.panel_plot.setLabel("bottom", "sample")
        self.panel_plot.setLabel("left", "amplitude")
        self.panel_plot.setXRange(0, n_samples, padding=0.05)

    def _draw_template_heatmap_on_panel(self, spike_index):
        template_waveform_2d = self.get_heatmap_data(spike_index)
        n_samples, n_chans = template_waveform_2d.shape[0], template_waveform_2d.shape[1]

        self.panel_plot.clear()

        image_item = pg.ImageItem()
        self.panel_plot.addItem(image_item)

        colors = [
            (0, 0, 180, 255),
            (255, 255, 255, 255),
            (180, 0, 0, 255),
        ]
        cmap = pg.ColorMap(pos=[0.0, 0.5, 1.0], color=colors)
        image_item.setColorMap(cmap)

        if self.cfgs["left_panel_y_axis"]["on"]:
            image_item.setLevels((
                self.cfgs["left_panel_y_axis"]["y_min"],
                self.cfgs["left_panel_y_axis"]["y_max"]
            ))
        image_item.setImage(template_waveform_2d)
        image_item.setRect(0, 0, n_samples, n_chans)

    def get_max_waveform_data(self, spike_index):
        template_idx = self.spike_templates[spike_index]
        scaled_template = self.templates[template_idx, :, :] * self.spike_amplitudes[spike_index]
        peak_ch = np.argmax(np.max(np.abs(scaled_template), axis=0))
        return scaled_template[:, peak_ch]

    def get_heatmap_data(self, spike_index):
        """"""
        template_idx = self.spike_templates[spike_index]
        scaled_template = self.templates[template_idx, :, :] * self.spike_amplitudes[spike_index]

        if self.cfgs["right_panel_view_mode"] == "heatmap_all_channels":
            scaled_template = scaled_template.copy()  # TODO: check if this is necessary
            scaled_template[:, scaled_template[0, :] == 0] = np.nan
        else:
            contains_data_idx = np.where(scaled_template[0, :] != 0)[0]
            scaled_template = scaled_template[:, contains_data_idx]

        return scaled_template
