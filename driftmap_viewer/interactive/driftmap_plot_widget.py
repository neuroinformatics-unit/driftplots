import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import matplotlib.pyplot as plt
import warnings

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOption("antialias", True)


class DriftmapPlotWidget(QtWidgets.QWidget):
    """
    """
    def __init__(self, spike_times, spike_amplitudes, spike_depths,
                 spike_templates, templates, channel_positions,
                 amplitude_scaling="linear", n_color_bins=20,
                 point_size=5.0):
        super().__init__()

        self.spike_times = spike_times
        self.spike_amplitudes = spike_amplitudes
        self.spike_depths = spike_depths
        self.spike_templates = spike_templates
        self.templates = templates
        self.channel_positions = channel_positions

        self.cfgs = {
            "right_panel_view_mode": "heatmap",
            "left_panel_y_axis": {
                "on": False,
                "y_max": 200,
                "y_min": -200,
            },
        }

        self.selected_spot = None

        self.resize(1400, 820)

        # Instantiate UI and scatter plot
        win_left, win_right = self._init_ui()
        self._init_scatter_plot(win_left, spike_times, spike_amplitudes,
                                spike_depths, amplitude_scaling,
                                n_color_bins, point_size)
        self._init_panel_plot(win_right)

        # Connect widgets
        self.ymin_spin.valueChanged.connect(self.handle_y_spinbox_min)
        self.ymax_spin.valueChanged.connect(self.handle_y_spinbox_max)
        self._fix_limits_cb.toggled.connect(self.handle_fix_ylim_cb)
        self.scatter.sigClicked.connect(self.handle_click)
        self._view_radio_group.idToggled.connect(self.handle_view_radio_toggled)

    def _init_scatter_plot(self, win_left, spike_times, spike_amplitudes,
                           spike_depths, amplitude_scaling, n_color_bins,
                           point_size):
        """Create the scatter plot on the left panel.

        Parameters
        ----------
        win_left : pg.GraphicsLayoutWidget
            The left graphics area to host the scatter plot.
        spike_times, spike_amplitudes, spike_depths : np.ndarray
            Spike data arrays.
        amplitude_scaling : str | tuple
            Colour-scaling mode or explicit (min, max) range.
        n_color_bins : int
            Number of grey-scale colour bins.
        point_size : float
            Scatter-point diameter in pixels.
        """
        self.p_scatter = win_left.addPlot(row=0, col=0)
        self.p_scatter.setLabel("bottom", "Time (s)")
        self.p_scatter.setLabel("left", "Depth (µm)")
        self.p_scatter.showGrid(x=False, y=False)

        # set amplitude colors
        rgba_colors = self._compute_amplitude_colors(
            spike_amplitudes, amplitude_scaling, n_color_bins
        )

        # set axis limits, pad around them slightly
        x_pad = (spike_times.max() - spike_times.min()) * 0.025
        y_pad = (spike_depths.max() - spike_depths.min()) * 0.05

        self.p_scatter.getViewBox().setLimits(
            xMin=spike_times.min() - x_pad,
            xMax=spike_times.max() + x_pad,
            yMin=spike_depths.min() - y_pad,
            yMax=spike_depths.max() + y_pad,
        )

        # create plot — each point stores its spike index in 'data' for click/tooltip lookup
        self.scatter = pg.ScatterPlotItem(
            spike_times,
            spike_depths,
            pxMode=True,
            size=point_size,
            hoverable=True,
            antialias=True,
            data=np.arange(spike_times.size),
            brush=rgba_colors,
            pen=None,
            tip=lambda x, y, data: (
                f"x={x:.3f}\ny={y:.1f}\n"
                f"amp={self.spike_amplitudes[int(data)]:.2f}"
            ),
        )
        self.p_scatter.addItem(self.scatter)

    def _init_panel_plot(self, win_right):
        """Create the template panel plot on the right side.

        Parameters
        ----------
        win_right : pg.GraphicsLayoutWidget
            The right graphics area to host the panel plot.
        """
        self.panel_plot = win_right.addPlot(row=0, col=0)
        self.panel_plot.setLabel("bottom", "sample")
        self.panel_plot.setLabel("left", "amplitude")
        self.panel_plot.showGrid(x=False, y=False)

    def _connect_signals(self):
        """Wire up Qt signal/slot connections."""
        self.ymin_spin.valueChanged.connect(self.handle_y_spinbox_min)
        self.ymax_spin.valueChanged.connect(self.handle_y_spinbox_max)
        self._fix_limits_cb.toggled.connect(self.handle_fix_ylim_cb)
        self.scatter.sigClicked.connect(self.handle_click)
        self._view_radio_group.idToggled.connect(self.handle_view_radio_toggled)

    def handle_view_radio_toggled(self, button_id, checked):
        if not checked:
            return

        mode_map = {0: "heatmap", 1: "heatmap_all_channels"}
        mode = mode_map[button_id]
        self.cfgs["right_panel_view_mode"] = mode

        self._fix_limits_cb.setText("Fix color limits")
        self._min_label.setText("C min:")
        self._max_label.setText("C max:")

        if self.selected_spot is not None:
            self.update_panel(int(self.selected_spot.data()))

    def handle_y_spinbox_min(self, value):
        self.cfgs["left_panel_y_axis"]["y_min"] = value
        if self.selected_spot is not None:
            self.update_panel(int(self.selected_spot.data()))

    def handle_y_spinbox_max(self, value):
        self.cfgs["left_panel_y_axis"]["y_max"] = value
        if self.selected_spot is not None:
            self.update_panel(int(self.selected_spot.data()))

    def handle_fix_ylim_cb(self, active):
        self.ymin_spin.setEnabled(active)
        self.ymax_spin.setEnabled(active)
        self.cfgs["left_panel_y_axis"]["on"] = active
        if self.selected_spot is None:
            return
        self.update_panel(int(self.selected_spot.data()))

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

        self._draw_template_heatmap_on_panel(spike_idx)

    # TODO: carefully check these!!!

    def _draw_template_heatmap_on_panel(self, spike_index):
        template_waveform_2d = self.get_heatmap_data(spike_index)

        n_samples, n_chans = template_waveform_2d.shape[0], template_waveform_2d.shape[1]

        self.panel_plot.clear()

        if self.cfgs["right_panel_view_mode"] == "heatmap_all_channels":
            self.panel_plot.setLabel("left", "channel")
            self.panel_plot.getAxis("left").setTicks(None)
            self.panel_plot.getAxis("left").setStyle(showValues=True)
        else:
            self.panel_plot.setLabel("left", "")
            self.panel_plot.getAxis("left").setTicks([])
            self.panel_plot.getAxis("left").setStyle(showValues=False)
        self.panel_plot.setLabel("bottom", "sample")

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
                self.ymin_spin.value(),
                self.ymax_spin.value(),
            ))
        image_item.setImage(template_waveform_2d)
        image_item.setRect(0, 0, n_samples, n_chans)
        self.panel_plot.setXRange(0, n_samples, padding=0.05)
        self.panel_plot.setYRange(0, n_chans, padding=0.0)

    def get_heatmap_data(self, spike_index):
        """"""
        template_idx = self.spike_templates[spike_index]
        scaled_template = self.templates[template_idx, :, :] #  * self.spike_amplitudes[spike_index]

        # find the shank with signal, in case multi-shank probe was sorted
        mid_idx = int(scaled_template.shape[0] / 2)
        chan_with_signal = np.where(scaled_template[mid_idx, :] != 0)  # arbitrary cutoff, should check more
        positions_with_signal = self.channel_positions[chan_with_signal]
        shank_contacts_with_signal = np.unique(positions_with_signal[:, 0])

        if len(shank_contacts_with_signal) != 2:
            warnings.warn("This spikes template has signal on more than one shank.")

        shank_select = np.zeros(self.channel_positions.shape[0], dtype=bool)
        for pos in shank_contacts_with_signal:
            shank_select = np.logical_or(shank_select, self.channel_positions[:, 0] == pos)

        sort_idx = np.argsort(self.channel_positions[shank_select, 1], axis=0)

        # TODO: could include shank it is on, will need to actually
        scaled_template = scaled_template[:, shank_select]
        scaled_template = scaled_template[:, sort_idx]

        if self.cfgs["right_panel_view_mode"] == "heatmap_all_channels":
            scaled_template = scaled_template.copy()  # TODO: check if this is necessary
            scaled_template[:, scaled_template[mid_idx, :] == 0] = np.nan
        else:
            contains_data_idx = np.where(scaled_template[mid_idx, :] != 0)[0]
            scaled_template = scaled_template[:, contains_data_idx]

        return scaled_template

    @staticmethod
    def _compute_amplitude_colors(spike_amplitudes, amplitude_scaling, n_color_bins):
        """Map spike amplitudes to RGBA colours via grey-scale binning.

        Parameters
        ----------
        spike_amplitudes : np.ndarray
            (num_spikes,) raw amplitude values.
        amplitude_scaling : {"linear", "log2", "log10"} | tuple
            Scaling mode.  A 2-tuple ``(min, max)`` fixes the colour
            range explicitly.
        n_color_bins : int
            Number of grey-scale bins.

        Returns
        -------
        np.ndarray
            (num_spikes, 4) uint8 RGBA values in [0, 255].
        """
        amp_values = spike_amplitudes.copy()

        if isinstance(amplitude_scaling, tuple):
            amp_min, amp_max = amplitude_scaling
        elif amplitude_scaling == "log2":
            amp_values = np.log2(amp_values)
            amp_min, amp_max = amp_values.min(), amp_values.max()
        elif amplitude_scaling == "log10":
            amp_values = np.log10(amp_values)
            amp_min, amp_max = amp_values.min(), amp_values.max()
        else:  # "linear"
            amp_min, amp_max = amp_values.min(), amp_values.max()

        color_bins = np.linspace(amp_min, amp_max, n_color_bins)
        gray_colors = plt.get_cmap("gray")(np.linspace(0, 1, n_color_bins))[::-1]
        bin_indices = np.clip(
            np.searchsorted(color_bins, amp_values, side="right") - 1,
            0, n_color_bins - 2,
        )
        return (gray_colors[bin_indices] * 255).astype(np.uint8)

    # UI - To possibly be moved to QtDesigner
    # ----------------------------------------------------------------------------------

    def _init_ui(self):
        """Build the widget layout: splitter, controls bar, radio buttons, spinboxes.

        Returns
        -------
        win_left : pg.GraphicsLayoutWidget
            Left graphics area (for the scatter plot).
        win_right : pg.GraphicsLayoutWidget
            Right graphics area (for the template panel).
        """
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

        self.radio_heatmap = QtWidgets.QRadioButton("Template heatmap")
        self.radio_heatmap_all = QtWidgets.QRadioButton("Template heatmap (all channels)")
        self.radio_heatmap.setChecked(True)

        self._view_radio_group = QtWidgets.QButtonGroup(self)
        self._view_radio_group.addButton(self.radio_heatmap, 0)
        self._view_radio_group.addButton(self.radio_heatmap_all, 1)

        radio_layout.addWidget(self.radio_heatmap)
        radio_layout.addWidget(self.radio_heatmap_all)
        radio_layout.addStretch()
        controls_layout.addWidget(radio_row)

        # Limits controls row
        self._limits_page = QtWidgets.QWidget()
        limits_layout = QtWidgets.QHBoxLayout(self._limits_page)
        limits_layout.setContentsMargins(0, 0, 0, 0)
        limits_layout.setSpacing(8)

        self._fix_limits_cb = QtWidgets.QCheckBox("Fix color limits")
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

        limits_layout.addWidget(self._fix_limits_cb)
        self._min_label = QtWidgets.QLabel("C min:")
        self._max_label = QtWidgets.QLabel("C max:")
        limits_layout.addWidget(self._min_label)
        limits_layout.addWidget(self.ymin_spin)
        limits_layout.addWidget(self._max_label)
        limits_layout.addWidget(self.ymax_spin)
        limits_layout.addStretch()

        controls_layout.addWidget(self._limits_page)

        # Add controls below the splitter, spanning full width
        outer_layout.addWidget(controls_widget)

        return win_left, win_right
