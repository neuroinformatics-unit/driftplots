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
                 spike_templates, templates, channel_positions,
                 amplitude_scaling="linear", n_color_bins=20,
                 point_size=5.0, sorter_path=None):
        super().__init__()

        print(f"Loaded {spike_times.size} spikes from {sorter_path}")

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
        self.radio_trace_view = QtWidgets.QRadioButton("Trace view")
        self.radio_heatmap.setChecked(True)

        self._view_radio_group = QtWidgets.QButtonGroup(self)
        self._view_radio_group.addButton(self.radio_max_wf, 0)
        self._view_radio_group.addButton(self.radio_heatmap, 1)
        self._view_radio_group.addButton(self.radio_heatmap_all, 2)
        self._view_radio_group.addButton(self.radio_trace_view, 3)

        radio_layout.addWidget(self.radio_max_wf)
        radio_layout.addWidget(self.radio_heatmap)
        radio_layout.addWidget(self.radio_heatmap_all)
        radio_layout.addWidget(self.radio_trace_view)
        radio_layout.addStretch()
        controls_layout.addWidget(radio_row)

        # Limits controls row
        self._limits_page = QtWidgets.QWidget()
        limits_layout = QtWidgets.QHBoxLayout(self._limits_page)
        limits_layout.setContentsMargins(0, 0, 0, 0)
        limits_layout.setSpacing(8)

        self._fix_limits_cb = QtWidgets.QCheckBox("Fix y-limits")
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
        self._min_label = QtWidgets.QLabel("Y min:")
        self._max_label = QtWidgets.QLabel("Y max:")
        limits_layout.addWidget(self._min_label)
        limits_layout.addWidget(self.ymin_spin)
        limits_layout.addWidget(self._max_label)
        limits_layout.addWidget(self.ymax_spin)
        limits_layout.addStretch()

        controls_layout.addWidget(self._limits_page)

        # Add controls below the splitter, spanning full width
        outer_layout.addWidget(controls_widget)

        # Scatter Plot
        # --------------------------------------------------------------------------------------------------------------

        self.p_scatter = win_left.addPlot(row=0, col=0)
        self.p_scatter.setLabel("bottom", "Time (s)")
        self.p_scatter.setLabel("left", "Depth (µm)")
        self.p_scatter.showGrid(x=False, y=False)

        # set amplitude colors
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
        bin_indices = np.clip(np.searchsorted(color_bins, amp_values, side="right") - 1, 0, n_color_bins - 2)
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
        self.scatter = pg.ScatterPlotItem(
            spike_times, spike_depths,
            pxMode=True, size=point_size, hoverable=True, antialias=True, data=np.arange(spike_times.size), brush=rgba_float*255, pen=None,
            tip=lambda x, y, data: f"x={x:.3f}\ny={y:.1f}\namp={spike_amplitudes[int(data)]:.2f}",  # TODO: this is super weird
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
        self._fix_limits_cb.toggled.connect(self.handle_fix_ylim_cb)
        self.scatter.sigClicked.connect(self.handle_click)
        self._view_radio_group.idToggled.connect(self.handle_view_radio_toggled)

    def handle_view_radio_toggled(self, button_id, checked):
        if not checked:
            return

        mode_map = {0: "max_waveform", 1: "heatmap", 2: "heatmap_all_channels", 3: "trace_view"}
        mode = mode_map[button_id]
        self.cfgs["right_panel_view_mode"] = mode
        self._trace_view_initialized = False

        if mode == "trace_view":
            self._limits_page.setVisible(False)
        else:
            self._limits_page.setVisible(True)
            if mode == "max_waveform":
                self._fix_limits_cb.setText("Fix y-limits")
                self._min_label.setText("Y min:")
                self._max_label.setText("Y max:")
            else:
                self._fix_limits_cb.setText("Fix color limits")
                self._min_label.setText("C min:")
                self._max_label.setText("C max:")

        if self.selected_spot is not None:
            self.set_y_limit()
            self.update_panel(int(self.selected_spot.data()))

    def handle_y_spinbox_min(self, value):
        self.cfgs["left_panel_y_axis"]["y_min"] = value
        if self.selected_spot is not None:
            self.set_y_limit()
            self.update_panel(int(self.selected_spot.data()))

    def handle_y_spinbox_max(self, value):
        self.cfgs["left_panel_y_axis"]["y_max"] = value
        if self.selected_spot is not None:
            self.set_y_limit()
            self.update_panel(int(self.selected_spot.data()))

    def handle_fix_ylim_cb(self, active):
        self.ymin_spin.setEnabled(active)
        self.ymax_spin.setEnabled(active)
        self.cfgs["left_panel_y_axis"]["on"] = active
        self.set_y_limit()
        if self.selected_spot is None:
            return
        self.update_panel(int(self.selected_spot.data()))

    def set_y_limit(self):
        mode = self.cfgs["right_panel_view_mode"]
        if mode == "max_waveform":
            if self.cfgs["left_panel_y_axis"]["on"]:
                self.panel_plot.setYRange(self.ymin_spin.value(), self.ymax_spin.value(), padding=0)
            else:
                self.panel_plot.enableAutoRange(axis='y')
        elif mode in ("heatmap", "heatmap_all_channels"):
            pass  # color limits applied during draw
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
        print(spike_idx)
        print(template_id)

        if self.cfgs["right_panel_view_mode"] == "max_waveform":
            self._draw_max_waveform_on_panel(spike_idx)
        elif self.cfgs["right_panel_view_mode"] == "trace_view":
            self._draw_template_trace_view_on_panel(spike_idx)
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
        self.panel_plot.getAxis("left").setTicks(None)
        self.panel_plot.getAxis("left").setStyle(showValues=True)
        self.panel_plot.setXRange(0, n_samples, padding=0.05)

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
        self.panel_plot.setYRange(0, n_chans, padding=0.05)

    def _draw_template_trace_view_on_panel(self, spike_index):
        template_idx = self.spike_templates[spike_index]
        wv = self.templates[template_idx].copy() * self.spike_amplitudes[spike_index]
        n_samples, n_chan = wv.shape

        # Use only channels with non-zero data, unwrapping if KS wrapped them
        contains_data_idx, is_wrapped = self._get_nonzero_channel_indices(wv)
        wv = wv[:, contains_data_idx]
        xc = self.channel_positions[contains_data_idx, 0]
        yc = self.channel_positions[contains_data_idx, 1]

        if is_wrapped:
            xc, yc = self._make_positions_contiguous(xc, yc)

        # Scale amplitude so the largest waveform fills ~half the channel spacing
        unique_y = np.unique(yc)
        if len(unique_y) > 1:
            chan_spacing = np.min(np.diff(np.sort(unique_y)))
        else:
            chan_spacing = 1.0
        max_abs = np.max(np.abs(wv))
        amp = (chan_spacing * 0.45) / max_abs if max_abs > 0 else 1.0

        # Scale time axis proportional to x-spacing
        unique_x = np.unique(xc)
        if len(unique_x) > 1:
            x_spacing = np.min(np.diff(np.sort(unique_x)))
        else:
            x_spacing = 20.0

        self.panel_plot.clear()
        for ii, (xi, yi) in enumerate(zip(xc, yc)):
            t = np.arange(-n_samples // 2, n_samples // 2, 1, dtype=np.float32)
            t /= n_samples / (x_spacing * 0.9)
            self.panel_plot.plot(xi + t, yi + wv[:, ii] * amp, pen=pg.mkPen('k', width=0.5))

        self.panel_plot.setLabel("bottom", "x position")
        self.panel_plot.setLabel("left", "y position (\u00b5m)")
        self.panel_plot.getAxis("left").setTicks(None)

        # Center the view on the displayed channels, preserving current zoom level
        y_center = (yc.min() + yc.max()) / 2
        x_center = (xc.min() + xc.max()) / 2
        view_box = self.panel_plot.getViewBox()
        [[x_lo, x_hi], [y_lo, y_hi]] = view_box.viewRange()
        y_half = (y_hi - y_lo) / 2
        x_half = (x_hi - x_lo) / 2

        if not hasattr(self, '_trace_view_initialized') or not self._trace_view_initialized:
            # First time: set a sensible default range
            y_pad = (yc.max() - yc.min()) * 0.15 + chan_spacing
            x_pad = (xc.max() - xc.min()) * 0.15 + x_spacing
            self.panel_plot.setXRange(xc.min() - x_pad, xc.max() + x_pad, padding=0)
            self.panel_plot.setYRange(yc.min() - y_pad, yc.max() + y_pad, padding=0)
            self._trace_view_initialized = True
        else:
            # Subsequent clicks: keep zoom, just re-center
            self.panel_plot.setXRange(x_center - x_half, x_center + x_half, padding=0)
            self.panel_plot.setYRange(y_center - y_half, y_center + y_half, padding=0)
        self.panel_plot.getAxis("left").setStyle(showValues=True)

    def get_max_waveform_data(self, spike_index):
        template_idx = self.spike_templates[spike_index]
        scaled_template = self.templates[template_idx, :, :] * self.spike_amplitudes[spike_index]
        peak_ch = np.argmax(np.max(np.abs(scaled_template), axis=0))
        return scaled_template[:, peak_ch]

    @staticmethod
    def _get_nonzero_channel_indices(scaled_template):
        """Get indices of channels with data, unwrapping if KS wrapped them.

        Kilosort can wrap template channel assignments around the probe
        boundaries (e.g. channels [0, 1, 2, 380, 381, 382]). This detects
        the wrap and reorders so the high-index group comes first,
        giving a spatially contiguous result.

        Returns
        -------
        contains_data_idx : np.ndarray
            Channel indices with data, reordered if wrapping detected.
        is_wrapped : bool
            True if wrapping was detected and corrected.
        """
        contains_data_idx = np.where(scaled_template[0, :] != 0)[0]

        if len(contains_data_idx) < 2:
            return contains_data_idx, False

        # Check for a gap (non-contiguous indices)
        diffs = np.diff(contains_data_idx)
        gap_positions = np.where(diffs > 1)[0]

        if len(gap_positions) == 1:
            # Wrapped: split at gap, put the higher-index group first
            split = gap_positions[0] + 1
            contains_data_idx = np.concatenate([
                contains_data_idx[split:],
                contains_data_idx[:split],
            ])
            return contains_data_idx, True

        return contains_data_idx, False

    @staticmethod
    def _make_positions_contiguous(xc, yc):
        """Remap channel positions so wrapped channels sit contiguously.

        When KS wraps, channels from the top and bottom of the probe
        end up in the same template. We shift the lower-position group
        to sit just above the higher-position group (or vice versa)
        so they display as one contiguous block.
        """
        sorted_y = np.sort(np.unique(yc))
        if len(sorted_y) < 2:
            return xc.copy(), yc.copy()

        # Find the largest gap in y positions
        y_diffs = np.diff(sorted_y)
        largest_gap_idx = np.argmax(y_diffs)
        gap_size = y_diffs[largest_gap_idx]
        typical_spacing = np.min(y_diffs)

        # If the largest gap is much bigger than typical spacing, it's a wrap
        if gap_size > typical_spacing * 3:
            gap_threshold = sorted_y[largest_gap_idx] + gap_size / 2
            yc_new = yc.copy()
            # Move the upper group down to sit just above the lower group
            upper_mask = yc >= gap_threshold
            lower_max = yc[~upper_mask].max() if np.any(~upper_mask) else yc.min()
            upper_min = yc[upper_mask].min() if np.any(upper_mask) else yc.max()
            shift = upper_min - lower_max - typical_spacing
            yc_new[upper_mask] -= shift
            return xc.copy(), yc_new

        return xc.copy(), yc.copy()

    def get_heatmap_data(self, spike_index):
        """"""
        template_idx = self.spike_templates[spike_index]
        scaled_template = self.templates[template_idx, :, :] * self.spike_amplitudes[spike_index]

        if self.cfgs["right_panel_view_mode"] == "heatmap_all_channels":
            scaled_template = scaled_template.copy()  # TODO: check if this is necessary
            scaled_template[:, scaled_template[0, :] == 0] = np.nan
        else:
            contains_data_idx, _ = self._get_nonzero_channel_indices(scaled_template)
            scaled_template = scaled_template[:, contains_data_idx]

        return scaled_template
