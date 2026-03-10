import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PySide6 import QtWidgets


class DriftmapPlotWidget(QtWidgets.QWidget):
    """
    Controls + data for a single session driftmap.

    Does NOT own a Figure or Canvas — axes are injected by the parent
    (MultiSessionDriftmapWidget), which owns the single shared Figure.
    Call _redraw() to ask the parent canvas to refresh.
    """

    def __init__(self, spike_times, spike_amplitudes, spike_depths,
                 amplitude_range_all_spikes, spike_templates, templates,
                 ax_scatter, ax_panel):
        super().__init__()

        self.spike_times = spike_times
        self.spike_amplitudes = spike_amplitudes
        self.spike_depths = spike_depths
        self.amplitude_range_all_spikes = amplitude_range_all_spikes
        self.spike_templates = spike_templates
        self.templates = templates

        self.ax_scatter = ax_scatter
        self.ax_panel = ax_panel

        self.cfgs = {
            "right_panel_view_mode": "max_waveform",
            "left_panel_y_axis": {
                "on": False,
                "y_max": 200,
                "y_min": -200,
            },
        }

        self.selected_spike_idx = None
        self._scatter_offsets = None

        # Qt controls — no canvas here, just the controls bar
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(6, 6, 6, 6)
        controls_layout.setSpacing(6)

        # Radio buttons
        radio_row = QtWidgets.QWidget()
        radio_layout = QtWidgets.QHBoxLayout(radio_row)
        radio_layout.setContentsMargins(0, 0, 0, 0)
        radio_layout.setSpacing(12)

        self.radio_max_wf  = QtWidgets.QRadioButton("Max waveform")
        self.radio_heatmap = QtWidgets.QRadioButton("Heatmap")
        self.radio_max_wf.setChecked(True)

        self._view_radio_group = QtWidgets.QButtonGroup(self)
        self._view_radio_group.addButton(self.radio_heatmap, 0)
        self._view_radio_group.addButton(self.radio_max_wf,  1)

        radio_layout.addWidget(self.radio_max_wf)
        radio_layout.addWidget(self.radio_heatmap)
        radio_layout.addStretch()
        controls_layout.addWidget(radio_row)

        # Fix y-limits row
        ylim_row = QtWidgets.QWidget()
        ylim_layout = QtWidgets.QHBoxLayout(ylim_row)
        ylim_layout.setContentsMargins(0, 0, 0, 0)
        ylim_layout.setSpacing(8)

        self.fix_ylim_cb = QtWidgets.QCheckBox("Fix y-limits")
        self.ymin_spin = QtWidgets.QDoubleSpinBox()
        self.ymax_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.ymin_spin, self.ymax_spin):
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(1)
            spin.setFixedWidth(90)
            spin.setMinimumWidth(100)
            spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setEnabled(False)
        self.ymin_spin.setValue(self.cfgs["left_panel_y_axis"]["y_min"])
        self.ymax_spin.setValue(self.cfgs["left_panel_y_axis"]["y_max"])

        ylim_layout.addWidget(self.fix_ylim_cb)
        ylim_layout.addWidget(QtWidgets.QLabel("Y min:"))
        ylim_layout.addWidget(self.ymin_spin)
        ylim_layout.addWidget(QtWidgets.QLabel("Y max:"))
        ylim_layout.addWidget(self.ymax_spin)
        ylim_layout.addStretch()
        controls_layout.addWidget(ylim_row)

        outer_layout.addWidget(controls_widget)

        # Draw into the injected axes
        self._build_scatter()

        # Connections
        self.ymin_spin.valueChanged.connect(self.handle_y_spinbox_min)
        self.ymax_spin.valueChanged.connect(self.handle_y_spinbox_max)
        self.fix_ylim_cb.toggled.connect(self.handle_fix_ylim_cb)
        self._view_radio_group.idToggled.connect(self.handle_view_radio_toggled)

    # ------------------------------------------------------------------
    # Canvas helper
    # ------------------------------------------------------------------

    def _redraw(self):
        self.ax_scatter.get_figure().canvas.draw_idle()

    # ------------------------------------------------------------------
    # Scatter build
    # ------------------------------------------------------------------

    def _build_scatter(self):
        ax = self.ax_scatter
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Depth (µm)")
        ax.grid(False)

        n_color_bins = 20
        amp_min = self.spike_amplitudes.min()
        amp_max = self.spike_amplitudes.max()
        color_bins  = np.linspace(amp_min, amp_max, n_color_bins)
        gray_colors = plt.get_cmap("gray")(np.linspace(0, 1, n_color_bins))[::-1]
        bin_indices = np.clip(
            np.searchsorted(color_bins, self.spike_amplitudes, side="right") - 1,
            0, n_color_bins - 2,
        )
        rgba_float = gray_colors[bin_indices]

        self._scatter = ax.scatter(
            self.spike_times, self.spike_depths,
            c=rgba_float, s=4, linewidths=0,
            rasterized=True,
        )
        self._scatter_offsets = self._scatter.get_offsets()

        x_pad = (self.spike_times.max() - self.spike_times.min()) * 0.025
        y_pad = (self.spike_depths.max() - self.spike_depths.min()) * 0.05
        ax.set_xlim(self.spike_times.min() - x_pad, self.spike_times.max() + x_pad)
        ax.set_ylim(self.spike_depths.min() - y_pad, self.spike_depths.max() + y_pad)

        self._highlight, = ax.plot([], [], 'o', ms=6,
                                   mfc='none', mec='red', mew=1.5, zorder=5)

    # ------------------------------------------------------------------
    # Click handling — called by MultiSessionDriftmapWidget's canvas callback
    # ------------------------------------------------------------------

    def handle_click(self, event):
        """Called by the parent when a click lands in this widget's ax_scatter."""
        offsets    = self._scatter_offsets
        xy_disp    = self.ax_scatter.transData.transform(offsets)
        click_disp = np.array([[event.x, event.y]])
        dists      = np.linalg.norm(xy_disp - click_disp, axis=1)

        nearest = int(np.argmin(dists))
        if dists[nearest] > 12:
            return

        self.selected_spike_idx = nearest
        x, y = offsets[nearest]
        self._highlight.set_data([x], [y])
        self.update_panel(nearest)
        self._redraw()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def handle_view_radio_toggled(self, button_id, checked):
        if not checked:
            return
        self.cfgs["right_panel_view_mode"] = "heatmap" if button_id == 0 else "max_waveform"
        self.ax_panel.cla()
        self.set_y_limit()
        if self.selected_spike_idx is not None:
            self.update_panel(self.selected_spike_idx)
        self._redraw()

    def handle_y_spinbox_min(self, value):
        self.cfgs["left_panel_y_axis"]["y_min"] = value
        if self.cfgs["left_panel_y_axis"]["on"]:
            self.ax_panel.set_ylim(bottom=value)
            self._redraw()

    def handle_y_spinbox_max(self, value):
        self.cfgs["left_panel_y_axis"]["y_max"] = value
        if self.cfgs["left_panel_y_axis"]["on"]:
            self.ax_panel.set_ylim(top=value)
            self._redraw()

    def handle_fix_ylim_cb(self, active):
        self.ymin_spin.setEnabled(active)
        self.ymax_spin.setEnabled(active)
        self.cfgs["left_panel_y_axis"]["on"] = active
        self.set_y_limit()
        if self.selected_spike_idx is not None:
            self.update_panel(self.selected_spike_idx)
        self._redraw()

    def set_y_limit(self):
        if self.cfgs["right_panel_view_mode"] == "max_waveform":
            if self.cfgs["left_panel_y_axis"]["on"]:
                self.ax_panel.set_ylim(
                    self.cfgs["left_panel_y_axis"]["y_min"],
                    self.cfgs["left_panel_y_axis"]["y_max"],
                )
            else:
                self.ax_panel.autoscale(enable=True, axis='y')
        else:
            self.ax_panel.autoscale(enable=True)

    # ------------------------------------------------------------------
    # Panel update
    # ------------------------------------------------------------------

    def update_panel(self, spike_idx):
        template_id = int(self.spike_templates[spike_idx])
        self.ax_panel.cla()
        self.ax_panel.set_title(f"Template {template_id}", fontsize=9)

        if self.cfgs["right_panel_view_mode"] == "max_waveform":
            self._draw_max_waveform_on_panel(spike_idx)
        else:
            self._draw_template_heatmap_on_panel(spike_idx)

        self.set_y_limit()

    def _draw_max_waveform_on_panel(self, spike_index):
        wf = self.get_max_waveform_data(spike_index)
        xs = np.arange(wf.size)
        self.ax_panel.plot(xs, wf, color='k', linewidth=1.5)
        self.ax_panel.set_xlabel("sample")
        self.ax_panel.set_ylabel("amplitude")
        self.ax_panel.set_xlim(xs[0], xs[-1])

    def _draw_template_heatmap_on_panel(self, spike_index):
        data = self.get_heatmap_data(spike_index)
        colors = [(0, 0, 0.7), (1, 1, 1), (0.7, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("bwr_custom", colors)
        self.ax_panel.imshow(
            data.T, aspect="auto", origin="lower",
            cmap=cmap, interpolation="nearest",
        )
        self.ax_panel.set_xlabel("sample")
        self.ax_panel.set_ylabel("channel")

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def get_max_waveform_data(self, spike_index):
        template_idx = self.spike_templates[spike_index]
        scaled = self.templates[template_idx] * self.spike_amplitudes[spike_index]
        peak_ch = np.argmax(np.max(np.abs(scaled), axis=0))
        return scaled[:, peak_ch]

    def get_heatmap_data(self, spike_index):
        template_idx = self.spike_templates[spike_index]
        scaled = self.templates[template_idx] * self.spike_amplitudes[spike_index]
        contains_data_idx = np.where(scaled[0, :] != 0)[0]
        return scaled[:, contains_data_idx]
