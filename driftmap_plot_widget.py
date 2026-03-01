
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import matplotlib.pyplot as plt

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOption("antialias", True)




class DriftmapPlotWidget(QtWidgets.QWidget):
    def __init__(self, spike_times, spike_amplitudes, spike_depths,
                 amplitude_range_all_spikes, spike_templates, templates):
        super().__init__()

        self.spike_times = spike_times
        self.spike_amplitudes = spike_amplitudes
        self.spike_depths = spike_depths
        self.amplitude_range_all_spikes = amplitude_range_all_spikes
        self.spike_templates = spike_templates
        self.templates = templates

        self.resize(1400, 820)

        # Core layout
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer_layout.addWidget(splitter)

        win_left = pg.GraphicsLayoutWidget()
        splitter.addWidget(win_left)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        win_right = pg.GraphicsLayoutWidget()
        right_layout.addWidget(win_right, stretch=1)
        splitter.addWidget(right_widget)

        # Scatter Plot
        # --------------------------------------------------------------------------------------------------------------

        p_scatter = win_left.addPlot(row=0, col=0)
        p_scatter.setLabel("bottom", "Time (s)")
        p_scatter.setLabel("left", "Depth (µm)")
        p_scatter.showGrid(x=False, y=False)

        # set amplitude colors
        n_color_bins = 20
        amp_min, amp_max = spike_amplitudes.min(), spike_amplitudes.max()
        color_bins = np.linspace(amp_min, amp_max, n_color_bins)
        gray_colors = plt.get_cmap("gray")(np.linspace(0, 1, n_color_bins))[::-1]
        bin_indices = np.clip(np.searchsorted(color_bins, spike_amplitudes, side="right") - 1, 0, n_color_bins - 2)
        rgba_float = gray_colors[bin_indices]

        # set axis limits
        x_pad = (spike_times.max() - spike_times.min()) * 0.025
        y_pad = (spike_depths.max() - spike_depths.min()) * 0.05
        p_scatter.getViewBox().setLimits(
            xMin=spike_times.min() - x_pad,
            xMax=spike_times.max() + x_pad,
            yMin=spike_depths.min() - y_pad,
            yMax=spike_depths.max() + y_pad,
        )

        # create plot
        point_size = 6.0
        self.scatter = pg.ScatterPlotItem(
            spike_times, spike_depths,
            pxMode=True, size=point_size, hoverable=True, antialias=True, data=np.arange(len(spike_times)), brush=rgba_float*255, pen=None
        )


        # Template Plot
        # --------------------------------------------------------------------------------------------------------------

        self.panel_plot = win_right.addPlot(row=0, col=0)
        self.panel_plot.setLabel("bottom", "sample")
        self.panel_plot.setLabel("left", "amplitude")
        self.panel_plot.showGrid(x=False, y=False)

        # Tidying Up
        # --------------------------------------------------------------------------------------------------------------

        # Add widgets
        p_scatter.addItem(self.scatter)

        # Connect widgets
        self.scatter.sigHovered.connect(self.handle_hover)


        self.show()

    def handle_hover(self, _, points, __):
        if points is None or len(points) != 1:
            return

        spot = points[0]
        idx = int(spot.data())
        self.update_panel(idx)

    def update_panel(self, spike_idx):
        template_id = int(self.spike_templates[spike_idx])  # y convert to int dumb
        self.panel_plot.setTitle(f"Template {template_id}")
        self._draw_template_on_panel(spike_idx)

    def _draw_template_on_panel(self, spike_index):
        self.panel_plot.getViewBox().disableAutoRange()

        template_waveform = self.get_waveform_data(spike_index, "max_channel")
        n_samples = template_waveform.size

        pen = pg.mkPen("k", width=2.5)
        self.panel_plot.clear()
        self.panel_plot.plot(np.arange(n_samples), template_waveform, pen=pen)

        self.panel_plot.setLabel("bottom", "sample")
        self.panel_plot.setLabel("left", "amplitude")

        self.panel_plot.setXRange(0, n_samples, padding=0.05)

        y_max = float(np.max(np.abs(template_waveform)))
        self.panel_plot.setYRange(-y_max * 1.2, y_max * 1.2, padding=0)

    def get_waveform_data(self, spike_index, view_mode):

        template_idx = self.spike_templates[spike_index]
        scaled_template = self.templates[template_idx, :, :] * self.spike_amplitudes[spike_index]

        peak_ch = np.argmax(np.max(np.abs(scaled_template), axis=0))

        template_waveform = scaled_template[:, peak_ch]

        if view_mode == "max_channel":
            return template_waveform
      #  elif view_mode == "colormap":
       #     return [("image", out)]


def get_drift_map_plot_interactive(
        self,
        only_include_large_amplitude_spikes=True,
        decimate=False,
        exclude_noise=True,
        log_transform_amplitudes=True,
        large_amplitude_only_segment_size=800,
        point_size=6.0,
):

    spike_times, spike_amplitudes, spike_depths, spike_idx, _ = self._process_data(
        exclude_noise,
        log_transform_amplitudes,
        decimate,
        only_include_large_amplitude_spikes,
        large_amplitude_only_segment_size
    )

    # --- amplitude -> RGBA colors ---
    n_color_bins = 20
    amp_min, amp_max = spike_amplitudes.min(), spike_amplitudes.max()
    color_bins = np.linspace(amp_min, amp_max, n_color_bins)
    gray_colors = plt.get_cmap("gray")(np.linspace(0, 1, n_color_bins))[::-1]
    bin_indices = np.clip(np.searchsorted(color_bins, spike_amplitudes, side="right") - 1, 0, n_color_bins - 2)
    rgba_float = gray_colors[bin_indices]
    rgba_bytes = (rgba_float * 255).astype(np.uint8)

    templates = self.templates
    spike_templates_all = self.spike_templates

    N_CH_DISPLAY = 30
    N_SAMPLES = templates.shape[1]  # fixed number of samples per template

    def main_channel(tmpl_tc):
        return int(np.argmax(np.max(np.abs(tmpl_tc), axis=0)))

    def get_waveform_data(spike_index, view_mode):
        tid = int(spike_templates_all[spike_index])
        tmpl = templates[tid]  # (n_samples, n_channels)
        amp = float(self.spike_amplitudes[spike_index])
        tmpl_scaled = tmpl * amp

        peak_ch = main_channel(tmpl)
        n_ch_total = tmpl_scaled.shape[1]
        n_samples = tmpl_scaled.shape[0]
        half = N_CH_DISPLAY // 2

        # always exactly N_CH_DISPLAY channels, zero-padded, peak always at index half
        out = np.zeros((n_samples, N_CH_DISPLAY), dtype=np.float32)
        for i in range(N_CH_DISPLAY):
            src_ch = peak_ch - half + i
            if 0 <= src_ch < n_ch_total:
                out[:, i] = tmpl_scaled[:, src_ch]

        if view_mode == "max_channel":
            wf = out[:, half]
            return [("line", np.arange(n_samples), wf, "k")]
        elif view_mode == "colormap":
            return [("image", out)]

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    state = {
        "mode": "panel",
        "wave_view": "max_channel",
        "last_spike_idx": None,
        "last_spot": None,
        "last_spot_arr_idx": None,
        "popup": None,
        "panel_splitter_sizes": [900, 500],
    }

    # ------------------------------------------------------------------ layout
    container = QtWidgets.QWidget()
    container.resize(1400, 820)
    container.setWindowTitle("Drift map")
    outer_layout = QtWidgets.QVBoxLayout(container)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    outer_layout.setSpacing(0)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    splitter.setHandleWidth(6)

    win_left = pg.GraphicsLayoutWidget()

    right_widget = QtWidgets.QWidget()
    right_layout = QtWidgets.QVBoxLayout(right_widget)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(0)

    win_right = pg.GraphicsLayoutWidget()
    right_layout.addWidget(win_right, stretch=1)

    # controls bar
    controls_widget = QtWidgets.QWidget()
    controls_widget.setFixedHeight(36)
    controls_layout = QtWidgets.QHBoxLayout(controls_widget)
    controls_layout.setContentsMargins(6, 4, 6, 4)
    controls_layout.setSpacing(8)

    fix_ylim_cb = QtWidgets.QCheckBox("Fix y-limits")
    ymin_spin = QtWidgets.QDoubleSpinBox()
    ymax_spin = QtWidgets.QDoubleSpinBox()
    for spin in (ymin_spin, ymax_spin):
        spin.setRange(-1e9, 1e9)
        spin.setDecimals(1)
        spin.setFixedWidth(90)
        spin.setEnabled(False)
    ymin_spin.setValue(-100.0)
    ymax_spin.setValue(100.0)

    controls_layout.addWidget(fix_ylim_cb)
    controls_layout.addWidget(QtWidgets.QLabel("Y min:"))
    controls_layout.addWidget(ymin_spin)
    controls_layout.addWidget(QtWidgets.QLabel("Y max:"))
    controls_layout.addWidget(ymax_spin)
    controls_layout.addStretch()
    right_layout.addWidget(controls_widget)

    splitter.addWidget(win_left)
    splitter.addWidget(right_widget)
    splitter.setSizes(state["panel_splitter_sizes"])
    outer_layout.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------ y-limit helpers
    def apply_ylimits():
        if state["wave_view"] == "colormap":
            return
        if fix_ylim_cb.isChecked():
            p_wave.setYRange(ymin_spin.value(), ymax_spin.value(), padding=0)
        else:
            p_wave.enableAutoRange(axis='y')

    fix_ylim_cb.toggled.connect(lambda checked: (
        ymin_spin.setEnabled(checked),
        ymax_spin.setEnabled(checked),
        apply_ylimits()
    ))
    ymin_spin.valueChanged.connect(lambda _: fix_ylim_cb.isChecked() and apply_ylimits())
    ymax_spin.valueChanged.connect(lambda _: fix_ylim_cb.isChecked() and apply_ylimits())

    # ------------------------------------------------------------------ popup
    def ensure_popup():
        if state["popup"] is None:
            pop = pg.GraphicsLayoutWidget(title="Template (popup)")
            pop.setParent(container, QtCore.Qt.WindowType.Window)
            pop.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
            pop.setWindowFlag(QtCore.Qt.WindowType.Tool, True)
            pop.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
            pop.resize(520, 320)
            p = pop.addPlot()
            p.setLabel("bottom", "sample")
            p.setLabel("left", "amplitude")
            p.showGrid(x=False, y=False)
            pop.show()
            pop.raise_()
            pop.activateWindow()
            state["popup"] = (pop, p, [])
        else:
            pop, _, _ = state["popup"]
            pop.show()
            pop.raise_()
        return state["popup"]

    def close_popup():
        if state["popup"] is not None:
            pop, _, _ = state["popup"]
            pop.hide()

    # ------------------------------------------------------------------ waveform drawing
    panel_curves = []

    def _draw_waveform_on_plot(plot, curve_list, spike_index):
        for c in curve_list:
            plot.removeItem(c)
        curve_list.clear()

        # disable autorange completely before touching the plot
        plot.getViewBox().disableAutoRange()

        segments = get_waveform_data(spike_index, state["wave_view"])

        if segments and segments[0][0] == "image":
            _, img_data = segments[0]  # always (N_SAMPLES, N_CH_DISPLAY)

            image_item = pg.ImageItem()
            plot.addItem(image_item)

            colors = [
                (0, 0, 180, 255),
                (255, 255, 255, 255),
                (180, 0, 0, 255),
            ]
            cmap = pg.ColorMap(pos=[0.0, 0.5, 1.0], color=colors)
            image_item.setColorMap(cmap)

            vmax = float(np.max(np.abs(img_data))) + 1e-9
            image_item.setLevels((-vmax, vmax))
            image_item.setImage(img_data)

            plot.setLabel("bottom", "sample")
            plot.setLabel("left", "channel (relative to peak)")
            curve_list.append(image_item)

            # hard-lock both axes — always the same fixed values
            plot.setXRange(0, N_SAMPLES, padding=0)
            plot.setYRange(0, N_CH_DISPLAY, padding=0)

        else:
            for _, x, y, color in segments:
                pen = pg.mkPen(color, width=2.5)
                c = plot.plot(x, y, pen=pen)
                curve_list.append(c)
            plot.setLabel("bottom", "sample")
            plot.setLabel("left", "amplitude")
            plot.setXRange(0, N_SAMPLES, padding=0)
            apply_ylimits()

        return curve_list

    def update_panel(sidx):
        nonlocal panel_curves
        tid = int(spike_templates_all[sidx])
        p_wave.setTitle(f"Template {tid}")
        panel_curves = _draw_waveform_on_plot(p_wave, panel_curves, sidx)

    def update_popup(sidx):
        tid = int(spike_templates_all[sidx])
        pop, pop_plot, pop_curves = ensure_popup()
        pop_plot.setTitle(f"Template {tid}")
        _draw_waveform_on_plot(pop_plot, pop_curves, sidx)

    def redraw_current():
        sidx = state["last_spike_idx"]
        if sidx is None:
            return
        if state["mode"] == "panel":
            update_panel(sidx)
        else:
            update_popup(sidx)

    # ------------------------------------------------------------------ spot highlight
    def _restore_last_spot():
        if state["last_spot"] is not None and state["last_spot_arr_idx"] is not None:
            orig = rgba_bytes[state["last_spot_arr_idx"]]
            state["last_spot"].setBrush(pg.mkBrush(int(orig[0]), int(orig[1]), int(orig[2]), int(orig[3])))
            state["last_spot"].setSize(point_size)
        state["last_spot"] = None
        state["last_spot_arr_idx"] = None

    def _highlight_spot(spot, arr_idx):
        _restore_last_spot()
        spot.setBrush(pg.mkBrush(255, 0, 0, 220))
        spot.setSize(point_size * 2.2)
        state["last_spot"] = spot
        state["last_spot_arr_idx"] = arr_idx

    # ------------------------------------------------------------------ viewbox menu
    class MenuViewBox(pg.ViewBox):
        def mouseClickEvent(self, ev):
            if ev.button() == QtCore.Qt.MouseButton.RightButton:
                menu = QtWidgets.QMenu()

                act_panel = menu.addAction("Panel mode (right plot)")
                act_popup = menu.addAction("Popup mode (hover preview)")
                menu.addSeparator()

                wave_menu = menu.addMenu("Waveform view")
                act_max_ch = wave_menu.addAction("Max channel")
                act_cmap = wave_menu.addAction("Heatmap (all channels)")
                for a, key in [(act_max_ch, "max_channel"), (act_cmap, "colormap")]:
                    a.setCheckable(True)
                    a.setChecked(state["wave_view"] == key)
                menu.addSeparator()

                act_toggle_drift = menu.addAction("Toggle drift plot")

                act_panel.setCheckable(True)
                act_popup.setCheckable(True)
                act_panel.setChecked(state["mode"] == "panel")
                act_popup.setChecked(state["mode"] == "popup")

                gp = ev.screenPos()
                if hasattr(gp, "toPoint"):
                    gp = gp.toPoint()
                chosen = menu.exec_(gp)

                if chosen == act_panel:
                    if state["mode"] != "panel":
                        state["mode"] = "panel"
                        close_popup()
                        splitter.setSizes(state["panel_splitter_sizes"])

                elif chosen == act_popup:
                    if state["mode"] != "popup":
                        state["panel_splitter_sizes"] = splitter.sizes()
                        state["mode"] = "popup"
                        total = sum(splitter.sizes())
                        splitter.setSizes([total, 0])
                        ensure_popup()

                elif chosen in (act_max_ch, act_cmap):
                    state["wave_view"] = "max_channel" if chosen == act_max_ch else "colormap"
                    redraw_current()

                elif chosen == act_toggle_drift:
                    p_scatter.setVisible(not p_scatter.isVisible())

                ev.accept()
                return
            super().mouseClickEvent(ev)

    # ------------------------------------------------------------------ plots
    vb = MenuViewBox(enableMenu=False)
    p_scatter = win_left.addPlot(row=0, col=0, viewBox=vb)
    p_scatter.setLabel("bottom", "Time (s)")
    p_scatter.setLabel("left", "Depth (µm)")
    p_scatter.showGrid(x=False, y=False)

    # set limits so user can't zoom/pan outside the data extent
    x_pad = (spike_times.max() - spike_times.min()) * 0.05
    y_pad = (spike_depths.max() - spike_depths.min()) * 0.05
    vb.setLimits(
        xMin=spike_times.min() - x_pad,
        xMax=spike_times.max() + x_pad,
        yMin=spike_depths.min() - y_pad,
        yMax=spike_depths.max() + y_pad,
    )

    p_wave = win_right.addPlot(row=0, col=0)
    p_wave.setLabel("bottom", "sample")
    p_wave.setLabel("left", "amplitude")
    p_wave.showGrid(x=False, y=False)

    # ------------------------------------------------------------------ scatter
    scatter = pg.ScatterPlotItem(pxMode=True, size=point_size, hoverable=True, antialias=True)
    p_scatter.addItem(scatter)

    spots = []
    for i, (t, d, sidx, color) in enumerate(zip(spike_times, spike_depths, spike_idx, rgba_bytes)):
        r, g, b, a = int(color[0]), int(color[1]), int(color[2]), int(color[3])
        spots.append({
            "pos": (float(t), float(d)),
            "data": int(sidx),
            "brush": pg.mkBrush(r, g, b, a),
            "pen": pg.mkPen(None),
        })
    scatter.addPoints(spots)

    sidx_to_arr_idx = {int(s): i for i, s in enumerate(spike_idx)}

    # ------------------------------------------------------------------ interaction
    def _handle_spot(spot):
        sidx = int(spot.data())
        arr_idx = sidx_to_arr_idx.get(sidx)
        if arr_idx is not None:
            _highlight_spot(spot, arr_idx)
        return sidx

    def _pts_empty(pts):
        if pts is None:
            return True
        if isinstance(pts, (list, tuple)):
            return len(pts) == 0
        if hasattr(pts, "size"):
            return pts.size == 0
        try:
            return len(pts) == 0
        except Exception:
            return True

    def on_click(plot, points, ev):
        if _pts_empty(points) or state["mode"] != "panel":
            return
        sidx = _handle_spot(points[0])
        if sidx == state["last_spike_idx"]:
            return
        state["last_spike_idx"] = sidx
        update_panel(sidx)

    scatter.sigClicked.connect(on_click)

    def on_hover(plot, points, ev):
        if state["mode"] != "popup":
            return
        if _pts_empty(points):
            return
        sidx = _handle_spot(points[0])
        if sidx == state["last_spike_idx"]:
            return
        state["last_spike_idx"] = sidx
        update_popup(sidx)

    scatter.sigHovered.connect(on_hover)

    container.show()
    app.exec()