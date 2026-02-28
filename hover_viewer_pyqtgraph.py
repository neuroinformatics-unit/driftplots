"""Interactive drift map with hover-to-template waveform preview (PyQtGraph).

Run this as a script (see example_run_hover_viewer.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

import ks_template_loader


@dataclass
class ViewerConfig:
    ks_version: str = "kilosort4"
    decimate: int | None = 20            # keep every Nth spike for speed
    max_points: int | None = None        # optional hard cap after decimate
    point_size: float = 4.0
    hover_radius_px: float = 12.0        # hover "snap" radius


class DriftMapHoverViewer(QtWidgets.QMainWindow):
    def __init__(self, sorter_output: str | Path, cfg: ViewerConfig = ViewerConfig()):
        super().__init__()
        self.setWindowTitle("Drift map hover viewer")

        sorter_output = Path(sorter_output)

        # Load spikes + templates
        spike_times, spike_amps, spike_depths = ks_template_loader.load_spikes(sorter_output, cfg.ks_version)
        spike_templates, template_waveforms = ks_template_loader.load_templates(sorter_output, cfg.ks_version)

        # Basic validation
        n = min(len(spike_times), len(spike_templates), len(spike_depths), len(spike_amps))
        spike_times = spike_times[:n]
        spike_amps = spike_amps[:n]
        spike_depths = spike_depths[:n]
        spike_templates = spike_templates[:n]

        # Decimate for responsiveness
        if cfg.decimate and cfg.decimate > 1:
            idx = np.arange(0, n, cfg.decimate, dtype=np.int64)
            spike_times = spike_times[idx]
            spike_amps = spike_amps[idx]
            spike_depths = spike_depths[idx]
            spike_templates = spike_templates[idx]

        if cfg.max_points is not None and len(spike_times) > cfg.max_points:
            spike_times = spike_times[:cfg.max_points]
            spike_amps = spike_amps[:cfg.max_points]
            spike_depths = spike_depths[:cfg.max_points]
            spike_templates = spike_templates[:cfg.max_points]

        self._spike_times = spike_times
        self._spike_depths = spike_depths
        self._spike_templates = spike_templates
        self._template_waveforms = template_waveforms

        # Central widget with two plots
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QHBoxLayout(cw)

        # Left: scatter
        self.scatter_plot = pg.PlotWidget(title="Drift map (hover a point)")
        self.scatter_plot.setLabel("bottom", "time (s)")
        self.scatter_plot.setLabel("left", "depth (um)")
        self.scatter_plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.scatter_plot, stretch=3)

        # Right: waveform
        self.wave_plot = pg.PlotWidget(title="Template waveform")
        self.wave_plot.setLabel("bottom", "sample")
        self.wave_plot.setLabel("left", "a.u.")
        self.wave_plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.wave_plot, stretch=2)

        self.wave_curve = self.wave_plot.plot([])

        # Scatter item
        self.scatter_item = pg.ScatterPlotItem(pxMode=True, size=cfg.point_size, hoverable=True)
        self.scatter_plot.addItem(self.scatter_item)

        # Attach per-point metadata via "data"
        spots = [
            {"pos": (float(t), float(d)), "data": int(tid)}
            for t, d, tid in zip(self._spike_times, self._spike_depths, self._spike_templates)
        ]
        self.scatter_item.addPoints(spots)

        # Hover handling: use a signal proxy on scene mouse move, then find nearest point
        self._cfg = cfg
        self._last_tid = None

        self._proxy = pg.SignalProxy(
            self.scatter_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )


    def _on_mouse_moved(self, evt):
        pos = evt[0]  # QPointF in scene coords
        vb = self.scatter_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return

        mouse_point = vb.mapSceneToView(pos)
        mx, my = float(mouse_point.x()), float(mouse_point.y())

        # Find candidate spikes in a time/depth window (fast coarse filter)
        # Window size is based on the current view width/height.
        xr = vb.viewRange()[0]
        yr = vb.viewRange()[1]
        xw = (xr[1] - xr[0]) * 0.02
        yw = (yr[1] - yr[0]) * 0.02

        x0, x1 = mx - xw, mx + xw
        y0, y1 = my - yw, my + yw

        mask = (self._spike_times >= x0) & (self._spike_times <= x1) & (self._spike_depths >= y0) & (self._spike_depths <= y1)
        if not np.any(mask):
            return

        # Among candidates, pick the nearest in screen pixels
        cand_idx = np.where(mask)[0]
        if cand_idx.size == 0:
            return

        # Map candidates to scene pixels to compute distance in px
        pts = np.vstack([self._spike_times[cand_idx], self._spike_depths[cand_idx]]).T
        scene_pts = np.array([vb.mapViewToScene(pg.Point(p[0], p[1])) for p in pts], dtype=object)

        dx = np.array([float(p.x()) - float(pos.x()) for p in scene_pts])
        dy = np.array([float(p.y()) - float(pos.y()) for p in scene_pts])
        dist2 = dx * dx + dy * dy
        k = int(np.argmin(dist2))

        if dist2[k] > (self._cfg.hover_radius_px ** 2):
            return

        tid = int(self._spike_templates[cand_idx[k]])
        if tid == self._last_tid:
            return

        self._last_tid = tid
        wf = self._template_waveforms[tid]
        self.wave_curve.setData(np.arange(wf.size), wf)
        self.wave_plot.setTitle(f"Template waveform (id={tid})")


def run(sorter_output: str | Path, ks_version: str = "kilosort4", decimate: int | None = 20):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg = ViewerConfig(ks_version=ks_version, decimate=decimate)
    w = DriftMapHoverViewer(sorter_output, cfg)
    w.resize(1300, 750)
    w.show()
    app.exec()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drift map hover-to-template viewer (PyQtGraph)")
    parser.add_argument("sorter_output", type=str, help="Path to Kilosort output folder")
    parser.add_argument("--ks_version", type=str, default="kilosort4", help="kilosort4 or kilosort1_3")
    parser.add_argument("--decimate", type=int, default=20, help="Keep every Nth spike")
    args = parser.parse_args()

    run(args.sorter_output, ks_version=args.ks_version, decimate=args.decimate)
