import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from PySide6 import QtWidgets, QtCore

from driftmap_plot_widget import DriftmapPlotWidget


class MultiSessionDriftmapWidget(QtWidgets.QWidget):
    """
    Owns the single shared matplotlib Figure and FigureCanvas.

    Layout per session (repeated horizontally):
        [ scatter axes (3x) | panel axes (1x) ]

    Qt controls for each session sit below their respective axes columns,
    aligned using a QSplitter so the proportions stay in sync visually.

    Saving captures the whole figure in one call.
    """

    def __init__(self, session_data: list[dict]):
        """
        Parameters
        ----------
        session_data : list of dict, each containing:
            spike_times, spike_amplitudes, spike_depths,
            amplitude_range_all_spikes, spike_templates, templates
        """
        super().__init__()
        self.setWindowTitle("Drift map — multi session")

        n = len(session_data)
        self.resize(700 * n, 900)

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Single shared figure
        # Each session gets 2 columns: scatter (weight 3) + panel (weight 1)
        # Between sessions add a small gap via wspace
        col_ratios = []
        for i in range(n):
            col_ratios += [3, 1]

        self.fig = plt.Figure(figsize=(7 * n, 7))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        outer_layout.addWidget(self.canvas, stretch=1)

        gs = GridSpec(
            1, n * 2,
            figure=self.fig,
            width_ratios=col_ratios,
            left=0.06, right=0.98,
            top=0.97, bottom=0.08,
            wspace=0.45,
        )

        # Controls bar — one QWidget per session, laid out in a matching splitter
        controls_bar = QtWidgets.QWidget()
        controls_bar_layout = QtWidgets.QHBoxLayout(controls_bar)
        controls_bar_layout.setContentsMargins(0, 0, 0, 0)
        controls_bar_layout.setSpacing(2)

        self._panels: list[DriftmapPlotWidget] = []

        for i, data in enumerate(session_data):
            ax_scatter = self.fig.add_subplot(gs[0, i * 2])
            ax_panel   = self.fig.add_subplot(gs[0, i * 2 + 1])

            panel = DriftmapPlotWidget(
                spike_times=data["spike_times"],
                spike_amplitudes=data["spike_amplitudes"],
                spike_depths=data["spike_depths"],
                amplitude_range_all_spikes=data["amplitude_range_all_spikes"],
                spike_templates=data["spike_templates"],
                templates=data["templates"],
                ax_scatter=ax_scatter,
                ax_panel=ax_panel,
            )
            self._panels.append(panel)
            controls_bar_layout.addWidget(panel, stretch=4)   # 3+1 columns worth

        # Link y-axes across all sessions
        ref_ax = self._panels[0].ax_scatter
        for panel in self._panels[1:]:
            panel.ax_scatter.sharey(ref_ax)

        # Save button sits at the far right of the controls bar
        save_btn = QtWidgets.QPushButton("Save figure…")
        save_btn.setFixedWidth(120)
        save_btn.clicked.connect(self.handle_save)
        controls_bar_layout.addWidget(save_btn, stretch=0)

        outer_layout.addWidget(controls_bar)

        # Route canvas click events to the correct session panel
        self.canvas.mpl_connect("button_press_event", self._dispatch_click)

        self.show()

    # ------------------------------------------------------------------
    # Click dispatch — figure out which session's scatter was clicked
    # ------------------------------------------------------------------

    def _dispatch_click(self, event):
        if event.button != 1 or event.inaxes is None:
            return
        for panel in self._panels:
            if event.inaxes is panel.ax_scatter:
                panel.handle_click(event)
                return

    # ------------------------------------------------------------------
    # Saving — the whole figure in one shot
    # ------------------------------------------------------------------

    def handle_save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save figure",
            "driftmap.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tiff)",
        )
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
