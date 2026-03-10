import matplotlib
matplotlib.use("QtAgg")
from PySide6 import QtWidgets
from driftmap_plot_widget import DriftmapPlotWidget


class MultiSessionDriftmapWidget(QtWidgets.QWidget):
    def __init__(self, panels: list[DriftmapPlotWidget]):
        super().__init__()
        self.setWindowTitle("Drift map — multi session")

        layout = QtWidgets.QHBoxLayout(self)
        self.resize(700 * len(panels), 820)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        for panel in panels:
            panel.setParent(self)
            layout.addWidget(panel)

        # Link y-axes across sessions
        ref = panels[0]
        for panel in panels[1:]:
            panel.share_y_with(ref)

        self.show()

    def handle_save(self):
        """Save all session figures as a single combined PNG/PDF/SVG."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save all sessions",
            "driftmap_all_sessions.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return

        n = len(self.findChildren(DriftmapPlotWidget))
        panels = self.findChildren(DriftmapPlotWidget)

        fig = plt.figure(figsize=(7 * n, 8))
        outer_gs = gridspec.GridSpec(1, n, figure=fig, wspace=0.4)

        for col, panel in enumerate(panels):
            # Copy each panel's figure content into the combined figure
            # by re-drawing from the axes data directly
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer_gs[col], width_ratios=[3, 1], wspace=0.35
            )
            ax_s = fig.add_subplot(inner_gs[0])
            ax_p = fig.add_subplot(inner_gs[1])

            # Re-draw scatter
            for coll in panel.ax_scatter.collections:
                ax_s.add_collection(
                    type(coll)(
                        coll.get_paths(),
                        offsets=coll.get_offsets(),
                        transOffset=ax_s.transData,
                        facecolors=coll.get_facecolors(),
                        sizes=coll.get_sizes(),
                        linewidths=coll.get_linewidths(),
                    )
                )
            ax_s.set_xlim(panel.ax_scatter.get_xlim())
            ax_s.set_ylim(panel.ax_scatter.get_ylim())
            ax_s.set_xlabel(panel.ax_scatter.get_xlabel())
            ax_s.set_ylabel(panel.ax_scatter.get_ylabel())

            # Re-draw panel
            for line in panel.ax_panel.lines:
                ax_p.plot(line.get_xdata(), line.get_ydata(),
                          color=line.get_color(), linewidth=line.get_linewidth())
            for im in panel.ax_panel.images:
                ax_p.imshow(im.get_array(), aspect="auto", origin="lower",
                            cmap=im.get_cmap(), extent=im.get_extent())
            ax_p.set_xlim(panel.ax_panel.get_xlim())
            ax_p.set_ylim(panel.ax_panel.get_ylim())
            ax_p.set_xlabel(panel.ax_panel.get_xlabel())
            ax_p.set_ylabel(panel.ax_panel.get_ylabel())
            ax_p.set_title(panel.ax_panel.get_title())

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
