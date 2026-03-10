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

        ref = panels[0].p_scatter
        for panel in panels[1:]:
            panel.p_scatter.setYLink(ref)

        self.show()