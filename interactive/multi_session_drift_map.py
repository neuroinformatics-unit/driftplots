import math
from PySide6 import QtWidgets
from driftmap_plot_widget import DriftmapPlotWidget

class MultiSessionDriftmapWidget(QtWidgets.QWidget):
    def __init__(self, panels: list[DriftmapPlotWidget], grid=None):
        super().__init__()
        self.setWindowTitle("Drift map — multi session")

        n = len(panels)

        if grid is not None:
            n_rows, n_cols = grid
            if n_rows * n_cols != n:
                raise ValueError(
                    f"grid {grid} expects {n_rows * n_cols} panels but got {n}"
                )
        else:
            n_cols = math.ceil(math.sqrt(n))
            n_rows = math.ceil(n / n_cols)

        grid_layout = QtWidgets.QGridLayout(self)
        self.resize(700 * n_cols, 820 * n_rows)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(2)

        idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if idx >= n:
                    break
                panels[idx].setParent(self)
                grid_layout.addWidget(panels[idx], row, col)
                idx += 1

        ref = panels[0].p_scatter
        for panel in panels[1:]:
            panel.p_scatter.setYLink(ref)

        self.show()