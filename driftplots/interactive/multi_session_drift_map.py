import math

from PySide6 import QtWidgets

from .driftmap_plot_widget import DriftmapPlotWidget


class MultiSessionDriftmapWidget(QtWidgets.QWidget):
    """A grid container that displays multiple :class:`DriftmapPlotWidget` panels.

    Panels are laid out on an auto-computed (or user-specified) grid and
    their scatter-plot y-axes are linked so scrolling / zooming in one
    panel keeps all panels in sync.

    Parameters
    ----------
    panels
        Drift-map widgets to arrange in the grid.
    grid
        Explicit ``(n_rows, n_cols)`` layout. If ``None``, a roughly
        square layout is computed automatically.
    width
        Width allocated per panel column (pixels).
    height
        Height allocated per panel row (pixels).
    """

    def __init__(
        self,
        panels: list[DriftmapPlotWidget],
        grid: tuple[int, int] | None = None,
        width: int = 700,
        height: int = 820,
    ):
        super().__init__()
        self.setWindowTitle("Drift map — multi session")

        num_panels = len(panels)
        n_rows, n_cols = self._compute_grid_dimensions(num_panels, grid)

        self.resize(width * n_cols, height * n_rows)
        self._populate_grid(panels, n_rows, n_cols)
        self._link_y_axes(panels)

        self.show()

    @staticmethod
    def _compute_grid_dimensions(
        num_panels: int,
        grid: tuple[int, int] | None,
    ) -> tuple[int, int]:
        """Return ``(n_rows, n_cols)`` for the panel layout.

        Parameters
        ----------
        num_panels
            Total number of panels to arrange.
        grid
            User-specified ``(n_rows, n_cols)``. If ``None``, a roughly
            square grid is computed automatically.

        Returns
        -------
        tuple of int
            (n_rows, n_cols).
        """
        if grid is not None:
            n_rows, n_cols = grid
            if n_rows * n_cols != num_panels:
                raise ValueError(
                    f"grid {grid} expects {n_rows * n_cols} panels but got {num_panels}"
                )
            return n_rows, n_cols

        n_cols = math.ceil(math.sqrt(num_panels))
        n_rows = math.ceil(num_panels / n_cols)

        return n_rows, n_cols

    def _populate_grid(
        self,
        panels: list[DriftmapPlotWidget],
        n_rows: int,
        n_cols: int,
    ) -> None:
        """Place each panel into a :class:`QGridLayout`.

        Parameters
        ----------
        panels
            Widgets to add.
        n_rows, n_cols
            Grid dimensions.
        """
        grid_layout = QtWidgets.QGridLayout(self)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(2)

        panel_idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if panel_idx >= len(panels):
                    break
                panels[panel_idx].setParent(self)
                grid_layout.addWidget(panels[panel_idx], row, col)
                panel_idx += 1

    @staticmethod
    def _link_y_axes(panels: list[DriftmapPlotWidget]) -> None:
        """Link the scatter-plot y-axes of all panels to the first panel.

        This ensures scrolling or zooming the depth axis in any panel
        updates all other panels to match.

        Parameters
        ----------
        panels
            Must contain at least one panel.
        """
        ref = panels[0].p_scatter
        for panel in panels[1:]:
            panel.p_scatter.setYLink(ref)
