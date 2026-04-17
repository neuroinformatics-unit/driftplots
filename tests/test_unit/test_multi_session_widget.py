import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from driftplots import DriftPlotter
from driftplots.interactive.multi_session_drift_map import MultiSessionDriftmapWidget

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def app():
    instance = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield instance


@pytest.fixture()
def make_panels(synthetic_ks4_output):
    """Factory that creates N panels via the DriftPlotter API."""
    def _make(n):
        return [
            DriftPlotter(synthetic_ks4_output).drift_map_plot_interactive(
                decimate=False, exclude_noise=False,
            )
            for _ in range(n)
        ]
    return _make


class TestGridComputation:
    """_compute_grid_dimensions should produce a valid layout."""

    def test_explicit_grid(self):
        rows, cols = MultiSessionDriftmapWidget._compute_grid_dimensions(6, (2, 3))
        assert (rows, cols) == (2, 3)

    def test_explicit_grid_mismatch_raises(self):
        with pytest.raises(ValueError, match="expects 6 panels but got 4"):
            MultiSessionDriftmapWidget._compute_grid_dimensions(4, (2, 3))

    @pytest.mark.parametrize("num_panels, expected", [
        (1, (1, 1)),
        (2, (1, 2)),
        (3, (2, 2)),
        (4, (2, 2)),
        (5, (2, 3)),
        (6, (2, 3)),
        (9, (3, 3)),
    ])
    def test_auto_grid(self, num_panels, expected):
        rows, cols = MultiSessionDriftmapWidget._compute_grid_dimensions(num_panels, None)
        assert (rows, cols) == expected
        assert rows * cols >= num_panels


class TestMultiWidget:
    """MultiSessionDriftmapWidget with real panels."""

    def test_y_axes_linked(self, make_panels):
        panels = make_panels(3)
        multi_widget = MultiSessionDriftmapWidget(panels)

        reference_viewbox = panels[0].p_scatter.getViewBox()
        for panel in panels[1:]:
            linked = panel.p_scatter.getViewBox().linkedView(1)
            assert linked is reference_viewbox

        multi_widget.close()

    def test_panels_parented(self, make_panels):
        panels = make_panels(2)
        multi_widget = MultiSessionDriftmapWidget(panels)
        for panel in panels:
            assert panel.parent() is multi_widget
        multi_widget.close()

    def test_window_title(self, make_panels):
        panels = make_panels(2)
        multi_widget = MultiSessionDriftmapWidget(panels)
        assert "multi session" in multi_widget.windowTitle().lower()
        multi_widget.close()
