from pathlib import Path

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from driftplots.data_loader import DataLoader
from driftplots.interactive.driftmap_plot_widget import DriftmapPlotWidget
from driftplots.interactive.multi_session_drift_map import MultiSessionDriftmapWidget

SORTER_OUTPUT = Path(__file__).parent.parent.parent / "examples" / "example_data" / "sorting" / "sorter_output"

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def app():
    instance = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield instance


@pytest.fixture()
def data_model():
    loader = DataLoader(SORTER_OUTPUT)
    return loader.get_processed_data(
        exclude_noise=False, decimate=False,
        filter_amplitude_mode=None, filter_amplitude_values=(),
    )


@pytest.fixture()
def make_panels(app, data_model):
    """Factory that creates N panels."""
    def _make(n):
        return [DriftmapPlotWidget(data_model, app) for _ in range(n)]
    return _make


class TestGridComputation:
    """_compute_grid_dimensions should produce a valid layout."""

    def test_explicit_grid(self):
        r, c = MultiSessionDriftmapWidget._compute_grid_dimensions(6, (2, 3))
        assert (r, c) == (2, 3)

    def test_explicit_grid_mismatch_raises(self):
        with pytest.raises(ValueError, match="expects 6 panels but got 4"):
            MultiSessionDriftmapWidget._compute_grid_dimensions(4, (2, 3))

    @pytest.mark.parametrize("n, expected", [
        (1, (1, 1)),
        (2, (1, 2)),
        (3, (2, 2)),
        (4, (2, 2)),
        (5, (2, 3)),
        (6, (2, 3)),
        (9, (3, 3)),
    ])
    def test_auto_grid(self, n, expected):
        r, c = MultiSessionDriftmapWidget._compute_grid_dimensions(n, None)
        assert (r, c) == expected
        assert r * c >= n


class TestMultiWidget:
    """MultiSessionDriftmapWidget with real panels."""

    def test_is_qwidget(self, make_panels):
        panels = make_panels(2)
        multi = MultiSessionDriftmapWidget(panels)
        assert isinstance(multi, QtWidgets.QWidget)
        multi.close()

    def test_y_axes_linked(self, make_panels):
        panels = make_panels(3)
        multi = MultiSessionDriftmapWidget(panels)

        ref = panels[0].p_scatter.getViewBox()
        for panel in panels[1:]:
            linked = panel.p_scatter.getViewBox().linkedView(1)
            assert linked is ref

        multi.close()

    def test_panels_parented(self, make_panels):
        panels = make_panels(2)
        multi = MultiSessionDriftmapWidget(panels)
        for panel in panels:
            assert panel.parent() is multi
        multi.close()

    def test_window_title(self, make_panels):
        panels = make_panels(2)
        multi = MultiSessionDriftmapWidget(panels)
        assert "multi session" in multi.windowTitle().lower()
        multi.close()
