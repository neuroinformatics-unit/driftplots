from pathlib import Path

import numpy as np
import pytest
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from driftplots.data_loader import DataLoader
from driftplots.interactive.driftmap_plot_widget import DriftmapPlotWidget

SORTER_OUTPUT = Path(__file__).parent.parent.parent / "examples" / "example_data" / "sorting" / "sorter_output"

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def app():
    """One QApplication for the whole module."""
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
def widget(app, data_model):
    w = DriftmapPlotWidget(data_model, app)
    yield w
    w.close()


class TestWidgetCreation:
    """Basic sanity checks on the widget after construction."""

    def test_widget_is_qwidget(self, widget):
        assert isinstance(widget, QtWidgets.QWidget)

    def test_scatter_item_exists(self, widget):
        assert isinstance(widget.scatter, pg.ScatterPlotItem)

    def test_panel_plot_exists(self, widget):
        assert widget.panel_plot is not None


class TestScatterData:
    """The scatter plot should contain the correct spike data."""

    def test_scatter_point_count(self, widget, data_model):
        points = widget.scatter.data
        assert len(points) == data_model.spike_times.size

    def test_scatter_x_matches_spike_times(self, widget, data_model):
        x = np.array([pt[0] for pt in widget.scatter.data])
        np.testing.assert_array_almost_equal(x, data_model.spike_times.ravel())

    def test_scatter_y_matches_spike_depths(self, widget, data_model):
        y = np.array([pt[1] for pt in widget.scatter.data])
        np.testing.assert_array_almost_equal(y, data_model.spike_depths.ravel())


class TestControls:
    """UI controls should reflect initial config state."""

    def test_heatmap_radio_checked_by_default(self, widget):
        assert widget.radio_heatmap.isChecked()
        assert not widget.radio_heatmap_all.isChecked()

    def test_spinboxes_disabled_by_default(self, widget):
        assert not widget.ymin_spin.isEnabled()
        assert not widget.ymax_spin.isEnabled()

    def test_fix_limits_enables_spinboxes(self, widget):
        widget._fix_limits_cb.setChecked(True)
        assert widget.ymin_spin.isEnabled()
        assert widget.ymax_spin.isEnabled()

    def test_view_mode_toggle(self, widget):
        widget.radio_heatmap_all.setChecked(True)
        assert widget.cfgs["right_panel_view_mode"] == "heatmap_all_channels"

        widget.radio_heatmap.setChecked(True)
        assert widget.cfgs["right_panel_view_mode"] == "heatmap"

    def test_no_selected_spot_initially(self, widget):
        assert widget.selected_spot is None


class TestTitle:
    """Title parameter should set the scatter plot title."""

    def test_title_set(self, app, data_model):
        w = DriftmapPlotWidget(data_model, app, title="Test Title")
        assert w.p_scatter.titleLabel.text == "Test Title"
        w.close()

    def test_no_title(self, widget):
        assert widget.p_scatter.titleLabel.text == ""
