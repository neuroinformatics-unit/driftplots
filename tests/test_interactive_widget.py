import numpy as np
import pytest
import pyqtgraph as pg

from driftplots.driftplotter import DriftPlotter

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture()
def widget_and_data(synthetic_ks4_output, synthetic_data):
    """Default widget created via the high-level API + synthetic ground truth."""
    plotter = DriftPlotter(synthetic_ks4_output)
    widget = plotter.drift_map_plot_interactive(decimate=False, exclude_noise=False)
    yield widget, synthetic_data
    widget.close()


class TestScatterData:
    """The scatter plot should contain the correct spike data."""

    def test_spike_times_match_synthetic(self, widget_and_data):
        widget, data = widget_and_data
        np.testing.assert_array_equal(widget.scatter.data["x"], data["spike_times"])
        np.testing.assert_array_equal(widget.processed_data.spike_times, data["spike_times"])

    def test_spike_depths_match_synthetic(self, widget_and_data):
        widget, data = widget_and_data
        np.testing.assert_array_almost_equal(widget.scatter.data["y"], data["spike_depths"])
        np.testing.assert_array_almost_equal(widget.processed_data.spike_depths, data["spike_depths"])

    def test_scatter_data_field_stores_spike_index(self, widget_and_data):
        """Each scatter point's data field should be its spike index."""
        widget, _ = widget_and_data
        stored_indices = np.array([point.data() for point in widget.scatter.points()])
        np.testing.assert_array_equal(stored_indices, np.arange(widget.processed_data.spike_times.size))


class TestConstructorParameters:
    """Non-default constructor arguments should take effect."""

    def test_custom_point_size(self, synthetic_ks4_output):
        plotter = DriftPlotter(synthetic_ks4_output)
        widget = plotter.drift_map_plot_interactive(decimate=False, exclude_noise=False, point_size=12.0)
        assert widget.scatter.opts["size"] == 12.0
        widget.close()

    @pytest.mark.parametrize("scaling, n_bins", [
        ("linear", 20),
        ("log2", 20),
        ("log10", 20),
        ((5.0, 15.0), 20),
        ("linear", 10),
    ])
    def test_brush_colors_match_compute_amplitude_colors(self, synthetic_ks4_output, scaling, n_bins):
        """Scatter brush RGBA should exactly match compute_amplitude_colors for every scaling/bins combo."""
        plotter = DriftPlotter(synthetic_ks4_output)
        widget = plotter.drift_map_plot_interactive(decimate=False, exclude_noise=False, amplitude_cmap_scaling=scaling, n_color_bins=n_bins)
        expected_colors = widget.processed_data.compute_amplitude_colors(scaling, n_bins)
        for i, point in enumerate(widget.scatter.points()):
            brush_color = point.brush().color()
            assert (brush_color.red(), brush_color.green(), brush_color.blue(), brush_color.alpha()) == tuple(expected_colors[i])
        widget.close()

    def test_title_set(self, synthetic_ks4_output):
        plotter = DriftPlotter(synthetic_ks4_output)
        widget = plotter.drift_map_plot_interactive(decimate=False, exclude_noise=False, title="Test Title")
        assert widget.p_scatter.titleLabel.text == "Test Title"
        widget.close()

    def test_no_title(self, widget_and_data):
        widget, _ = widget_and_data
        assert widget.p_scatter.titleLabel.text == ""


class TestControls:
    """UI controls should reflect initial config state."""

    def test_heatmap_radio_checked_by_default(self, widget_and_data):
        widget, _ = widget_and_data
        assert widget.radio_heatmap.isChecked()
        assert not widget.radio_heatmap_all.isChecked()

    def test_spinboxes_disabled_by_default(self, widget_and_data):
        widget, _ = widget_and_data
        assert not widget.ymin_spin.isEnabled()
        assert not widget.ymax_spin.isEnabled()

    def test_fix_limits_enables_spinboxes(self, widget_and_data):
        widget, _ = widget_and_data
        widget._fix_limits_cb.setChecked(True)
        assert widget.ymin_spin.isEnabled()
        assert widget.ymax_spin.isEnabled()

    def test_fix_limits_updates_config(self, widget_and_data):
        widget, _ = widget_and_data
        widget._fix_limits_cb.setChecked(True)
        assert widget.cfgs["left_panel_y_axis"]["on"] is True
        widget._fix_limits_cb.setChecked(False)
        assert widget.cfgs["left_panel_y_axis"]["on"] is False

    def test_spinbox_min_updates_config(self, widget_and_data):
        widget, _ = widget_and_data
        widget.ymin_spin.setValue(42.0)
        assert widget.cfgs["left_panel_y_axis"]["y_min"] == 42.0

    def test_spinbox_max_updates_config(self, widget_and_data):
        widget, _ = widget_and_data
        widget.ymax_spin.setValue(99.0)
        assert widget.cfgs["left_panel_y_axis"]["y_max"] == 99.0

    def test_view_mode_toggle(self, widget_and_data):
        widget, _ = widget_and_data
        widget.radio_heatmap_all.setChecked(True)
        assert widget.cfgs["right_panel_view_mode"] == "heatmap_all_channels"

        widget.radio_heatmap.setChecked(True)
        assert widget.cfgs["right_panel_view_mode"] == "heatmap"

    def test_no_selected_spot_initially(self, widget_and_data):
        widget, _ = widget_and_data
        assert widget.selected_spot is None


class TestClickUpdatesPanel:
    """Clicking a spike should display the correct template in the panel."""

    # one spike from each cluster
    SPIKE_INDICES_PARAMETERISATION = pytest.mark.parametrize(
        "spike_idx", [0, 1, 2], ids=["cluster0", "cluster1", "cluster2"]
    )

    def _click_spike(self, widget, spike_idx):
        """Simulate a scatter click by emitting sigClicked with the target point."""
        spot = widget.scatter.points()[spike_idx]
        widget.handle_click(None, [spot], None)

    @SPIKE_INDICES_PARAMETERISATION
    def test_panel_title_shows_correct_template_id(self, widget_and_data, spike_idx):
        widget, data = widget_and_data
        self._click_spike(widget, spike_idx)
        expected_template = data["spike_templates"][spike_idx]
        assert widget.panel_plot.titleLabel.text == f"Template {expected_template}"

    @SPIKE_INDICES_PARAMETERISATION
    def test_panel_image_matches_expected_template(self, widget_and_data, spike_idx):
        widget, data = widget_and_data
        self._click_spike(widget, spike_idx)
        template_id = data["spike_templates"][spike_idx]
        expected = data["expected_heatmaps"][template_id]

        image_item = [item for item in widget.panel_plot.items if hasattr(item, "image")][0]
        np.testing.assert_array_almost_equal(image_item.image, expected)

    @SPIKE_INDICES_PARAMETERISATION
    def test_click_sets_selected_spot(self, widget_and_data, spike_idx):
        widget, _ = widget_and_data
        self._click_spike(widget, spike_idx)
        assert widget.selected_spot is not None
        assert int(widget.selected_spot.data()) == spike_idx

    def test_click_none_points_is_noop(self, widget_and_data):
        """handle_click with None or empty points should not crash or change state."""
        widget, _ = widget_and_data
        widget.handle_click(None, None, None)
        assert widget.selected_spot is None
        widget.handle_click(None, [], None)
        assert widget.selected_spot is None

    def test_second_click_replaces_first(self, widget_and_data):
        """Clicking a second spike should update selected_spot and reset the first pen."""
        widget, _ = widget_and_data
        self._click_spike(widget, 0)
        first_spot = widget.selected_spot

        self._click_spike(widget, 5)
        assert int(widget.selected_spot.data()) == 5
        assert widget.selected_spot is not first_spot
        # First spot's pen should be cleared (None pen)
        assert first_spot.pen() == pg.mkPen(None)

    def test_view_mode_toggle_updates_panel_image(self, widget_and_data):
        """Toggling view mode after a click should redraw with the new mode."""
        widget, data = widget_and_data
        self._click_spike(widget, 0)

        # Switch to all_channels mode — has more channels (includes zeroed outer rows)
        widget.radio_heatmap_all.setChecked(True)
        expected_all_chans = widget.processed_data.get_template_heatmap(0, "heatmap_all_channels")
        image_item = [item for item in widget.panel_plot.items if hasattr(item, "image")][0]
        np.testing.assert_array_equal(image_item.image, expected_all_chans)

        # Switch back to signal-only mode — fewer channels, compare against ground truth
        widget.radio_heatmap.setChecked(True)
        template_id = data["spike_templates"][0]
        expected_signal_chans = data["expected_heatmaps"][template_id]
        image_item = [item for item in widget.panel_plot.items if hasattr(item, "image")][0]
        np.testing.assert_array_almost_equal(image_item.image, expected_signal_chans)

        # The two modes must produce different shapes (outer rows are zero)
        assert expected_signal_chans.shape[1] < expected_all_chans.shape[1]

    def test_view_mode_sets_channel_label(self, widget_and_data):
        """y-axis label 'channel' should only appear in heatmap_all_channels mode."""
        widget, _ = widget_and_data
        self._click_spike(widget, 0)

        widget.radio_heatmap.setChecked(True)
        assert widget.panel_plot.getAxis("left").labelText != "channel"

        widget.radio_heatmap_all.setChecked(True)
        assert widget.panel_plot.getAxis("left").labelText == "channel"

    def test_fix_limits_applies_image_levels(self, widget_and_data):
        """Enabling fix limits and setting spinboxes should apply levels to the image."""
        widget, _ = widget_and_data
        self._click_spike(widget, 0)

        # Without fix_limits, levels span the full data range
        image_item = [item for item in widget.panel_plot.items if hasattr(item, "image")][0]
        full_min, full_max = image_item.levels
        assert full_min < full_max

        # Enable fix_limits with a smaller range
        widget._fix_limits_cb.setChecked(True)
        widget.ymin_spin.setValue(-1.0)
        widget.ymax_spin.setValue(1.0)
        self._click_spike(widget, 0)

        image_item = [item for item in widget.panel_plot.items if hasattr(item, "image")][0]
        assert image_item.levels[0] == pytest.approx(-1.0)
        assert image_item.levels[1] == pytest.approx(1.0)
        assert image_item.levels[1] < full_max
