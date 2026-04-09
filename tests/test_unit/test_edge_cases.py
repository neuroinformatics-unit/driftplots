"""Edge-case tests with tiny synthetic data."""
import matplotlib
import numpy as np
import pytest

from driftplots.data_model import DataModel
from driftplots.mpl_plotting import plot_matplotlib

matplotlib.use("Agg")


def _make_data_model(spike_times, spike_amplitudes, spike_depths, spike_clusters,
                     num_templates=2, num_samples=61, num_channels=10):
    """Build a minimal DataModel from plain arrays."""
    templates = np.random.default_rng(0).standard_normal(
        (num_templates, num_samples, num_channels)
    ).astype(np.float32)
    channel_locations = np.column_stack([
        np.zeros(num_channels),
        np.arange(num_channels, dtype=float) * 20,
    ])
    return DataModel(
        spike_times, spike_amplitudes, spike_depths, spike_clusters,
        templates, channel_locations,
    )


# ---------------------------------------------------------------------------
# Single spike
# ---------------------------------------------------------------------------
class TestSingleSpike:

    @pytest.fixture()
    def model(self):
        return _make_data_model(
            np.array([1.0]), np.array([5.0]), np.array([100.0]), np.array([0]),
        )

    def test_scatter_data(self, model):
        times, depths, amps = model.get_scatter_data()
        assert times.size == 1

    def test_amplitude_colors(self, model):
        colors = model.compute_amplitude_colors("linear", 10)
        assert colors.shape == (1, 4)

    def test_matplotlib_renders(self, model):
        fig = plot_matplotlib(model, "linear", 10, 5.0, False, False)
        assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# All-same-amplitude (ptp == 0)
# ---------------------------------------------------------------------------
class TestUniformAmplitude:

    @pytest.fixture()
    def model(self):
        n = 50
        return _make_data_model(
            np.arange(n, dtype=float),
            np.full(n, 3.0),
            np.linspace(0, 200, n),
            np.zeros(n, dtype=int),
        )

    def test_colors_do_not_nan(self, model):
        colors = model.compute_amplitude_colors("linear", 10)
        assert not np.any(np.isnan(colors.astype(float)))

    def test_histogram_weighted(self, model):
        """Weighted histogram with ptp==0 triggers a divide-by-zero."""
        with pytest.warns(RuntimeWarning, match="invalid value"):
            _, values = model.compute_activity_histogram(True)


# ---------------------------------------------------------------------------
# Decimate larger-than-data factor
# ---------------------------------------------------------------------------
class TestDecimation:

    def test_decimate_larger_than_n(self):
        from driftplots.data_loader import DataLoader
        from pathlib import Path

        sorter_path = Path(__file__).parent.parent.parent / "examples" / "example_data" / "sorting" / "sorter_output"
        loader = DataLoader(sorter_path)
        model = loader.get_processed_data(
            exclude_noise=False, decimate=100_000,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        # decimate factor far exceeds num_spikes → very few spikes remain
        assert model.spike_times.size >= 1


# ---------------------------------------------------------------------------
# Amplitude filtering modes
# ---------------------------------------------------------------------------
class TestAmplitudeFiltering:

    @pytest.fixture()
    def loader(self):
        from pathlib import Path
        from driftplots.data_loader import DataLoader
        sorter_path = Path(__file__).parent.parent.parent / "examples" / "example_data" / "sorting" / "sorter_output"
        return DataLoader(sorter_path)

    def test_percentile_filter(self, loader):
        model = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode="percentile", filter_amplitude_values=(25, 75),
        )
        full = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        assert model.spike_times.size < full.spike_times.size

    def test_absolute_filter(self, loader):
        full = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode=None, filter_amplitude_values=(),
        )
        median_amp = np.median(np.abs(full.spike_amplitudes))
        model = loader.get_processed_data(
            exclude_noise=False, decimate=False,
            filter_amplitude_mode="absolute",
            filter_amplitude_values=(median_amp, np.abs(full.spike_amplitudes).max()),
        )
        assert model.spike_times.size < full.spike_times.size
