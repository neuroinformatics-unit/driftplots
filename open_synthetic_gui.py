"""Open the DriftMapPlotWidget with synthetic KS4 data for visual inspection."""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Re-use the same generation logic as the test fixtures
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests.test_unit.conftest import (
    NUM_SPIKES,
    _generate_positions_and_templates,
    _make_whitening_matrix,
    _write_ks4_output,
)

from driftplots.driftplotter import DriftPlotter


def main():
    rng = np.random.default_rng(0)
    channel_locations, templates = _generate_positions_and_templates(rng)
    n_channels = channel_locations.shape[0]

    mid_t = templates.shape[1] // 2
    num_clusters = templates.shape[0]
    peak_channels = [
        int(np.argmax(np.abs(templates[c, mid_t, :]))) for c in range(num_clusters)
    ]
    whitening_mat, whitening_mat_inv = _make_whitening_matrix(rng, n_channels)
    whitened_templates = np.array(
        [template @ whitening_mat for template in templates], dtype=np.float32
    )

    spike_templates = np.array(
        [i % num_clusters for i in range(NUM_SPIKES)], dtype=np.int32
    )
    spike_times = np.sort(rng.uniform(1.0, 100.0, NUM_SPIKES))
    spike_depths = (
        channel_locations[np.array(peak_channels)[spike_templates], 1]
        + rng.uniform(-0.1, 0.1, NUM_SPIKES)
    )
    scaling_factors_first_session = 1.0 + 0.1 * rng.standard_normal(NUM_SPIKES)

    # Write KS4-format files to a temp directory
    tmp_dir = tempfile.mkdtemp(prefix="synthetic_ks4_")
    tmp_path = Path(tmp_dir)
    print(f"Writing synthetic KS4 output to: {tmp_path}")

    data = {
        "spike_times": spike_times,
        "spike_templates": spike_templates,
        "spike_depths": spike_depths,
        "whitened_templates": whitened_templates,
        "whitening_mat_inv": whitening_mat_inv,
        "channel_locations": channel_locations,
        "scaling_factors_first_session": scaling_factors_first_session,
    }
    _write_ks4_output(tmp_path, data, "scaling_factors_first_session")

    # Launch the widget via DriftPlotter (same as tests)
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    plotter = DriftPlotter(tmp_path)
    widget = plotter.drift_map_plot_interactive(decimate=False, exclude_noise=False)
    widget.show()
    app.exec()


if __name__ == "__main__":
    main()
