from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


DEFAULT_SORTER_OUTPUT = Path(
    r"Y:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\joe\scratch\derivatives\BG_046_24062025\shank_0\sorting\no_motion\sorter_output"
)


def load_spike_arrays(sorter_output: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    spike_times = np.load(sorter_output / "spike_times.npy", mmap_mode="r")
    spike_amplitudes = np.load(sorter_output / "amplitudes.npy", mmap_mode="r")
    spike_depths = np.load(sorter_output / "spike_positions.npy", mmap_mode="r")[:, 1]

    kept_path = sorter_output / "kept_spikes.npy"
    kept_spikes = int(np.load(kept_path, mmap_mode="r").sum()) if kept_path.exists() else spike_times.size

    return spike_times, spike_amplitudes, spike_depths, kept_spikes


def choose_decimation(total_spikes: int, max_points: int) -> int:
    if total_spikes <= max_points:
        return 1
    return int(np.ceil(total_spikes / max_points))


def build_figure(
    spike_times: np.ndarray,
    spike_amplitudes: np.ndarray,
    spike_depths: np.ndarray,
    decimation: int,
) -> go.Figure:
    x = spike_times[::decimation]
    y = spike_depths[::decimation]
    colors = spike_amplitudes[::decimation]

    fig = go.Figure(
        data=go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=2,
                color=colors,
                colorscale="Greys",
                showscale=True,
                colorbar=dict(title="Amplitude"),
                opacity=0.65,
            ),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Drift map (Plotly Scattergl)",
        xaxis_title="Spike time (sample index)",
        yaxis_title="Depth (µm)",
        template="plotly_white",
        height=800,
        margin=dict(l=70, r=20, t=60, b=60),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plotly drift map for Kilosort output")
    parser.add_argument("--sorter-output", type=Path, default=DEFAULT_SORTER_OUTPUT)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--decimate", type=int, default=None)
    args = parser.parse_args()

    sorter_output = args.sorter_output
    if not sorter_output.exists():
        raise FileNotFoundError(f"Sorter output path does not exist: {sorter_output}")

    spike_times, spike_amplitudes, spike_depths, kept_spikes = load_spike_arrays(sorter_output)
    total_spikes = int(spike_times.size)

    decimation = args.decimate or choose_decimation(total_spikes, args.max_points)

    print(f"sorter_output: {sorter_output}")
    print(f"total_spikes:  {total_spikes:,}")
    print(f"kept_spikes:   {kept_spikes:,}")
    print(f"decimation:    1/{decimation}")
    print(f"points_plotted:{int(np.ceil(total_spikes / decimation)):,}")

    fig = build_figure(spike_times, spike_amplitudes, spike_depths, decimation)
    fig.show(renderer="browser")


if __name__ == "__main__":
    main()
