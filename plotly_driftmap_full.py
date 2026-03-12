from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import threading
import webbrowser

import dash
from dash import Input, Output, State, dcc, html
import numpy as np
import plotly.graph_objects as go

from more_playing import DriftMapView


DEFAULT_SORTER_OUTPUT = Path(
    r"Y:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\joe\scratch\derivatives\BG_046_24062025\shank_0\sorting\no_motion\sorter_output"
)


@dataclass
class SessionData:
    name: str
    sorter_output: Path
    spike_times: np.ndarray
    spike_amplitudes: np.ndarray
    spike_depths: np.ndarray
    spike_templates: np.ndarray
    templates: np.ndarray
    display_indices: np.ndarray
    total_spikes: int
    kept_spikes: int


def choose_decimation(total_spikes: int, max_points: int) -> int:
    if total_spikes <= max_points:
        return 1
    return int(np.ceil(total_spikes / max_points))


def load_session(
    sorter_output: Path,
    only_include_large_amplitude_spikes: bool,
    exclude_noise: bool,
    log_transform_amplitudes: bool,
    processing_decimate: int | None,
    large_amplitude_only_segment_size: float,
    display_max_points: int,
) -> SessionData:
    drift = DriftMapView(sorter_output)
    session = drift.get_session_data(
        only_include_large_amplitude_spikes=only_include_large_amplitude_spikes,
        decimate=processing_decimate,
        exclude_noise=exclude_noise,
        log_transform_amplitudes=log_transform_amplitudes,
        large_amplitude_only_segment_size=large_amplitude_only_segment_size,
    )

    spike_times = np.asarray(session["spike_times"])
    spike_amplitudes = np.asarray(session["spike_amplitudes"])
    spike_depths = np.asarray(session["spike_depths"])
    spike_templates = np.asarray(session["spike_templates"])
    templates = np.asarray(session["templates"])

    total_spikes = int(spike_times.size)
    kept_path = sorter_output / "kept_spikes.npy"
    kept_spikes = int(np.load(kept_path).sum()) if kept_path.exists() else total_spikes

    display_step = choose_decimation(total_spikes, display_max_points)
    display_indices = np.arange(total_spikes, dtype=np.int64)[::display_step]

    return SessionData(
        name=sorter_output.name,
        sorter_output=sorter_output,
        spike_times=spike_times,
        spike_amplitudes=spike_amplitudes,
        spike_depths=spike_depths,
        spike_templates=spike_templates,
        templates=templates,
        display_indices=display_indices,
        total_spikes=total_spikes,
        kept_spikes=kept_spikes,
    )


def make_drift_figure(session: SessionData, selected_idx: int | None) -> go.Figure:
    x = session.spike_times[session.display_indices]
    y = session.spike_depths[session.display_indices]
    colors = session.spike_amplitudes[session.display_indices]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=2,
                color=colors,
                colorscale="Greys",
                reversescale=True,
                opacity=0.65,
                showscale=True,
                colorbar=dict(title="Amplitude"),
            ),
            hovertemplate="time=%{x}<br>depth=%{y}<extra></extra>",
            name="spikes",
        )
    )

    if selected_idx is not None and 0 <= selected_idx < session.total_spikes:
        fig.add_trace(
            go.Scattergl(
                x=[session.spike_times[selected_idx]],
                y=[session.spike_depths[selected_idx]],
                mode="markers",
                marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(color="red", width=2)),
                name="selected",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Drift map — {session.name}",
        xaxis_title="Time (sample index)",
        yaxis_title="Depth (µm)",
        template="plotly_white",
        margin=dict(l=70, r=20, t=50, b=60),
        height=800,
    )
    return fig


def max_waveform_for_spike(session: SessionData, spike_index: int) -> np.ndarray:
    template_idx = int(session.spike_templates[spike_index])
    scaled = session.templates[template_idx] * session.spike_amplitudes[spike_index]
    peak_channel = int(np.argmax(np.max(np.abs(scaled), axis=0)))
    return scaled[:, peak_channel]


def heatmap_for_spike(session: SessionData, spike_index: int) -> np.ndarray:
    template_idx = int(session.spike_templates[spike_index])
    scaled = session.templates[template_idx] * session.spike_amplitudes[spike_index]
    contains_data_idx = np.where(scaled[0, :] != 0)[0]
    if contains_data_idx.size == 0:
        contains_data_idx = np.where(np.any(scaled != 0, axis=0))[0]
    if contains_data_idx.size == 0:
        return scaled
    return scaled[:, contains_data_idx]


def make_panel_figure(
    session: SessionData,
    selected_idx: int | None,
    view_mode: str,
    fix_ylim: bool,
    y_min: float,
    y_max: float,
) -> go.Figure:
    fig = go.Figure()

    if selected_idx is None:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Click a spike in drift map",
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template="plotly_white", height=800, margin=dict(l=70, r=20, t=50, b=60))
        return fig

    template_id = int(session.spike_templates[selected_idx])
    if view_mode == "max_waveform":
        waveform = max_waveform_for_spike(session, selected_idx)
        fig.add_trace(
            go.Scatter(
                x=np.arange(waveform.size),
                y=waveform,
                mode="lines",
                line=dict(color="black", width=2),
                name="waveform",
            )
        )
        fig.update_xaxes(title="Sample")
        fig.update_yaxes(title="Amplitude")
        if fix_ylim:
            fig.update_yaxes(range=[y_min, y_max])
    else:
        data = heatmap_for_spike(session, selected_idx)
        zmax = float(np.max(np.abs(data))) if data.size > 0 else 1.0
        fig.add_trace(
            go.Heatmap(
                z=data.T,
                colorscale="RdBu",
                zmid=0,
                zmin=-zmax,
                zmax=zmax,
                colorbar=dict(title="Amplitude"),
            )
        )
        fig.update_xaxes(title="Sample")
        fig.update_yaxes(title="Channel")

    fig.update_layout(
        title=f"Template {template_id}",
        template="plotly_white",
        height=800,
        margin=dict(l=70, r=20, t=50, b=60),
    )
    return fig


def build_app(sessions: dict[str, SessionData]) -> dash.Dash:
    app = dash.Dash(__name__)

    options = [{"label": f"{k} ({v.total_spikes:,} spikes)", "value": k} for k, v in sessions.items()]
    default_session = options[0]["value"]

    app.layout = html.Div(
        [
            html.H3("Driftmap Viewer (Plotly)", style={"marginBottom": "8px"}),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id="drift-graph", config={"displaylogo": False}),
                        ],
                        style={"width": "64%", "display": "inline-block", "verticalAlign": "top"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="panel-graph", config={"displaylogo": False}),
                        ],
                        style={"width": "36%", "display": "inline-block", "verticalAlign": "top"},
                    ),
                ],
                style={"padding": "6px 0"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Session"),
                            dcc.Dropdown(id="session-id", options=options, value=default_session, clearable=False),
                        ],
                        style={"minWidth": "250px", "maxWidth": "420px", "flex": "1 1 300px"},
                    ),
                    html.Div(
                        [
                            html.Label("Right panel view"),
                            dcc.RadioItems(
                                id="view-mode",
                                options=[
                                    {"label": "Max waveform", "value": "max_waveform"},
                                    {"label": "Heatmap", "value": "heatmap"},
                                ],
                                value="max_waveform",
                                inline=True,
                            ),
                        ],
                        style={"minWidth": "260px", "flex": "1 1 300px"},
                    ),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="fix-ylim",
                                options=[{"label": "Fix y-limits", "value": "on"}],
                                value=[],
                                inline=True,
                            ),
                            html.Div(
                                [
                                    html.Label("Y min"),
                                    dcc.Input(id="y-min", type="number", value=-200, style={"width": "90px"}),
                                    html.Label("Y max", style={"marginLeft": "10px"}),
                                    dcc.Input(id="y-max", type="number", value=200, style={"width": "90px"}),
                                ],
                                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                            ),
                        ],
                        style={"minWidth": "300px", "flex": "1 1 340px"},
                    ),
                    html.Div(id="stats", style={"minWidth": "280px", "flex": "1 1 320px"}),
                    html.Div(id="selected-spike-info", style={"minWidth": "280px", "flex": "1 1 320px"}),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "14px",
                    "alignItems": "center",
                    "padding": "12px",
                    "borderTop": "1px solid #ddd",
                },
            ),
            dcc.Store(id="selected-spike-store", data={"session": default_session, "index": None}),
        ],
        style={"fontFamily": "Arial, sans-serif", "padding": "8px 12px"},
    )

    @app.callback(
        Output("selected-spike-store", "data"),
        Input("session-id", "value"),
        prevent_initial_call=True,
    )
    def clear_selection_on_session_change(session_id):
        return {"session": session_id, "index": None}

    @app.callback(
        Output("selected-spike-store", "data", allow_duplicate=True),
        Input("drift-graph", "clickData"),
        State("session-id", "value"),
        State("selected-spike-store", "data"),
        prevent_initial_call=True,
    )
    def update_selected_spike(click_data, session_id, store):
        if click_data is None or "points" not in click_data or len(click_data["points"]) == 0:
            return store
        point = click_data["points"][0]
        point_index = int(point.get("pointIndex", -1))
        session = sessions[session_id]
        if point_index < 0 or point_index >= session.display_indices.size:
            return store
        selected = int(session.display_indices[point_index])
        return {"session": session_id, "index": selected}

    @app.callback(
        Output("drift-graph", "figure"),
        Output("panel-graph", "figure"),
        Output("stats", "children"),
        Output("selected-spike-info", "children"),
        Input("session-id", "value"),
        Input("selected-spike-store", "data"),
        Input("view-mode", "value"),
        Input("fix-ylim", "value"),
        Input("y-min", "value"),
        Input("y-max", "value"),
    )
    def update_views(session_id, selected_store, view_mode, fix_ylim_values, y_min, y_max):
        session = sessions[session_id]
        selected_idx = None
        if selected_store and selected_store.get("session") == session_id:
            selected_idx = selected_store.get("index")

        fix_ylim = "on" in (fix_ylim_values or [])
        y_min = -200 if y_min is None else float(y_min)
        y_max = 200 if y_max is None else float(y_max)

        drift_fig = make_drift_figure(session, selected_idx)
        panel_fig = make_panel_figure(session, selected_idx, view_mode, fix_ylim, y_min, y_max)

        stats_text = html.Div(
            [
                html.Div(f"Sorter output: {session.sorter_output}"),
                html.Div(f"Total spikes: {session.total_spikes:,}"),
                html.Div(f"Kept spikes:  {session.kept_spikes:,}"),
                html.Div(f"Points plotted: {session.display_indices.size:,}"),
            ]
        )

        if selected_idx is None:
            selected_text = "Selected spike: none"
        else:
            selected_text = (
                f"Selected spike idx={selected_idx:,} | "
                f"time={session.spike_times[selected_idx]:.0f} | "
                f"depth={session.spike_depths[selected_idx]:.2f}"
            )

        return drift_fig, panel_fig, stats_text, selected_text

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Full Plotly driftmap viewer")
    parser.add_argument(
        "--sorter-output",
        action="append",
        type=Path,
        help="Path(s) to Kilosort sorter_output. Repeat this argument for multiple sessions.",
    )
    parser.add_argument("--display-max-points", type=int, default=400_000)
    parser.add_argument("--processing-decimate", type=int, default=None)
    parser.add_argument("--large-amplitude-only-segment-size", type=float, default=800.0)
    parser.add_argument("--exclude-noise", action="store_true")
    parser.add_argument("--log-transform-amplitudes", action="store_true")
    parser.add_argument("--include-all-amplitudes", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically open the Dash app in your default browser.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sorter_outputs = args.sorter_output or [DEFAULT_SORTER_OUTPUT]
    sessions: dict[str, SessionData] = {}

    for idx, sorter_output in enumerate(sorter_outputs):
        sorter_output = Path(sorter_output)
        if not sorter_output.exists():
            raise FileNotFoundError(f"Sorter output path does not exist: {sorter_output}")

        session = load_session(
            sorter_output=sorter_output,
            only_include_large_amplitude_spikes=not args.include_all_amplitudes,
            exclude_noise=args.exclude_noise,
            log_transform_amplitudes=args.log_transform_amplitudes,
            processing_decimate=args.processing_decimate,
            large_amplitude_only_segment_size=args.large_amplitude_only_segment_size,
            display_max_points=args.display_max_points,
        )
        session_key = f"session_{idx + 1}"
        sessions[session_key] = session

        print(f"[{session_key}] {session.sorter_output}")
        print(f"  total_spikes:   {session.total_spikes:,}")
        print(f"  kept_spikes:    {session.kept_spikes:,}")
        print(f"  points_plotted: {session.display_indices.size:,}")

    app = build_app(sessions)

    if args.open_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Timer(1.0, lambda: webbrowser.open(url, new=2)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
