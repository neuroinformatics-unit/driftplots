import inspect

from driftplots import DriftPlotter

MATPLOTLIB_ONLY_PARAMS = ["self", "add_histogram_plot", "weight_histogram_by_amplitude"]


def test_interactive_and_matplotlib_share_signature():
    """Check shared parameters between interactive and matplotlib
    methods have the same name, default and type annotation.
    """
    interactive = inspect.signature(DriftPlotter.drift_map_plot_interactive)
    matplotlib = inspect.signature(DriftPlotter.drift_map_plot_matplotlib)

    interactive_params = {k: v for k, v in interactive.parameters.items() if k != "self"}
    matplotlib_params = {k: v for k, v in matplotlib.parameters.items() if k not in MATPLOTLIB_ONLY_PARAMS}

    assert len(interactive_params) == len(matplotlib_params)

    for name, param in interactive_params.items():
        assert param == matplotlib_params[name]
