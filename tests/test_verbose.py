import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from driftplots import DriftPlotter


class TestVerbosity:
    """verbose=True should print messages; verbose=False should suppress them."""

    # ------------------------------------------------------------------
    # DriftPlotter construction
    # ------------------------------------------------------------------

    def test_constructor_verbose_true_prints(self, synthetic_ks4_output, capsys):
        DriftPlotter(synthetic_ks4_output, verbose=True)
        out = capsys.readouterr().out
        assert "Loading data from" in out
        assert "spikes" in out.lower()

    def test_constructor_verbose_false_suppresses(self, synthetic_ks4_output, capsys):
        DriftPlotter(synthetic_ks4_output, verbose=False)
        out = capsys.readouterr().out
        assert out == ""

    # ------------------------------------------------------------------
    # get_processed_data via drift_map_plot_matplotlib
    # ------------------------------------------------------------------

    def test_matplotlib_verbose_true_prints(self, synthetic_ks4_output, capsys):
        plotter = DriftPlotter(synthetic_ks4_output, verbose=False)
        capsys.readouterr()  # clear construction output
        fig = plotter.drift_map_plot_matplotlib(verbose=True)
        out = capsys.readouterr().out
        assert "spikes" in out.lower()
        plt.close(fig)

    def test_matplotlib_verbose_false_suppresses(self, synthetic_ks4_output, capsys):
        plotter = DriftPlotter(synthetic_ks4_output, verbose=False)
        capsys.readouterr()
        fig = plotter.drift_map_plot_matplotlib(verbose=False)
        out = capsys.readouterr().out
        assert out == ""
        plt.close(fig)

    # ------------------------------------------------------------------
    # get_processed_data via drift_map_plot_interactive
    # ------------------------------------------------------------------

    def test_interactive_verbose_true_prints(self, synthetic_ks4_output, capsys):
        plotter = DriftPlotter(synthetic_ks4_output, verbose=False)
        capsys.readouterr()
        widget = plotter.drift_map_plot_interactive(verbose=True)
        out = capsys.readouterr().out
        assert "spikes" in out.lower()
        widget.close()

    def test_interactive_verbose_false_suppresses(self, synthetic_ks4_output, capsys):
        plotter = DriftPlotter(synthetic_ks4_output, verbose=False)
        capsys.readouterr()
        widget = plotter.drift_map_plot_interactive(verbose=False)
        out = capsys.readouterr().out
        assert out == ""
        widget.close()
