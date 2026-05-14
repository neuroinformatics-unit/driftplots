from __future__ import annotations

from pathlib import Path
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyqtgraph.Qt import QtWidgets

from driftplots import mpl_plotting
from driftplots.data_loader import DataLoader
from driftplots.interactive.driftmap_plot_widget import DriftmapPlotWidget

# test ideas:
# check signatures match default args between interactive and matplotlib


class DriftPlotter:
    """Load Kilosort or SpikeInterface output and plot drift maps.

    On construction, spike data is loaded from a Kilosort output directory
    or a SpikeInterface ``SortingAnalyzer`` and stored as read-only arrays.

    Parameters
    ----------
    path_or_analyzer
        Path to a Kilosort sorter output directory, or a SpikeInterface
        ``SortingAnalyzer`` object.  When a path is given it must contain
        exactly one ``kilosort*.log`` file, which is used to detect the
        Kilosort version.
    """

    def __init__(self, path_or_analyzer: str | Path, verbose: bool = True) -> None:
        """Load spike data from a Kilosort output directory or SortingAnalyzer.

        Parameters
        ----------
        path_or_analyzer
            Path to a Kilosort sorter output directory, or a SpikeInterface
            ``SortingAnalyzer`` object.
        verbose :
            If `True`, messages are printed.
        """
        self._data_loader = DataLoader(path_or_analyzer, verbose)

    def drift_map_plot_interactive(
        self,
        decimate: int | bool | None | Literal["estimate"] = "estimate",
        good_units_only: bool | str = False,
        amplitude_cmap_scaling: str | tuple[float, float] = "linear",
        n_color_bins: int = 20,
        point_size: float = 5.0,
        filter_amplitude_mode: str | None = None,
        filter_amplitude_values: tuple[float, ...] = (),
        title: bool | str | None = None,
        verbose: bool = True,
    ) -> DriftmapPlotWidget:
        """Create an interactive pyqtgraph-based drift map widget.

        Parameters
        ----------
        decimate
            Thin the spike dataset before plotting.  Pass ``"estimate"`` to
            automatically reduce spikes to a reasonable count (≈ 100 000).
            Pass ``False``, ``None``, or ``0`` to disable decimation.  Pass
            an integer *n* to keep every *n*-th spike.
        good_units_only
            If ``True``, only spikes beloning to "good" units are displayed.
            For Kilosort, this is taken from the
            cluster_groups.csv / cluster_group.tsv file that reflects labels
            set in Phy. For a SortingAnalyzer, a string must be passed.
            The labels are taken from the sorting property with the
            passed name (e.g. "KSLabel").
        amplitude_cmap_scaling
            Controls how spike amplitudes are mapped to the greyscale
            colormap.  Pass ``"linear"`` or ``"log2"`` or ``"log10"`` for automatic
            scaling, or a ``(min, max)`` tuple to set explicit bounds.
            When explicit bounds are set, the scaling is linear.
        n_color_bins
            Number of discrete greyscale bins used to colour spikes by
            amplitude.
        point_size
            Diameter of each scatter point in pixels.
        filter_amplitude_mode
            Controls how spikes are filtered based on amplitude before plotting.
            ``"percentile"`` treats the bounds as percentile ranks;
            ``"absolute"`` treats them as raw amplitude values.
            ``None`` disables amplitude filtering.
        filter_amplitude_values
            ``(low, high)`` bounds for amplitude filtering, used as set by
            ``filter_amplitude_mode``.  Ignored when ``filter_amplitude_mode``
            is ``None``.
        title
            Plot title.  Pass a string to set a custom title, ``True`` to
            use a default title, or ``None`` / ``False`` to suppress the
            title entirely.
        verbose :
            If `True`, messages are printed.

        Returns
        -------
        DriftmapPlotWidget
            The pyqtgraph widget.  The widget is already populated but not
            yet shown; call ``app.exec()`` to display it.
        """
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        processed_data = self._data_loader.get_processed_data(
            exclude_noise,
            decimate,
            filter_amplitude_mode,
            filter_amplitude_values,
            verbose,
        )

        self.plot = DriftmapPlotWidget(
            processed_data,
            app,
            amplitude_cmap_scaling=amplitude_cmap_scaling,
            n_color_bins=n_color_bins,
            point_size=point_size,
            title=title,
        )

        return self.plot

    def drift_map_plot_matplotlib(
        self,
        decimate: int | bool | None | Literal["estimate"] = "estimate",
        good_units_only: bool | str = False,
        amplitude_cmap_scaling: str | tuple[float, float] = "linear",
        n_color_bins: int = 20,
        point_size: float = 5.0,
        filter_amplitude_mode: str | None = None,
        filter_amplitude_values: tuple[float, ...] = (),
        add_histogram_plot: bool = False,
        weight_histogram_by_amplitude: bool = False,
        title: bool | str | None = None,
        ax: Axes | None = None,
        verbose: bool = True,
    ) -> Figure:
        """Create a static Matplotlib drift map figure.

        Parameters
        ----------
        decimate
            Thin the spike dataset before plotting.  Pass ``"estimate"`` to
            automatically reduce spikes to a reasonable count (≈ 100 000).
            Pass ``False``, ``None``, or ``0`` to disable decimation.  Pass
            an integer *n* to keep every *n*-th spike.
        good_units_only
            If ``True``, only spikes beloning to "good" units are displayed.
            For Kilosort, this is taken from the
            cluster_groups.csv / cluster_group.tsv file that reflects labels
            set in Phy. For a SortingAnalyzer, a string must be passed.
            The labels are taken from the sorting property with the
            passed name (e.g. "KSLabel").
        amplitude_cmap_scaling
            Controls how spike amplitudes are mapped to the greyscale
            colormap.  Pass ``"linear"`` or ``"log2"`` or ``"log10"`` for automatic
            scaling, or a ``(min, max)`` tuple to set explicit bounds.
            When explicit bounds are set, the scaling is linear.
        n_color_bins
            Number of discrete greyscale bins used to colour spikes by
            amplitude.
        point_size
            Diameter of each scatter point in pixels.
        filter_amplitude_mode
            Controls how spikes are filtered based on amplitude before plotting.
            ``"percentile"`` treats the bounds as percentile ranks;
            ``"absolute"`` treats them as raw amplitude values.
            ``None`` disables amplitude filtering.
        filter_amplitude_values
            ``(low, high)`` bounds for amplitude filtering, used as set by
            ``filter_amplitude_mode``.  Ignored when ``filter_amplitude_mode``
            is ``None``.
        add_histogram_plot
            If ``True``, add a side panel showing a depth histogram of
            spike activity.
        weight_histogram_by_amplitude
            If ``True``, weight the depth histogram by spike amplitude
            rather than counting spikes uniformly.  Only used when
            ``add_histogram_plot`` is ``True``.
        title
            Plot title.  Pass a string to set a custom title, ``True`` to
            use a default title, or ``None`` / ``False`` to suppress the
            title entirely.
        ax
            Existing Matplotlib axis to draw the drift map on.  When
            ``add_histogram_plot`` is ``True``, a histogram axis is added
            beside this axis.
        verbose :
            If `True`, messages are printed.

        Returns
        -------
        Figure
            The populated Matplotlib figure.
        """
        processed_data = self._data_loader.get_processed_data(
            good_units_only,
            decimate,
            filter_amplitude_mode,
            filter_amplitude_values,
            verbose,
        )

        fig = mpl_plotting.plot_matplotlib(
            processed_data,
            amplitude_cmap_scaling,
            n_color_bins,
            point_size,
            add_histogram_plot,
            weight_histogram_by_amplitude,
            title=title,
            ax=ax,
        )

        return fig
