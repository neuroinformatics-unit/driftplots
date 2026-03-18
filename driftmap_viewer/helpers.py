from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
import pandas as pd
import numpy as np

def load_cluster_groups(cluster_path: Path) -> tuple[np.ndarray, ...]:
    """
    Load kilosort `cluster_groups` file, that contains a table of
    quality assignments, one per unit. These can be "noise", "mua", "good"
    or "unsorted".

    There is some slight formatting differences between the `.tsv` and `.csv`
    versions, presumably from different kilosort versions.

    This function was ported from Nick Steinmetz's `spikes` repository MATLAB code,
    https://github.com/cortex-lab/spikes

    Parameters
    ----------
    cluster_path : Path
        The full filepath to the `cluster_groups` tsv or csv file.

    Returns
    -------
    cluster_ids : np.ndarray
        (num_clusters,) Array of (integer) unit IDs.

    cluster_groups : np.ndarray
        (num_clusters,) Array of (integer) unit quality assignments, see code
        below for mapping to "noise", "mua", "good" and "unsorted".
    """
    cluster_groups_table = pd.read_csv(cluster_path, sep="\t")

    group_key = cluster_groups_table.columns[1]  # "groups" (csv) or "KSLabel" (tsv)

    for key, _id in zip(
            ["noise", "mua", "good", "unsorted"],
            ["0", "1", "2", "3"],
            # required as str to avoid pandas replace downcast FutureWarning
    ):
        cluster_groups_table[group_key] = cluster_groups_table[group_key].replace(key, _id)

    cluster_ids = cluster_groups_table["cluster_id"].to_numpy()
    cluster_groups = cluster_groups_table[group_key].astype(int).to_numpy()

    return cluster_ids, cluster_groups

# This is such a jankily written function fix it
def exclude_noise(sorter_output, spike_times, spike_amplitudes, spike_depths):
    """"""
    if (cluster_path := sorter_output / "spike_clusters.npy").is_file():  # TODO: this can be csv?!?!?
        spike_clusters = np.load(cluster_path)
    else:
        # this is a pain to have here,  I don't think this case is realistic.
        raise NotImplementedError("spike clusters.csv does not exist. Under what circumstance is this? probably very old.")
        # spike_clusters = spike_templates.copy()

    if ( # short circuit ensures cluster_path is assigned appropriately
        (cluster_path := sorter_output / "cluster_groups.csv").is_file()
        or (cluster_path := sorter_output / "cluster_group.tsv").is_file()
    ):
        cluster_ids, cluster_groups = load_cluster_groups(cluster_path)

        noise_cluster_ids = cluster_ids[cluster_groups == 0]
        not_noise_clusters_by_spike = ~np.isin(spike_clusters.ravel(),
                                               noise_cluster_ids)
        spike_times = spike_times[not_noise_clusters_by_spike]
        spike_amplitudes = spike_amplitudes[not_noise_clusters_by_spike]
        spike_depths = spike_depths[not_noise_clusters_by_spike]

        return spike_times, spike_amplitudes, spike_depths

    raise ValueError(
        f"`exclude_noise` is `True` but there is no `cluster_groups.csv` or `.tsv` "
        f"in the sorting output at: {sorter_output}"
    )
