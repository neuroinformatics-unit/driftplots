from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
if TYPE_CHECKING:
    pass
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


def get_noise_mask(sorter_output):
    """"""
    if (cluster_path := sorter_output / "spike_clusters.npy").is_file():
        spike_clusters = np.load(cluster_path)
    else:
        raise NotImplementedError("spike_clusters.npy does not exist.")

    if not (
        (cluster_path := sorter_output / "cluster_groups.csv").is_file()
        or (cluster_path := sorter_output / "cluster_group.tsv").is_file()
    ):
        raise ValueError(
            f"`exclude_noise` is `True` but there is no `cluster_groups.csv` or `.tsv` "
            f"in the sorting output at: {sorter_output}"
        )

    cluster_ids, cluster_groups = load_cluster_groups(cluster_path)

    noise_cluster_ids = cluster_ids[cluster_groups == 0]

    exclude_bool_mask = np.isin(spike_clusters.ravel(), noise_cluster_ids)

    return exclude_bool_mask


def get_pooled_amplitudes(paths):
    """Load and concatenate amplitudes.npy from multiple sorter output paths.

    Parameters
    ----------
    paths : list of Path
        List of sorter output directories, each containing amplitudes.npy.

    Returns
    -------
    np.ndarray
        Concatenated amplitudes from all paths.
    """
    return np.concatenate([np.load(Path(p) / "amplitudes.npy").ravel() for p in paths])
