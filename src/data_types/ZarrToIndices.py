import numpy as np
from metpy.units import units
import zarr
from datetime import datetime
import re
from ..methods.MetpyIndices import compute_metpy_indices

METHODS = {"metpy": compute_metpy_indices}


def read_zarr_sounding(zarr_path, group_key):
    """
    Read a single sounding from a Zarr store.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr store
    group_key : str
        Group name (typically datetime string)

    Returns
    -------
    p : ndarray (hPa)
    T : ndarray (degC)
    Td : ndarray (degC)
    """

    g = zarr.open_group(zarr_path, mode="r")[group_key]

    p = g["pres"][:]
    T = g["tdry"][:]
    Td = g["dp"][:]

    # Remove entries where pressure is nan
    valid = ~np.isnan(p)
    p = p[valid]
    T = T[valid]
    Td = Td[valid]

    # Remove duplicate pressure levels
    _, unique_idx = np.unique(p, return_index=True)
    p = p[unique_idx]
    T = T[unique_idx]
    Td = Td[unique_idx]

    # Sort pressure from greatest to least
    sort_idx = np.argsort(p)[::-1]
    p = p[sort_idx]
    T = T[sort_idx]
    Td = Td[sort_idx]

    return p, T, Td

def extract_datetime_from_group(group_key):
    """
    Extract datetime from a string like:
    NCAR_M2HATS_ISS1_RS41_v1_20230825_220529_asc
    """

    match = re.search(r'_(\d{8})_(\d{6})_', group_key)

    if not match:
        raise ValueError(f"Could not extract datetime from {group_key}")

    date_str, time_str = match.groups()

    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")


def compute_indices_for_zarr(zarr_path, group_key):

    p, T, Td = read_zarr_sounding(zarr_path, group_key)
    dt = extract_datetime_from_group(group_key)

    if p is None or len(p) == 0:
        return group_key, np.nan, np.nan

    results = []

    for method_name, method_fn in METHODS.items():

        out = method_fn(p, T, Td)

        row = {
            "method": method_name,
            "datetime": dt,
            **out
        }

        results.append(row)

    return results