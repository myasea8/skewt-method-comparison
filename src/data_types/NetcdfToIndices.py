import numpy as np
import xarray as xr
from datetime import datetime
import re

from ..methods.MetpyIndices import compute_metpy_indices
from ..methods.SharppyIndices import compute_sharppy_indices

#METHODS = {"metpy": compute_metpy_indices, "sharppy": compute_sharppy_indices}
METHODS = {"metpy": compute_metpy_indices}

def read_netcdf_sounding(nc_path):
    """
    Read a sounding from a NetCDF file.

    Returns
    -------
    p : ndarray (hPa)
    T : ndarray (degC)
    Td : ndarray (degC)
    """

    ds = xr.open_dataset(nc_path)

    p = ds["pres"].values
    T = ds["tdry"].values
    Td = ds["dp"].values

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


def extract_datetime_from_filename(nc_path):
    """
    Extract datetime from filename like:
    NCAR_M2HATS_ISS1_RS41_v1_20230825_220529_asc.nc
    """

    filename = nc_path.name  # works if nc_path is a Path object

    match = re.search(r'_(\d{8})_(\d{6})_', filename)

    if not match:
        raise ValueError(f"Could not extract datetime from {filename}")

    date_str, time_str = match.groups()

    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")

def compute_indices_for_netcdf(nc_path):

    p, T, Td = read_netcdf_sounding(nc_path)
    dt = extract_datetime_from_filename(nc_path)

    if p is None or len(p) == 0:
        return nc_path, np.nan, np.nan

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