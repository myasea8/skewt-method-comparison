import zarr
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import pandas as pd

def compute_lcl_for_group(store_path, group_key):
    g = zarr.open_group(store_path, mode="r")[group_key]
    p = g["pres"][:] * units.hPa
    T = g["tdry"][:] * units.celsius
    Td = g["dp"][:] * units.celsius

    mask = np.isfinite(p) & np.isfinite(T) & np.isfinite(Td)
    if mask.sum() == 0:
        return group_key, np.nan, np.nan

    idx = np.argmax(p[mask].magnitude)  # surface = max pressure
    p0 = p[mask][idx]
    T0 = T[mask][idx]
    Td0 = Td[mask][idx]

    lcl_p, lcl_T = mpcalc.lcl(p0, T0, Td0)

    return group_key, lcl_p.magnitude, lcl_T.magnitude