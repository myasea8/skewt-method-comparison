import numpy as np
from metpy.units import units
import metpy.calc as mpcalc
from ..methods.MetpyIndices import compute_metpy_indices

MISSING_VALUES = {999, 999.0, 9999, 9999.0, 99999, 99999.0}

def read_cls_file(cls_path):
    """
    Read an NCAR / Aspen .cls sounding file.

    Returns
    -------
    p : ndarray (hPa)
    T : ndarray (degC)
    Td : ndarray (degC)
    """

    with open(cls_path, "r") as f:
        lines = f.readlines()

    # Find the dashed separator line
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("------"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Could not find data section in {cls_path}")

    data = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue

        data.append(row)

    if len(data) == 0:
        return None, None, None

    data = np.array(data)

    # Column indices from header
    p = data[:, 1]
    T = data[:, 2]
    Td = data[:, 3]

    # Mask Aspen missing values
    def mask_missing(arr):
        out = arr.astype(float)
        for mv in MISSING_VALUES:
            out[out == mv] = np.nan
        return out

    p = mask_missing(p)
    T = mask_missing(T)
    Td = mask_missing(Td)

    return p, T, Td


METHODS = {"metpy": compute_metpy_indices}

def compute_indices_for_cls(cls_path):

    p, T, Td = read_cls_file(cls_path)

    if p is None:
        return cls_path, np.nan, np.nan


    results = []

    for method_name, method_fn in METHODS.items():
        try:
            out = method_fn(p, T, Td)

            row = {
                "method": method_name,
                **out
            }

        except Exception:
            row = {
                "method": method_name,
                "lcl_p": None,
                "lfc_p": None,
                "el_p": None,
                "cape": None,
                "cin": None,
            }

        results.append(row)

    return results