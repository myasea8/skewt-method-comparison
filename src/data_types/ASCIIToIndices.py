import numpy as np
from metpy.units import units
import metpy.calc as mpcalc
import re
from datetime import datetime
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

        if line.strip().startswith("------"):
            break

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


def extract_release_datetime(cls_path):
    """
    Extract UTC Release Time from a .cls file and return a datetime object.
    """
    with open(cls_path, 'r') as f:
        for line in f:
            if "UTC Release Time" in line:
                # Extract the date-time portion after the colon
                match = re.search(r":\s*(\d{4}),\s*(\d{2}),\s*(\d{2}),\s*(\d{2}):(\d{2}):(\d{2})", line)
                if match:
                    year, month, day, hour, minute, second = map(int, match.groups())
                    return datetime(year, month, day, hour, minute, second)

    raise ValueError("UTC Release Time not found in file.")


METHODS = {"metpy": compute_metpy_indices}

def compute_indices_for_cls(cls_path):

    p, T, Td = read_cls_file(cls_path)
    datetime = extract_release_datetime(cls_path)

    if p is None:
        return cls_path, np.nan, np.nan

    results = []

    for method_name, method_fn in METHODS.items():
        #try:
        out = method_fn(p, T, Td)

        row = {
            "method": method_name,
            "datetime": datetime,
            **out
        }
        """
        except Exception:
            row = {
                "method": method_name,
                "datetime": datetime,
                "lcl_p": None,
                "lfc_p": None,
                "el_p": None,
                "cape": None,
                "cin": None,
            }
        """
        results.append(row)

    return results