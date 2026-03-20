# src/methods/metpy_indices.py

import numpy as np
import metpy.calc as mpcalc
from metpy.units import units


def compute_metpy_indices(p, T, Td):

    # Define units
    p = p * units.hPa
    T = T * units.celsius
    Td = Td * units.celsius

    # Surface parcel
    p0 = p[0]
    T0 = T[0]
    Td0 = Td[0]

    # Parcel profile
    parcel = mpcalc.parcel_profile(p, T0, Td0).to('degC')

    # Indices
    lcl_p, lcl_T = mpcalc.lcl(p0, T0, Td0)
    lfc_p, lfc_T = mpcalc.lfc(p, T, Td, which='bottom')
    el_p, el_T = mpcalc.el(p, T, Td, parcel, which='bottom')
    cape, cin = mpcalc.cape_cin(p, T, Td, parcel, which_lfc='bottom', which_el='bottom')

    return {
        "lcl_p": lcl_p.magnitude,
        "lcl_T": lcl_T.magnitude,
        "lfc_p": np.nan if lfc_p is None else lfc_p.magnitude,
        "lfc_T": np.nan if lfc_T is None else lfc_T.magnitude,
        "el_p": np.nan if el_p is None else el_p.magnitude,
        "el_T": np.nan if el_T is None else el_T.magnitude,
        "cape": cape.magnitude,
        "cin": cin.magnitude,
    }