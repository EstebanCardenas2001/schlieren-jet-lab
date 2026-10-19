"""Gas properties: Gladstone-Dale relation, dimensional scaling, rarefaction.

Gladstone-Dale: n - 1 = K * rho. For air at visible wavelengths
K ~ 2.26e-4 m^3/kg (Merzkirch, "Flow Visualization", 2nd ed., 1987); it varies
by only ~1% across the visible band. At 20 C and 1 atm this gives
n - 1 = 2.72e-4, matching standard-air refractometry (~2.71e-4 at 633 nm).
"""

from __future__ import annotations

import math

K_AIR = 2.26e-4          # Gladstone-Dale constant of air [m^3/kg]
R_AIR = 287.05           # specific gas constant of air [J/(kg K)]
T_ROOM = 293.15          # [K]
D_AIR = 3.7e-10          # kinetic (hard-sphere) diameter of an air molecule [m]
K_BOLTZMANN = 1.380649e-23


def density(p: float, T: float = T_ROOM, R: float = R_AIR) -> float:
    """Ideal-gas density [kg/m^3] from pressure [Pa] and temperature [K]."""
    return p / (R * T)


def refractive_index(rho, K: float = K_AIR):
    """n = 1 + K rho (works on floats, numpy arrays and torch tensors)."""
    return 1.0 + K * rho


def mean_free_path(p: float, T: float = T_ROOM, d: float = D_AIR) -> float:
    """Hard-sphere mean free path [m]: kT / (sqrt(2) pi d^2 p)."""
    return K_BOLTZMANN * T / (math.sqrt(2.0) * math.pi * d * d * p)
