"""Reference correlations for underexpanded sonic jets.

* Mach disk location, Ashkenas & Sherman (1966), Rarefied Gas Dynamics 4:
      x_md / D = 0.67 * sqrt(p0 / p_a)
  an empirical fit to free-jet data for p0/p_a ~ 15 ... 17 000 (gamma-independent
  to the accuracy of the fit).
* Fully expanded (ideally expanded) jet: isentropic expansion from p0 to p_a,
      M_j = sqrt( 2/(g-1) * ((p0/p_a)^((g-1)/g) - 1) ),
  and, by mass conservation from a sonic throat of diameter D,
      (D_j/D)^2 = A_j/A* = (1/M_j) * [ 2/(g+1) * (1 + (g-1)/2 M_j^2) ]^((g+1)/(2(g-1))).
* Shock-cell spacing, Prandtl-Pack (vortex-sheet) model in Tam's form:
      L_s ~ 1.306 * sqrt(M_j^2 - 1) * D_j
  derived for *weakly* underexpanded jets (no Mach disk). It is shown here only
  as a reference; strongly underexpanded jets with a Mach disk are not expected
  to follow it.
"""

from __future__ import annotations

import numpy as np

GAMMA = 1.4
AS_COEFF = 0.67


def ashkenas_sherman(npr):
    return AS_COEFF * np.sqrt(np.asarray(npr, dtype=float))


def fully_expanded_mach(npr, g: float = GAMMA):
    npr = np.asarray(npr, dtype=float)
    return np.sqrt(2.0 / (g - 1.0) * (npr ** ((g - 1.0) / g) - 1.0))


def fully_expanded_diameter(npr, g: float = GAMMA):
    """D_j / D for a sonic nozzle of exit diameter D."""
    M = fully_expanded_mach(npr, g)
    area = (1.0 / M) * (2.0 / (g + 1.0) * (1.0 + 0.5 * (g - 1.0) * M ** 2)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    return np.sqrt(area)


def pack_spacing(npr, g: float = GAMMA):
    """Prandtl-Pack shock-cell spacing L_s / D."""
    M = fully_expanded_mach(npr, g)
    return 1.306 * np.sqrt(M ** 2 - 1.0) * fully_expanded_diameter(npr, g)
