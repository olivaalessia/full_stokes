"""Permittivity and absorption coefficient of dry snow and pure ice.

The rest comes from SMRT:

- pure ice: smrt.permittivity.ice.ice_permittivity_maetzler06(frequency, temperature), Ulaby Eqs. 4.23-4.25;
- Fresnel field coefficients: smrt.core.fresnel.fresnel_reflection_coefficients(eps_1, eps_2, mu1) -> rv, rh, mu2;
  reflectivities smrt.core.lib.abs2(rv), abs2(rh), Ulaby Eqs. 2.107-2.108;
- in a SMRT run, make_snowpack(thickness, "homogeneous", density=..., temperature=...) with
  make_model("nonscattering", ...) computes the same eps and ka internally.

SMRT convention: exp(-iwt), eps = eps' + 1j * eps'' with eps'' >= 0 (Ulaby writes eps' - j eps'').
Units: frequency in Hz, temperature in K, density in kg m-3.

Example::

    eps_snow = drysnow_permittivity(1.4e9, 240.0, 350.0)
    ka_snow = power_absorption_coefficient(1.4e9, eps_snow)  # penetration depth 1 / ka_snow (m)
"""

import numpy as np
from smrt.core.globalconstants import C_SPEED, DENSITY_OF_ICE, PERMITTIVITY_OF_AIR
from smrt.permittivity.generic_mixing_formula import polder_van_santen
from smrt.permittivity.ice import ice_permittivity_maetzler06


def drysnow_permittivity(
    frequency,
    temperature,
    density,
    effective_permittivity_model=polder_van_santen,
    ice_permittivity_model=ice_permittivity_maetzler06,
):
    """Return the effective permittivity of dry snow, a mixture of air and ice grains.

    The default polder_van_santen (spheres, Ulaby Eq. 4.36b with eps* = eps_m) is the effective permittivity model of
    SMRT's iba and nonscattering models; maxwell_garnett_for_spheres gives the Tinga-Voss-Blossey Eq. 4.53.

    Args:
        frequency: frequency (Hz).
        temperature: temperature (K), at most 273.15.
        density: snow density (kg m-3), from 0 to DENSITY_OF_ICE.
        effective_permittivity_model: SMRT mixing formula with arguments (frac_volume, e0, eps).
        ice_permittivity_model: SMRT ice permittivity function with arguments (frequency, temperature).
    """
    frac_volume = np.asarray(density) / DENSITY_OF_ICE  # Ulaby Eq. 4.52
    if np.any((frac_volume < 0) | (frac_volume > 1)):
        raise ValueError(f"density must be between 0 and {DENSITY_OF_ICE} kg m-3")

    eps_ice = ice_permittivity_model(np.asarray(frequency), np.asarray(temperature))
    return effective_permittivity_model(frac_volume, PERMITTIVITY_OF_AIR, eps_ice)


def power_absorption_coefficient(frequency, eps):
    """Return the power absorption coefficient ka (m-1) of a non-scattering medium, as SMRT's nonscattering model.

    Ulaby Eqs. 4.9a and 4.11; the penetration depth is 1/ka (Eq. 4.12a).
    """
    if np.any(np.imag(eps) < -1e-10):  # same tolerance as SMRT's own check (rounding gives 1 - 1e-19j for air)
        raise ValueError("Im(eps) must be >= 0 (SMRT convention): write Ulaby's eps' - j eps'' as eps' + 1j * eps''")

    k0 = 2 * np.pi * np.asarray(frequency) / C_SPEED  # free-space wavenumber (rad m-1)
    return 2 * k0 * np.sqrt(eps).imag  # ka = 2 alpha, alpha = k0 Im(sqrt(eps))
