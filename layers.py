"""Thermal emission of flat horizontal layers without volume scattering, with its weighting functions.

Same physics as SMRT's multifresnel_thermalemission (MFTE): incoherent absorbing layers with flat interfaces (Ulaby
Sect. 12-12.2). The layers are added from the bottom up, i.e. Ulaby Eq. 12.51 (a = 0) applied layer by layer. The
emission is linear in the temperatures:

    Tb = sum_i W_i T_i + Gamma Tb_down,    sum_i W_i = 1 - Gamma = e  (Kirchhoff, Ulaby Eqs. 6.96-6.97),

where W_i, the weight of layer i (discrete form of Ulaby Eq. 6.89, with the reflections at the interfaces), tells
where the emission comes from. For TbV and TbH seen from air MFTE gives the same numbers; this module adds what MFTE
does not return: the weights, the Stokes reflection matrix of the stack, and the view from inside a top medium at any
angle (eps_above). The emission of the stack has Tb3 = Tb4 = 0 (azimuthal symmetry); the reflection matrix keeps the
phase of r_v r_h*, which acts on polarized radiation coming from above (e.g. reflected by a tilted facet).

Example, 1 m of snow over ice seen from air at 40 degrees under a 5 K sky::

    eps = drysnow_permittivity(1.4e9, [240.0, 250.0], [374.0, 916.7])
    s = stack_stokes(1.4e9, np.cos(np.radians(40)), eps, [240.0, 250.0], [1.0, np.inf], tbdown=5.0)
    s.tb[:, 0]  # [TbV, TbH, 0, 0]
    s.weights[0, 0]  # contribution of each layer to TbV per kelvin; sums to 1 - Gamma_v
"""

from typing import NamedTuple

import numpy as np

from smrt.core.fresnel import fresnel_reflection_coefficients, snell_angle
from smrt.core.lib import abs2

from smrt_fullstokes.utils.permittivity import power_absorption_coefficient


class StackStokes(NamedTuple):
    tb: np.ndarray  # (4, n_mu): upwelling [TbV, TbH, Tb3, Tb4] in the medium above, reflected tbdown included
    reflection: np.ndarray  # (4, 4, n_mu): Stokes reflection matrix; [0, 0] and [1, 1] are Gamma_v and Gamma_h
    weights: np.ndarray  # (2, n_mu, n_layers): W_i = dTbV/dT_i and dTbH/dT_i


def stack_stokes(frequency, mu, eps, temperature, thickness, tbdown=0.0, eps_above=1.0):
    """Return the Stokes vector, the reflection matrix and the weights of a stack of flat layers.

    Args:
        frequency: frequency (Hz).
        mu: cosines of the zenith angle in the medium above the stack.
        eps, temperature, thickness: permittivity, temperature (K) and thickness (m) of each layer, from the top. The
            last layer can be semi-infinite (np.inf) if it absorbs; below a finite last layer there is nothing (no
            emission, no reflection), as in MFTE without substrate.
        tbdown: unpolarized brightness temperature coming down in the medium above (K), e.g. an isotropic sky.
        eps_above: permittivity of the medium above: 1 for air, or that of a top layer (e.g. the sastrugi) to see the
            stack from inside it, also beyond the critical angle of its interface with air.
    """
    mu, eps = np.atleast_1d(mu), np.atleast_1d(eps)
    temperature, thickness = np.atleast_1d(temperature), np.atleast_1d(thickness)

    mu_layer = snell_angle(eps_above, eps[:, None], mu)  # cosine in each layer (n_layers, n_mu), Ulaby Eq. 6.79
    mu_1 = np.vstack([mu, mu_layer[:-1]])  # cosine above each interface
    eps_1 = np.append(eps_above, eps[:-1])[:, None]  # permittivity above each interface
    rv, rh, _ = fresnel_reflection_coefficients(eps_1, eps[:, None], mu_1)
    gamma = np.array([abs2(rv), abs2(rh)])  # reflectivity of each interface, Ulaby Eqs. 2.107-2.108, 6.88
    c = rv * np.conj(rh)  # a reflection multiplies Tb3 + i Tb4 by c

    ka = power_absorption_coefficient(frequency, eps)
    if np.isinf(thickness[-1]) and ka[-1] == 0:
        raise ValueError("a semi-infinite last layer must absorb: Im(eps) > 0")
    with np.errstate(divide="ignore", invalid="ignore"):  # mu_layer = 0 beyond a critical angle: no transmission
        upsilon = np.where(mu_layer > 0, np.exp(-(ka * thickness)[:, None] / mu_layer), 0)  # Ulaby Eq. 12.70, ks = 0

    gamma_s, c_s = np.zeros((2, len(mu))), np.zeros(len(mu), dtype=complex)  # nothing below the last layer
    source, F = np.empty_like(gamma), np.empty_like(gamma)
    for i in reversed(range(len(eps))):  # adding, from the bottom up
        source[:, i] = (1 - upsilon[i]) * (1 + upsilon[i] * gamma_s)  # emission of layer i that reaches its top
        below, below_c = upsilon[i] ** 2 * gamma_s, upsilon[i] ** 2 * c_s  # stack below, seen under interface i
        F[:, i] = (1 - gamma[:, i]) / (1 - gamma[:, i] * below)  # through interface i (Ulaby Eq. 6.86), with the
        gamma_s = gamma[:, i] + (1 - gamma[:, i]) * F[:, i] * below  # multiple reflections under it
        c_s = c[i] + (1 - gamma[0, i]) * (1 - gamma[1, i]) * below_c / (1 - c[i] * below_c)

    above = np.cumprod(np.concatenate([np.ones_like(F[:, :1]), (upsilon * F)[:, :-1]], axis=1), axis=1)
    weights = np.moveaxis(source * F * above, 1, 2)  # emission of layer i transmitted up to the medium above

    tb = np.zeros((4, len(mu)))
    tb[:2] = weights @ temperature + gamma_s * tbdown
    reflection = np.zeros((4, 4, len(mu)))
    reflection[0, 0], reflection[1, 1] = gamma_s
    reflection[2, 2] = reflection[3, 3] = c_s.real
    reflection[3, 2], reflection[2, 3] = c_s.imag, -c_s.imag
    return StackStokes(tb, reflection, weights)


def firn_profile(depth, density, temperature, dz_top=0.01, growth=1.05, dz_max=5.0):
    """Return the thickness, density and temperature of the layers that discretize a profile down to depth (m).

    The thickness grows from dz_top by the factor growth per layer, up to dz_max: thin layers near the surface, where
    the emission at 37 GHz comes from, thick layers at depth, reached only at L-band. density(z) and temperature(z) are
    functions of the depth z (m), evaluated at the middle of each layer.
    """
    bottom = [0.0]
    while bottom[-1] < depth:
        bottom.append(bottom[-1] + min(dz_top * growth ** (len(bottom) - 1), dz_max))
    bottom = np.minimum(bottom, depth)  # the last layer ends at depth
    middle = (bottom[1:] + bottom[:-1]) / 2
    ones = np.ones_like(middle)  # constant functions (lambda z: 350.0) give arrays too
    return np.diff(bottom), density(middle) * ones, temperature(middle) * ones
