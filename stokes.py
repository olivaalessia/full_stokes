"""Stokes brightness temperatures [TbV, TbH, Tb3, Tb4] of the upwelling emission and change of polarization basis.

Conventions, shared with SMRT and the passive chapters of Ulaby & Long (2014):

- Basis of the wave travelling along k, from the surface to the radiometer: h perpendicular and v parallel to the plane
  of incidence (Ulaby Fig. 2-16), oriented as h = z x k / |z x k| and v = h x k. (v, h, k) is right-handed.
- Tb = K [<|Ev|^2>, <|Eh|^2>, 2 Re<Ev Eh*>, 2 Im<Ev Eh*>] (Ulaby Eq. 6.116). Tb3 is the "U" of SMRT (npol=3).
- Time dependence exp(-iwt), as in SMRT (permittivity with positive imaginary part). With this basis, Tb3 = TbP - TbM
  and Tb4 = TbL - TbR (Ulaby Eqs. 6.115 and 6.117). Conjugate amplitudes from exp(+jwt) codes before use.
- Polarization axis first, as in SMRT: Stokes vectors (4, ...), matrices (4, 4, ...).
"""

import numpy as np

from smrt.core.vector3 import vector3
from smrt.emmodel.common import Lmatrix


def polarization_basis(theta, phi):
    """Return the unit vectors k, v, h of the upwelling wave.

    Ulaby Eq. 10.28 (used in Sec. 12-4) builds h from the antenna look direction -k: same v, opposite h, hence opposite
    Tb3 and Tb4.

    Args:
        theta: angle between k and the upward vertical, in radians (SMRT sensor.theta).
        phi: azimuth of k, in radians (look azimuth + pi).

    Returns:
        k, v, h as vector3.
    """
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)

    k = vector3.from_angles(1, cos_t, phi)  # from the surface to the radiometer
    h = vector3.from_xyz(-sin_p, cos_p, 0)  # z x k / |z x k|: horizontal, also defined at nadir
    v = vector3.from_xyz(cos_t * cos_p, cos_t * sin_p, -sin_t)  # h x k: in the plane of incidence
    return k, v, h


def rotation_angle(x, v, h):
    """Return the angle psi (radians) from v towards h of the antenna polarization vector x.

    Only the projection of x on the (v, h) plane is used.

    Args:
        x: antenna polarization vector (vector3), equal to v when psi = 0.
        v, h: vectors from :py:func:`polarization_basis`, in the same Cartesian frame as x.
    """
    return np.arctan2(x.dot(h), x.dot(v))


def rotate_stokes(tb, psi):
    """Return the Stokes vector tb in the basis x, y rotated by psi from v towards h.

    x = cos(psi) v + sin(psi) h and y = -sin(psi) v + cos(psi) h, so (x, y, k) stays right-handed. The TbV, TbH, Tb3
    block is SMRT's Lmatrix (Matzler 2006, Eq. 3.20; Ulaby Eq. 12.26 when Tb3 = 0) and Tb4 is unchanged. Use -psi for
    the inverse rotation. For an antenna basis with (x, y, k) left-handed (y -> -y), change the sign of Tb3 and Tb4.

    Args:
        tb: [TbV, TbH, Tb3, Tb4], shape (4, ...).
        psi: rotation angle in radians, broadcastable with tb[0].

    Returns:
        [Tbx, Tby, Tb3, Tb4] in the rotated basis, shape (4, ...).
    """
    L = np.zeros((4, 4) + np.shape(psi))  # L array of zeros
    L[:3, :3] = Lmatrix(np.cos(psi), np.sign(np.sin(psi)), (3, 3))
    L[3, 3] = 1  
    return np.einsum("ij...,j...->i...", L, tb)


def stokes_from_result(result, **kwargs):
    """Return [TbV, TbH, Tb3, Tb4], shape (4, ...), from a SMRT PassiveResult.

    Args:
        result: PassiveResult returned by Model.run.
        **kwargs: selection passed to result.TbV and result.TbH (e.g. frequency=37e9).
    """
    tbv = np.asarray(result.TbV(**kwargs))
    tbh = np.asarray(result.TbH(**kwargs))
    return np.stack([tbv, tbh, np.zeros_like(tbv), np.zeros_like(tbv)])