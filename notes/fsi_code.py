import numpy as np
import matplotlib.pyplot as plt

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.tools.geometry import rotate


def set_total_mass(pa):
    # left limit of body i
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.total_mass[i] = np.sum(pa.m[fltr])
        assert pa.total_mass[i] > 0., "Total mass has to be greater than zero"


def set_center_of_mass(pa):
    pa.xcm[0] = np.sum(pa.m[:] * pa.x[:]) / np.sum(pa.m)
    pa.xcm[1] = np.sum(pa.m[:] * pa.y[:]) / np.sum(pa.m)


def set_moment_of_inertia_Izz(pa):
    Izz = 0.
    for i in range(len(pa.x)):
        Izz += pa.m[i] * ((pa.x[i] - pa.xcm[0])**2. + (pa.y[i] - pa.xcm[1])**2.)

    pa.Izz[0] = Izz


def test_invariance_of_Izz_with_body_rotation():
    nx, ny = 10, 10
    dx = 0.1
    x, y = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j]
    x = x.flat
    y = y.flat
    rho0 = 1000.
    m = np.ones_like(x) * dx * dx * rho0
    h = np.ones_like(x) * 1.3 * dx
    # radius of each sphere constituting in cube
    rad_s = np.ones_like(x) * dx
    body = get_particle_array(name='body', x=x, y=y, z=0, h=h, m=m, rad_s=rad_s)
    # plt.scatter(body.x, body.y)
    # plt.show()

    body.add_constant('Izz', 0.0)
    body.add_constant('xcm', np.array([0., 0., 0.]))

    center_of_mass(body)
    compute_moment_of_inertia_Izz(body)
    print(body.Izz)

    angle = 30
    x_n, y_n, z_n = rotate(body.x, body.y, body.z, [0., 0., 1.0], angle)
    body.x = x_n
    body.y = y_n
    body.z = z_n
    body.x[:] += 3.
    plt.scatter(body.x, body.y)
    plt.axes().set_aspect('equal', 'datalim')

    plt.show()

    set_center_of_mass(body)
    set_moment_of_inertia_Izz(body)
    print(body.Izz)


if __name__ == "__main__":
    test_invariance_of_Izz_with_body_rotation()
