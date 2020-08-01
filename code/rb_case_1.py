"""A cube translating and rotating freely without the influence of gravity.
This is used to test the rigid body dynamics equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from rigid_body_2d import RigidBody2DScheme

from pysph.examples.solid_mech.impact import add_properties


class Case0(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 3

        self.dt = 1e-3
        self.tf = 1.18 * 1e-2

    def create_scheme(self):
        rb2d = RigidBody2DScheme(rigid_bodies=['body'], boundaries=None, dim=2)
        s = SchemeChooser(default='rb2d', rb2d=rb2d)
        return s

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=500)

    def create_particles(self):
        nx, ny, nz = 10, 10, 10
        dx = self.dx
        x, y, z = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j, 0:1:nz * 1j]
        x = x.flat
        y = y.flat
        z = (z - 1).flat
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array(name='body', x=x, y=y, z=z, h=h,
                                  m=m, rad_s=rad_s)
        body_id = np.zeros(len(x), dtype=int)
        body.add_property('body_id', type='int', data=body_id)
        print(body.body_id)

        # setup the properties
        self.scheme.setup_properties([body])

        # print(body.properties)
        # print(body.constants)
        # print(body.xcm0)
        # print(body.vcm0)
        # print(body.R)
        # print(body.orientation_angle)
        # print(body.omega)

        body.vcm[0] = 0.5
        body.vcm[1] = 0.5
        print(body.omega)
        body.omega[2] = 1.

        return [body]


if __name__ == '__main__':
    app = Case0()
    app.run()
