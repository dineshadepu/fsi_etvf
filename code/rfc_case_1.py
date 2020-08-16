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

from rigid_fluid_coupling import RigidFluidCouplingScheme
from geometry import hydrostatic_tank_2d

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary, create_fluid, create_sphere)


class RigidFluidCoupling(Application):
    def initialize(self):
        self.fluid_length = 1.0
        self.fluid_height = 1.0
        self.fluid_density = 1000.0
        self.fluid_spacing = 0.1

        self.tank_height = 1.5
        self.tank_layers = 3
        self.tank_spacing = 0.1

        self.hdx = 1.3
        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gy = -9.81

    def create_particles(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(self.fluid_length,
                                             self.fluid_height,
                                             self.tank_height, self.tank_layers,
                                             self.fluid_spacing,
                                             self.fluid_spacing)

        m = self.fluid_density * self.fluid_spacing**2.

        fluid = get_particle_array(x=xf, y=yf, m=m, h=self.h,
                                   rho=self.fluid_density, name="fluid")

        tank = get_particle_array(x=xt, y=yt, m=m, m_fluid=m, h=self.h,
                                  rho=self.fluid_density, name="tank")

        self.scheme.setup_properties([fluid, tank])

        return [fluid, tank]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=None, fluids=['fluid'],
                                       boundaries=['tank'], dim=2,
                                       rho0=self.fluid_density, p0=self.p0,
                                       c0=self.c0, gy=self.gy)
        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        dt = 0.125 * self.fluid_spacing * self.hdx / (self.co * 1.1) / 2.
        print("DT: %s" % dt)
        tf = 0.5

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    # app.post_process(app.info_filename)
