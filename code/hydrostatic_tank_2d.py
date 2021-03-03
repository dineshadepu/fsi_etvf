"""A 2d hydrostatic tank.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

# from rigid_fluid_coupling import RigidFluidCouplingScheme
from geometry import hydrostatic_tank_2d

from pysph.examples.solid_mech.impact import add_properties
# from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
#                                                                create_fluid,
#                                                                create_sphere)
from pysph.tools.geometry import get_2d_block

from fluids import ETVFScheme
from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)


class HydrostaticTank2D(Application):
    def initialize(self):
        spacing = 0.05
        self.hdx = 1.3

        self.fluid_length = 1.0
        self.fluid_height = 1.0
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_height = 1.5
        self.tank_layers = 3
        self.tank_spacing = spacing

        self.body_height = 0.2
        self.body_length = 0.2
        self.body_density = 2000
        self.body_spacing = spacing / 2.
        self.body_h = self.hdx * self.body_spacing

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.vref = np.sqrt(2 * 9.81 * self.fluid_height)
        self.u_max = self.vref
        self.co = 10 * self.vref
        self.mach_no = self.vref / self.co
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gy = -9.81
        self.dim = 2

        # for boundary particles
        self.seval = None
        self.boundary_equations = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "tank"],
            boundaries=None)
        # print(self.boundary_equations)

    def create_particles(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(
            self.fluid_length, self.fluid_height, self.tank_height,
            self.tank_layers, self.fluid_spacing, self.fluid_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**2.

        fluid = get_particle_array(x=xf,
                                   y=yf,
                                   m=m_fluid,
                                   h=self.h,
                                   rho=self.fluid_density,
                                   name="fluid")

        tank = get_particle_array(x=xt,
                                  y=yt,
                                  m=m_fluid,
                                  m_fluid=m_fluid,
                                  h=self.h,
                                  rho=self.fluid_density,
                                  rad_s=self.tank_spacing/2.,
                                  name="tank")

        self.scheme.setup_properties([fluid, tank])

        return [fluid, tank]

    def create_scheme(self):
        etvf = ETVFScheme(fluids=['fluid'],
                          solids=['tank'],
                          dim=2,
                          rho0=self.fluid_density,
                          pb=self.p0,
                          c0=self.c0,
                          nu=0.01,
                          u_max=1. * self.u_max,
                          mach_no=self.mach_no,
                          gy=self.gy)
        s = SchemeChooser(default='etvf', etvf=etvf)
        return s

    def consume_user_options(self):
        self.options.scheme = 'etvf'
        super(HydrostaticTank2D, self).consume_user_options()

    def configure_scheme(self):
        dt = 0.125 * self.fluid_spacing * self.hdx / (self.co * 1.1)
        print("DT: %s" % dt)
        tf = 2.

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

    def _make_accel_eval(self, equations, pa_arrays):
        from pysph.base.kernels import (QuinticSpline)
        from pysph.tools.sph_evaluator import SPHEvaluator
        if self.seval is None:
            # kernel = self.options.kernel(dim=self.dim)
            kernel = QuinticSpline(dim=self.dim)
            seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                                 dim=self.dim, kernel=kernel)
            self.seval = seval
            return self.seval
        else:
            self.seval.update()
            return self.seval
        return seval

    def pre_step(self, solver):
        if solver.count % 1 == 0:
            t = solver.t
            dt = solver.dt

            arrays = self.particles
            a_eval = self._make_accel_eval(self.boundary_equations, arrays)

            # When
            a_eval.evaluate(t, dt)


if __name__ == '__main__':
    app = HydrostaticTank2D()
    app.run()
    # app.post_process(app.info_filename)
