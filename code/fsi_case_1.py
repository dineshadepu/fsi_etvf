"""Dam break solved with DTSPH.

python dam_break_2d.py --openmp --integrator gtvf --no-internal-flow --pst sun2019 --no-set-solid-vel-project-uhat --no-set-uhat-solid-vel-to-u --no-vol-evol-solid --no-edac-solid --surf-p-zero -d dam_break_2d_etvf_integrator_gtvf_pst_sun2019_output --pfreq 1 --detailed-output


"""
import numpy as np

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.base.utils import get_particle_array

from pysph.examples import cavity as LDC
from pysph.sph.equation import Equation, Group
# from fluids import ETVFScheme
# from solid_mech import ComputeAuHatETVF

from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import SchemeChooser
from pysph.tools.geometry import rotate

# from boundary_particles import (add_boundary_identification_properties,
#                                 get_boundary_identification_etvf_equations)
from fluid_structure_interaction import FSIScheme


def get_fixed_beam(beam_length, beam_height, beam_inside_length,
                   boundary_layers, spacing):
    """
 |||=============
 |||=============
 |||===================================================================|
 |||===================================================================|Beam height
 |||===================================================================|
 |||=============
 |||=============
   <------------><---------------------------------------------------->
      Beam inside                   Beam length
      length
    """
    import matplotlib.pyplot as plt
    # create a block first
    xb, yb = get_2d_block(dx=spacing,
                          length=beam_length + beam_inside_length,
                          height=beam_height)

    # create a (support) block with required number of layers
    xs1, ys1 = get_2d_block(dx=spacing,
                            length=beam_inside_length,
                            height=boundary_layers * spacing)
    xs1 += np.min(xb) - np.min(xs1)
    ys1 += np.min(yb) - np.max(ys1) - spacing

    # create a (support) block with required number of layers
    xs2, ys2 = get_2d_block(dx=spacing,
                            length=beam_inside_length,
                            height=boundary_layers * spacing)
    xs2 += np.min(xb) - np.min(xs2)
    ys2 += np.max(ys2) - np.min(yb) + spacing

    xs = np.concatenate([xs1, xs2])
    ys = np.concatenate([ys1, ys2])

    xs3, ys3 = get_2d_block(dx=spacing,
                            length=boundary_layers * spacing,
                            height=np.max(ys) - np.min(ys))
    xs3 += np.min(xb) - np.max(xs3) - 1. * spacing
    # ys3 += np.max(ys2) - np.min(yb) + spacing

    xs = np.concatenate([xs, xs3])
    ys = np.concatenate([ys, ys3])
    # plt.scatter(xs, ys)
    # plt.scatter(xb, yb)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.show()

    return xb, yb, xs, ys


class Dambreak2DGate(Application):
    def initialize(self):
        self.dx = 0.001
        self.fluid_column_height = 0.05
        self.fluid_column_width = 0.05
        self.fluid_rho = 1000.0

        self.container_height = 0.2
        self.container_width = 0.5
        self.nboundary_layers = 4

        self.gate_height = 0.079
        self.gate_width = 0.005
        self.gate_rho = 1100.
        self.gate_K = 2. * 1e7
        self.gate_G = 4.27 * 1e6
        self.gate_E = (9. * self.gate_K * self.gate_G) / (3. * self.gate_K +
                                                          self.gate_G)
        self.gate_nu = self.gate_E / (2. * self.gate_G) - 1.
        self.gate_spacing = self.dx

        self.gy = 0.0
        self.hdx = 1.3
        self.h = self.hdx * self.dx
        self.vref = 1.0
        self.c0 = 10.0 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.tf = 0.05
        self.p0 = self.fluid_rho * self.c0**2

        self.seval = None
        self.dim = 2
        # self.boundary_equations = get_boundary_identification_etvf_equations(
        #     destinations=["fluid"],
        #     sources=["fluid", "boundary"],
        #     boundaries=['boundary'])
        # print(self.boundary_equations)

    def create_particles(self):
        # tank
        xt, yt = get_2d_tank(dx=self.dx,
                             length=self.container_width,
                             height=self.container_height,
                             num_layers=4)

        # block
        xf, yf = get_2d_block(dx=self.dx,
                              length=self.fluid_column_width,
                              height=self.fluid_column_height,
                              center=[-1.5, 1])

        # set the geometry
        # get the minimum of the tank, fluid and gate
        x_tank_min = np.min(xt)
        y_tank_min = np.min(yt)
        x_fluid_min = np.min(xf)
        y_fluid_min = np.min(yf)

        xf += ((x_tank_min + (self.nboundary_layers) * self.dx) - x_fluid_min)
        yf += ((y_tank_min + (self.nboundary_layers) * self.dx) - y_fluid_min)

        h = self.hdx * self.dx
        m = self.dx**2 * self.fluid_rho

        fluid = get_particle_array(name='fluid',
                                   x=xf,
                                   y=yf,
                                   h=h,
                                   m=m,
                                   rho=self.fluid_rho)
        fluid.u[:] = 0.1
        fluid.x[:] += 4. * self.dx

        tank = get_particle_array(name='tank',
                                  x=xt,
                                  y=yt,
                                  h=self.h,
                                  m=m,
                                  rho=self.fluid_rho)

        # fixed gate
        xg, yg, xg_support, yg_support = get_fixed_beam(
            self.gate_height, self.gate_width, self.gate_height / 4., 3,
            self.gate_spacing)
        # rotate them 90 degrees
        xg, yg, _tmp = rotate(xg, yg, np.zeros(len(xg)), angle=-90.)
        xg_support, yg_support, _tmp = rotate(xg_support, yg_support,
                                              np.zeros(len(xg)), angle=-90.)

        # set the gate coordinates
        x_gate_min = np.min(xg)
        y_gate_min = np.min(yg)
        x_fluid_max = np.max(xf)
        y_fluid_min = np.min(yf)

        x_scale = x_fluid_max - x_gate_min + (self.nboundary_layers + 1) * self.dx
        y_scale = y_fluid_min - y_gate_min
        xg += x_scale
        yg += y_scale

        xg_support += x_scale
        yg_support += y_scale

        # uncommnet this
        # xg[:] += 0.01
        # yg[:] += 0.01
        m = self.gate_rho * self.gate_spacing**2.
        gate = get_particle_array(name='gate',
                                  x=xg,
                                  y=yg,
                                  h=self.h,
                                  m=m,
                                  rho=self.gate_rho,
                                  constants={
                                      'E': self.gate_E,
                                      'n': 4,
                                      'nu': self.gate_nu,
                                      'spacing0': self.gate_spacing,
                                      'rho_ref': self.gate_rho,
                                  })

        # for i in range(len(gate.x)):
        #     if gate.y[i] < 0.08:
        #         gate.u[i] = 0.08 - gate.y[i]

        gate_support = get_particle_array(name='gate_support',
                                          x=xg_support,
                                          y=yg_support,
                                          h=self.h,
                                          m=m,
                                          rho=self.gate_rho,
                                          constants={
                                              'E': self.gate_E,
                                              'n': 4,
                                              'nu': self.gate_nu,
                                              'spacing0': self.gate_spacing,
                                              'rho_ref': self.gate_rho,
                                          })

        self.scheme.setup_properties([fluid, gate, gate_support])

        # comment these
        # fluid.x -= 0.5
        return [fluid, gate, gate_support]

    def configure_scheme(self):
        dt = 0.125 * self.h / self.vref
        print(dt)
        dt = 2.5 * 1e-6
        # dt = 2.5 * 1e-5
        # dt = 2.5 * 1e-5
        print("dt = %f" % dt)
        self.scheme.configure_solver(
            dt=dt,
            tf=self.tf,
            adaptive_timestep=False,
            pfreq=100,
        )

    def create_scheme(self):
        fsi = FSIScheme(fluids=['fluid'],
                        solids=['gate'],
                        boundaries=[],
                        solid_supports=['gate_support'],
                        dim=self.dim,
                        rho0=self.fluid_rho,
                        c0=self.c0,
                        p0=self.p0,
                        h0=self.h,
                        hdx=self.hdx,
                        gy=self.gy)

        s = SchemeChooser(default='fsi', fsi=fsi)
        return s

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')

    # def _make_accel_eval(self, equations, pa_arrays):
    #     if self.seval is None:
    #         kernel = QuinticSpline(dim=self.dim)
    #         seval = SPHEvaluator(arrays=pa_arrays,
    #                              equations=equations,
    #                              dim=self.dim,
    #                              kernel=kernel)
    #         self.seval = seval
    #         return self.seval
    #     else:
    #         self.seval.update()
    #         return self.seval
    #     return seval

    # def pre_step(self, solver):
    #     if solver.count % 1 == 0:
    #         t = solver.t
    #         dt = solver.dt

    #         arrays = self.particles
    #         a_eval = self._make_accel_eval(self.boundary_equations, arrays)

    #         # When
    #         a_eval.evaluate(t, dt)


if __name__ == '__main__':
    app = Dambreak2DGate()
    app.run()
    # app.post_process(app.info_filename)
    # get_fixed_beam(0.1, 0.03, 0.1, 3, 0.001)
