"""
Flow past cylinder
"""
import logging
from time import time
import os
import numpy as np
from numpy import pi, cos, sin, exp

from pysph.base.kernels import QuinticSpline
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from pysph.tools import geometry as G
from pysph.sph.scheme import add_bool_argument
from pysph.tools.geometry import (get_2d_tank, get_2d_block,
                                  remove_overlap_particles)
from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from pysph.examples.solid_mech.impact import add_properties
from fsi_substepping import FSISubSteppingScheme
from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)
from nrbc import (EvaluateCharacterisctics, EvaluateNumberDensity,
                  EvaluatePropertyfromCharacteristics,
                  ShepardInterpolateCharacteristics)

logger = logging.getLogger()

# Fluid mechanical/numerical parameters
u_freestream = 0.0
rho = 1000
umax = 1.0
c0 = 10 * umax
p0 = rho * c0 * c0


def create_circle_1(diameter=0.1, spacing=0.005, center=None):
    dx = spacing
    x = [0.0]
    y = [0.0]
    r = spacing
    nt = 0
    radius = diameter / 2.

    tmp_dist = radius - spacing/2.
    i = 0
    while tmp_dist > spacing/2.:
        perimeter = 2. * np.pi * tmp_dist
        no_of_points = int(perimeter / spacing) + 1
        theta = np.linspace(0., 2. * np.pi, no_of_points)
        for t in theta[:-1]:
            x.append(tmp_dist * np.cos(t))
            y.append(tmp_dist * np.sin(t))
        i = i + 1
        tmp_dist = radius - spacing/2. - i * spacing

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


def get_fixed_beam_no_clamp(beam_length, beam_height, cylinder_radius,
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
    # create a block first
    xb, yb = get_2d_block(dx=spacing,
                          length=beam_length,
                          height=beam_height)

    radius = cylinder_radius
    xs, ys = get_2d_block(dx=spacing,
                          length=2. * radius,
                          height=2. * radius)

    indices = []
    for i in range(len(xs)):
        if xs[i]**2. + ys[i]**2. > (radius - spacing/2)**2.:
            indices.append(i)

    xs = np.delete(xs, indices)
    ys = np.delete(ys, indices)

    # move the beam to the front of the wall
    xb += max(xb) - min(xs) + spacing

    return xb, yb, xs, ys


class FlappingWing(Application):
    def initialize(self):
        # Geometric parameters
        self.Lt = 0.8  # length of tunnel
        self.Wt = 0.41  # half width of tunnel
        self.dc = 1.2  # diameter of cylinder
        self.cxy = 20., 0.0  # center of cylinder
        self.nl = 10  # Number of layers for wall/inlet/outlet
        self.sol_adapt = 0.0
        self._nnps = None

        # parameters regarding the wing
        self.wing_length = 0.35
        self.wing_height = 0.02
        self.wing_rho0 = 1000
        self.wing_E = 1.4 * 1e6
        self.wing_nu = 0.4
        self.gy = -2.

        self.dim = 2

    def add_user_options(self, group):
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=200,
            help="Reynolds number."
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.0,
            help="Ratio h/dx."
        )

        group.add_argument(
            "--dc", action="store", type=float, dest="dc", default=2.0,
            help="Diameter of the cylinder."
        )

        group.add_argument(
            "--wing-rho", action="store", type=float, dest="wing_rho",
            default=1000.0, help="Density of wing"
        )

    def consume_user_options(self):
        self.wing_rho0 = self.options.wing_rho

        if self.options.n_damp is None:
            self.options.n_damp = 20
        re = self.options.re

        self.nu = nu = umax * self.dc / re

        spacing = self.wing_height / 4.
        self.dx = spacing
        self.dx_plate = spacing

        self.hdx = hdx = self.options.hdx
        self.nl = (int)(10.0*hdx)

        self.h = hdx * self.dx

        self.tf = 1.

        self.fluid_length = self.Lt
        self.fluid_height = self.Wt
        self.fluid_density = 1000.0
        self.fluid_spacing = self.dx
        self.h_fluid = self.hdx * self.fluid_spacing
        self.vref_fluid = np.sqrt(2 * 9.81 * self.fluid_height)
        self.u_max_fluid = self.vref_fluid
        self.c0_fluid = 10 * self.vref_fluid
        self.mach_no_fluid = self.vref_fluid / self.c0_fluid
        self.p0_fluid = self.fluid_density * self.c0_fluid**2.
        self.alpha = 0.1
        self.gy = -0.0

        self.fluid_spacing = self.dx
        self.dx_wing = self.dx
        self.wall_layers = 2

        self.c0_wing = get_speed_of_sound(self.wing_E, self.wing_nu,
                                          self.wing_rho0)
        self.u_max_wing = 2.
        self.mach_no_wing = self.u_max_wing / self.c0_wing

        # for boundary particles
        self.seval = None
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "wing", "wing",
                                             "wing_support"],
            boundaries=["wing", "wing", "wing_support"])

        # boundary equations
        # self.boundary_equations = get_boundary_identification_etvf_equations(
        #     destinations=["wing"], sources=["wing"])
        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["wing"], sources=["wing", "wing_support"],
            boundaries=["wing_support"])

        self.boundary_equations = self.boundary_equations_1 + self.boundary_equations_2

        self.dt_fluid = 0.25 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        self.dt_solid = 0.25 * self.h_fluid / (
            (self.wing_E / self.wing_rho0)**0.5 + self.u_max_wing)

    def _create_box(self):
        wing, wing_support = self._create_flapping_structure()

        dx = self.dx
        m = rho * dx * dx
        h0 = self.hdx*dx
        layers = self.nl * dx
        w = self.Wt
        l = self.Lt
        x, y = np.mgrid[-layers:l+layers:dx, -w-layers:w+layers:dx]
        x = np.ravel(x)
        y = np.ravel(y)

        # First form walls, then inlet and outlet, and finally fluid.
        fluid_cond = (y > w - dx/2) | (y < -w + dx/2) | (x > l - 3. * dx) | (x < -l + 3. * dx)

        # another way
        indices = []

        for i in range(len(x)):
            if x[i] < - 3. * dx or x[i] > l - 3. * dx:
                indices.append(i)
            elif y[i] < -w + 3. * dx or y[i] > w - 3. * dx:
                indices.append(i)

        xw, yw = x[indices], y[indices]
        xf, yf = np.delete(x, indices), np.delete(y, indices)

        scale = min(xf) - min(wing_support.x)
        wing_support.x[:] += scale + 0.2
        # xw = np.concatenate([xw, wing_support.x])
        # yw = np.concatenate([yw, wing_support.y])

        wall = get_particle_array(
            name='wall', x=xw, y=yw, m=m, h=h0, rho=rho
        )

        fluid = get_particle_array(
            name='fluid', x=xf, y=yf, m=m, h=h0, u=u_freestream, rho=rho,
            p=0.0, vmag=0.0
        )
        remove_overlap_particles(fluid, wall, dx)
        return fluid, wall

    def _create_flapping_structure(self):
        xp, yp, xw, yw = get_fixed_beam_no_clamp(self.wing_length, self.wing_height, 0.05,
                                                 self.wall_layers +1,
                                                 self.dx_wing)
        m = self.wing_rho0 * self.dx_plate**2.
        wing = get_particle_array(x=xp,
                                  y=yp,
                                  m=m,
                                  h=self.h,
                                  rho=self.wing_rho0,
                                  name="wing",
                                  constants={
                                      'E': self.wing_E,
                                      'n': 4.,
                                      'nu': self.wing_nu,
                                      'spacing0': self.dx_wing,
                                      'rho_ref': self.wing_rho0
                                  })

        # make sure that the beam intersection with wall starts at the 0.
        wing_support = get_particle_array(x=xw,
                                          y=yw,
                                          m=m,
                                          h=self.h,
                                          rho=self.wing_rho0,
                                          name="wing_support",
                                          constants={
                                              'E': self.wing_E,
                                              'n': 4.,
                                              'nu': self.wing_nu,
                                              'spacing0': self.dx_wing,
                                              'rho_ref': self.wing_rho0
                                          })
        return wing, wing_support

    def create_particles(self):
        fluid, wall = self._create_box()
        wing, wing_support = self._create_flapping_structure()

        particles = [fluid, wall, wing, wing_support]

        # move the solid in middle
        scale = min(fluid.x) - min(wing_support.x)
        wing.x[:] += scale + 0.2
        wing_support.x[:] += scale + 0.2

        remove_overlap_particles(fluid, wing, self.dx)
        remove_overlap_particles(fluid, wing_support, self.dx)

        self.scheme.setup_properties(particles)

        wing.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        wing.rho_fsi[:] = self.fluid_density

        wing_support.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        wing_support.rho_fsi[:] = self.fluid_density

        DEFAULT_PROPS = [
            'xn', 'yn', 'zn', 'J2v', 'J3v', 'J2u', 'J3u', 'J1', 'wij2', 'disp',
            'ioid', 'wij', 'nrbc_p_ref'
        ]
        for prop in DEFAULT_PROPS:
            if prop not in wall.properties:
                wall.add_property(prop)

            if prop not in fluid.properties:
                fluid.add_property(prop)

        wall.add_constant('uref', 0.0)
        consts = [
            'avg_j2u', 'avg_j3u', 'avg_j2v', 'avg_j3v', 'avg_j1', 'uref'
        ]
        for const in consts:
            if const not in wall.constants:
                wall.add_constant(const, 0.0)

        # set the normals
        indices = wall.x > max(fluid.x)
        wall.xn[indices] = 1
        wall.yn[indices] = 0.

        indices = wall.x < min(fluid.x)
        wall.xn[indices] = -1
        wall.yn[indices] = 0.

        indices = wall.y < min(fluid.y)
        wall.xn[indices] = 0
        wall.yn[indices] = -1

        indices = wall.y > max(fluid.y)
        wall.xn[indices] = 0
        wall.yn[indices] = 1
        return particles

    def create_scheme(self):
        nu = None
        substep = FSISubSteppingScheme(fluids=['fluid'],
                                       solids=['wall'],
                                       structures=['wing'],
                                       structure_solids=['wing_support'],
                                       dt_fluid=1.,
                                       dt_solid=1.,
                                       dim=2,
                                       h_fluid=0.,
                                       rho0_fluid=0.,
                                       pb_fluid=0.,
                                       c0_fluid=0.,
                                       nu_fluid=0.,
                                       mach_no_fluid=0.,
                                       mach_no_structure=0.,
                                       gy=0.)

        s = SchemeChooser(default='substep', substep=substep)

        return s

    def create_equations(self):
        from pysph.sph.equation import Group
        from edac import EvaluateNumberDensity
        from pysph.sph.bc.inlet_outlet_manager import (
            UpdateNormalsAndDisplacements
        )

        equations = self.scheme.get_equations()

        equations[-1].equations[-2].equations[0].gy = -2
        return equations

        equations.pop(3)

        equations[-1].equations[-2].equations[0].gy = -2

        eq = []
        eq.append(
            Group(equations=[
                EvaluateCharacterisctics(
                    dest='fluid', sources=None, c_ref=c0, rho_ref=rho,
                    u_ref=u_freestream, v_ref=0.0, p_ref=0.0
                )
            ])
        )
        eq.append(
            Group(equations=[
                EvaluateNumberDensity(dest='wall', sources=['fluid']),
                ShepardInterpolateCharacteristics(dest='wall', sources=['fluid']),
            ])
        )
        eq.append(Group(equations=[
            EvaluatePropertyfromCharacteristics(
                dest='wall', sources=None, c_ref=c0, rho_ref=rho,
                u_ref=u_freestream, v_ref=0.0, p_ref=0.0
            ),
            ])
        )
        for eqn in eq[::-1]:
            equations.insert(1, eqn)

        # equation.sources = ["tank", "fluid", "gate", "gate_support"]

        # elif self.options.scheme == 'gtvf':
        #     equation = eqns.groups[-1][4].equations[3]
        #     # print(equation)
        #     equation.sources = ["tank", "fluid", "gate", "gate_support"]
        return equations

    def configure_scheme(self):
        dt = 0.25 * self.h_fluid / (
            (self.wing_E / self.wing_rho0)**0.5 + self.u_max_wing)

        print("DT: %s" % dt)
        tf = 0.2

        self.scheme.configure_solver(dt=self.dt_fluid, tf=tf)
        self.scheme.configure(
            dim=2,
            h_fluid=self.h_fluid,
            rho0_fluid=self.fluid_density,
            pb_fluid=self.p0_fluid,
            c0_fluid=self.c0_fluid,
            nu_fluid=0. * 1e-6,
            mach_no_fluid=self.mach_no_fluid,
            mach_no_structure=self.mach_no_wing,
            gy=self.gy,
            artificial_vis_alpha=1.,
            alpha=0.1,
        )

        if self.options.scheme == 'substep' or 'sswcsph':
            self.scheme.configure(
                dt_fluid=self.dt_fluid,
                dt_solid=self.dt_solid
            )

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

    def customize_output(self):
        self._mayavi_config('''
        for name in ['fluid']:
            b = particle_arrays[name]
            b.scalar = 'p'
            b.range = '-1000, 1000'
            b.plot.module_manager.scalar_lut_manager.lut_mode = 'seismic'
        for name in ['fluid', 'wing', 'wing_support']:
            b = particle_arrays[name]
            b.point_size = 2.0
        ''')


if __name__ == '__main__':
    app = FlappingWing()
    app.run()
    app.post_process(app.info_filename)
