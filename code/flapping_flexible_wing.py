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
from pysph.sph.bc.inlet_outlet_manager import (
    InletInfo, OutletInfo)
from pysph.sph.scheme import add_bool_argument
from pysph.tools.geometry import (get_2d_tank, get_2d_block,
                                  remove_overlap_particles)
from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from pysph.examples.solid_mech.impact import add_properties
from fsi_substepping import FSISubSteppingScheme
from nrbc import (EvaluateCharacterisctics, EvaluateNumberDensity,
                  EvaluatePropertyfromCharacteristics,
                  ShepardInterpolateCharacteristics)
from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

logger = logging.getLogger()

# Fluid mechanical/numerical parameters
u_freestream = 1.0
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
        self.Lt = 2.5  # length of tunnel
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
        self.gy = 0.0

        self.fluid_spacing = self.dx
        self.dx_wing = self.dx
        self.wall_layers = 2

        self.c0_wing = get_speed_of_sound(self.wing_E, self.wing_nu,
                                          self.wing_rho0)
        self.u_max_wing = 2
        self.mach_no_wing = self.u_max_wing / self.c0_wing

        # for boundary particles
        self.seval = None
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "wing", "wing",
                                             "wing_support"],
            boundaries=["wing", "wing", "wing_support"])

        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["wing"], sources=["wing", "wing_support"],
            boundaries=["wing_support"])

        self.boundary_equations = self.boundary_equations_1 + self.boundary_equations_2

        self.dt_fluid = 0.25 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        self.dt_solid = 0.25 * self.h_fluid / (
            (self.wing_E / self.wing_rho0)**0.5 + self.u_max_wing)

    def _set_wall_normal(self, pa):
        props = ['xn', 'yn', 'zn']
        for p in props:
            pa.add_property(p)

        y = pa.y
        cond = y > 0.0
        pa.yn[cond] = 1.0
        cond = y < 0.0
        pa.yn[cond] = -1.0

    def _create_box(self):
        wing, wing_support = self._create_flapping_structure()

        dx = self.dx
        m = self.fluid_density * dx * dx
        h0 = self.hdx*dx
        layers = self.nl * dx
        w = self.Wt
        l = self.Lt
        x, y = np.mgrid[-layers:l+layers:dx, -w-layers:w+layers:dx]

        # First form walls, then inlet and outlet, and finally fluid.
        wall_cond = (y > w - dx/2) | (y < -w + dx/2)
        xw, yw = x[wall_cond], y[wall_cond]
        x, y = x[~wall_cond], y[~wall_cond]
        xw = np.concatenate([xw, wing_support.x])
        yw = np.concatenate([yw, wing_support.y])
        wall = get_particle_array(
            name='wall', x=xw, y=yw, m=m, h=h0, rho=self.fluid_density
        )

        inlet_cond = (x < dx/2)
        xi, yi = x[inlet_cond], y[inlet_cond]
        x, y = x[~inlet_cond], y[~inlet_cond]
        inlet = get_particle_array(
            name='inlet', x=xi, y=yi, m=m, h=h0, u=u_freestream, rho=rho,
            p=0.0, uhat=u_freestream
        )

        outlet_cond = (x > l - dx/2)
        xo, yo = x[outlet_cond], y[outlet_cond]
        # Use uhat=umax. So that the particles are moving out, if 0.0 is used
        # instead, the outlet particles will not move.
        outlet = get_particle_array(
            name='outlet', x=xo, y=yo, m=m, h=h0, u=u_freestream, rho=rho,
            p=0.0, uhat=u_freestream, vhat=0.0
        )

        xf, yf = x[~outlet_cond], y[~outlet_cond]
        fluid = get_particle_array(
            name='fluid', x=xf, y=yf, m=m, h=h0, u=u_freestream,
            rho=self.fluid_density, p=0.0, vmag=0.0
        )
        return fluid, wall, inlet, outlet

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
        fluid, wall, inlet, outlet = self._create_box()
        wing, wing_support = self._create_flapping_structure()

        particles = [fluid, inlet, outlet, wall, wing, wing_support]

        # move the solid in middle
        scale = min(fluid.x) - min(wing_support.x)
        wing.x[:] += scale + 0.2
        wing_support.x[:] += scale + 0.2

        remove_overlap_particles(fluid, wing, self.dx)
        remove_overlap_particles(fluid, wing_support, self.dx)

        self.scheme.setup_properties(particles)

        self._set_wall_normal(wall)

        DEFAULT_PROPS = [
            'xn', 'yn', 'zn', 'J2v', 'J3v', 'J2u', 'J3u', 'J1', 'wij2', 'disp',
            'ioid', 'wij'
        ]
        for prop in DEFAULT_PROPS:
            if prop not in wall.properties:
                wall.add_property(prop)

            if prop not in fluid.properties:
                fluid.add_property(prop)

            if prop not in inlet.properties:
                inlet.add_property(prop)

            if prop not in outlet.properties:
                outlet.add_property(prop)

        outlet.xn[:] = 1.0
        outlet.yn[:] = 0.0
        inlet.xn[:] = -1.0
        inlet.yn[:] = 0.0

        wall.add_constant('uref', 0.0)
        consts = [
            'avg_j2u', 'avg_j3u', 'avg_j2v', 'avg_j3v', 'avg_j1', 'uref'
        ]
        for const in consts:
            if const not in wall.constants:
                wall.add_constant(const, 0.0)

            if const not in fluid.constants:
                fluid.add_property(const)

            if const not in inlet.constants:
                inlet.add_property(const)

            if const not in outlet.constants:
                outlet.add_property(const)

        wing.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        wing.rho_fsi[:] = self.fluid_density

        wing_support.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        wing_support.rho_fsi[:] = self.fluid_density

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

    def configure_scheme(self):
        scheme = self.scheme
        self.iom = self._create_inlet_outlet_manager()
        scheme.inlet_outlet_manager = self.iom
        self.iom.update_dx(self.dx)
        # scheme.configure(nu=self.nu)

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

    def create_equations(self):
        from pysph.sph.equation import Group
        from edac import EvaluateNumberDensity
        from pysph.sph.bc.inlet_outlet_manager import (
            UpdateNormalsAndDisplacements
        )

        equations = self.scheme.get_equations()
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
                UpdateNormalsAndDisplacements(
                    'inlet', None, xn=-1, yn=0, zn=0, xo=0, yo=0, zo=0
                ),
                UpdateNormalsAndDisplacements(
                    'outlet', None, xn=1, yn=0, zn=0, xo=0, yo=0, zo=0
                ),
                EvaluateNumberDensity(dest='inlet', sources=['fluid']),
                ShepardInterpolateCharacteristics(dest='inlet', sources=['fluid']),
                # EvaluateNumberDensity(dest='wall', sources=['fluid']),
                # ShepardInterpolateCharacteristics(dest='wall', sources=['fluid']),
                EvaluateNumberDensity(dest='outlet', sources=['fluid']),
                ShepardInterpolateCharacteristics(dest='outlet', sources=['fluid']),
            ])
        )
        eq.append(Group(equations=[
            # EvaluatePropertyfromCharacteristics(
            #     dest='wall', sources=None, c_ref=c0, rho_ref=rho,
            #     u_ref=u_freestream, v_ref=0.0, p_ref=0.0
            # ),
            EvaluatePropertyfromCharacteristics(
                dest='inlet', sources=None, c_ref=c0, rho_ref=rho,
                u_ref=u_freestream, v_ref=0.0, p_ref=0.0
            ),
            EvaluatePropertyfromCharacteristics(
                dest='outlet', sources=None, c_ref=c0, rho_ref=rho,
                u_ref=u_freestream, v_ref=0.0, p_ref=0.0
            )])
        )
        # Remove solid wall bc in the walls.
        # wall_solid_bc = equations[2].equations.pop()
        equations = eq + equations
        # Remove Compute average pressure on inlet and outlet.
        return equations

    def _get_io_info(self):
        from pysph.sph.bc.hybrid.outlet import Outlet
        from hybrid_simple_inlet_outlet import Inlet, SimpleInletOutlet

        i_has_ghost = False
        o_has_ghost = False
        i_update_cls = Inlet
        o_update_cls = Outlet
        manager = SimpleInletOutlet

        props_to_copy = [
            'x0', 'y0', 'z0', 'uhat', 'vhat', 'what', 'x', 'y', 'z',
            'u', 'v', 'w', 'm', 'h', 'rho', 'p', 'ioid'
        ]
        props_to_copy += ['u0', 'v0', 'w0', 'p0']

        inlet_info = InletInfo(
            pa_name='inlet', normal=[-1.0, 0.0, 0.0],
            refpoint=[self.dx/2, 0.0, 0.0], equations=None,
            has_ghost=i_has_ghost, update_cls=i_update_cls,
            umax=u_freestream
        )

        outlet_info = OutletInfo(
            pa_name='outlet', normal=[1.0, 0.0, 0.0],
            refpoint=[self.Lt - self.dx/2, 0.0, 0.0], equations=None,
            has_ghost=o_has_ghost, update_cls=o_update_cls,
            props_to_copy=props_to_copy
        )

        return inlet_info, outlet_info, manager

    def _create_inlet_outlet_manager(self):
        inlet_info, outlet_info, manager = self._get_io_info()
        iom = manager(
            fluid_arrays=['fluid'], inletinfo=[inlet_info],
            outletinfo=[outlet_info]
        )
        return iom

    def create_inlet_outlet(self, particle_arrays):
        iom = self.iom
        io = iom.get_inlet_outlet(particle_arrays)

        return io

    def customize_output(self):
        self._mayavi_config('''
        for name in ['fluid', 'inlet', 'outlet']:
            b = particle_arrays[name]
            b.scalar = 'p'
            b.range = '-1000, 1000'
            b.plot.module_manager.scalar_lut_manager.lut_mode = 'seismic'
        for name in ['fluid', 'wing', 'wing_support']:
            b = particle_arrays[name]
            b.point_size = 2.0
        ''')

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
    app = FlappingWing()
    app.run()
    app.post_process(app.info_filename)
