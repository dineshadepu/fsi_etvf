"""
A \delta SPH-SPIM coupled method for fluid-structure interaction problems

https://doi.org/10.1016/j.jfluidstructs.2020.103210

Section 3.4 Dam-break flow impacting on an elastic plate
"""

import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

# from rigid_fluid_coupling import RigidFluidCouplingScheme
from geometry import hydrostatic_tank_2d, create_tank_2d_from_block_2d

from pysph.examples.solid_mech.impact import add_properties
# from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
#                                                                create_fluid,
#                                                                create_sphere)
from pysph.tools.geometry import get_2d_block, rotate

from fsi_coupling import FSIETVFScheme, FSIETVFSubSteppingScheme
from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from pysph.sph.scheme import add_bool_argument
from pysph.tools.geometry import (remove_overlap_particles)


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
    # create a block first
    xb, yb = get_2d_block(dx=spacing, length=beam_length, height=beam_height)

    # create a (support) block with required number of layers
    xs, ys = get_2d_block(dx=spacing, length=beam_length,
                          height=boundary_layers * spacing)
    ys -= np.max(yb) - np.min(ys) + spacing

    # import matplotlib.pyplot as plt
    # plt.scatter(xs, ys, s=1)
    # plt.scatter(xb, yb, s=1)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig("geometry", dpi=300)
    # plt.show()

    return xb, yb, xs, ys


def get_fixed_beam_with_clamp(beam_length, beam_height, beam_inside_length,
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
    xb, yb = get_2d_block(dx=spacing, length=beam_length + beam_inside_length,
                          height=beam_height)

    # create a (support) block with required number of layers
    xs1, ys1 = get_2d_block(dx=spacing, length=beam_inside_length,
                            height=boundary_layers * spacing)
    xs1 += np.min(xb) - np.min(xs1)
    ys1 += np.min(yb) - np.max(ys1) - spacing

    # create a (support) block with required number of layers
    xs2, ys2 = get_2d_block(dx=spacing, length=beam_inside_length,
                            height=boundary_layers * spacing)
    xs2 += np.min(xb) - np.min(xs2)
    ys2 += np.max(ys2) - np.min(yb) + spacing

    xs = np.concatenate([xs1, xs2])
    ys = np.concatenate([ys1, ys2])

    xs3, ys3 = get_2d_block(dx=spacing, length=boundary_layers * spacing,
                            height=np.max(ys) - np.min(ys))
    xs3 += np.min(xb) - np.max(xs3) - 1. * spacing
    # ys3 += np.max(ys2) - np.min(yb) + spacing

    xs = np.concatenate([xs, xs3])
    ys = np.concatenate([ys, ys3])
    # plt.scatter(xs, ys, s=1)
    # plt.scatter(xb, yb, s=1)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig("geometry", dpi=300)
    # plt.show()

    return xb, yb, xs, ys


def get_fixed_beam_with_out_clamp(beam_length, beam_height, beam_inside_length,
                                  boundary_length, boundary_height,
                                  boundary_layers, spacing):
    # create a block first
    xb, yb = get_2d_block(dx=spacing, length=beam_length,
                          height=beam_height)

    xs, ys = get_2d_block(dx=spacing, length=boundary_length,
                          height=boundary_height)
    ys -= max(ys) - min(yb) + spacing

    return xb, yb, xs, ys


class Ng2020DamBreakWithAnElasticStructureDB2(Application):
    def add_user_options(self, group):
        group.add_argument("--d0", action="store", type=float, dest="d0",
                           default=1e-3,
                           help="Spacing between the particles")

    def consume_user_options(self):
        # ================================================
        # consume the user options first
        # ================================================
        spacing = self.options.d0

        # ================================================
        # common properties
        # ================================================
        self.hdx = 1.0
        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 2
        self.seval = None

        # ================================================
        # Fluid properties
        # ================================================
        self.fluid_length = 0.2
        self.fluid_height = 0.2
        self.fluid_spacing = spacing
        self.h_fluid = self.hdx * self.fluid_spacing
        self.vref_fluid = np.sqrt(2 * 9.81 * self.fluid_height)
        self.c0_fluid = 10 * self.vref_fluid
        self.nu_fluid = 0.
        self.rho0_fluid = 1000.0
        self.mach_no_fluid = self.vref_fluid / self.c0_fluid
        self.pb_fluid = self.rho0_fluid * self.c0_fluid**2.
        self.alpha_fluid = 0.1
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0_fluid * self.h_fluid / 8
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "tank", "gate",
                                             "gate_support"],
            boundaries=["tank", "gate", "gate_support"])

        # ================================================
        # Tank properties
        # ================================================
        self.tank_height = 0.4
        self.tank_length = 0.8
        self.tank_layers = 3
        self.tank_spacing = spacing
        self.wall_layers = 3

        # ================================================
        # properties related to the elastic gate
        # ================================================
        # elastic gate is made of rubber
        self.gate_length = 0.004
        self.gate_height = 0.09
        self.gate_spacing = self.fluid_spacing
        self.gate_rho0 = 1161.54
        self.gate_E = 3.5 * 1e6
        self.gate_nu = 0.45
        self.c0_gate = get_speed_of_sound(self.gate_E, self.gate_nu,
                                          self.gate_rho0)
        self.u_max_gate = self.u_max_fluid
        self.mach_no_gate = self.u_max_gate / self.c0_gate
        self.alpha_solid = 1.
        self.beta_solid = 0.
        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["gate"], sources=["gate", "gate_support"],
            boundaries=["gate_support"])

        # ================================================
        # common properties
        # ================================================
        self.boundary_equations = (self.boundary_equations_1 +
                                   self.boundary_equations_2)

        self.dt_fluid = 0.25 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        self.dt_solid = 0.25 * self.h_fluid / (
            (self.gate_E_A / self.gate_rho0)**0.5 + self.u_max_gate)

    def create_particles(self):
        # ===================================
        # Create fluid
        # ===================================
        xf, yf = get_2d_block(dx=self.fluid_spacing, length=self.fluid_length,
                              height=self.fluid_length)

        xt, yt = create_tank_2d_from_block_2d(
            xf, yf, self.tank_length, self.tank_height, self.tank_spacing,
            self.tank_layers)

        m_fluid = self.fluid_density * self.fluid_spacing**2.

        # =============================================
        # Only fluids part particle properties
        # =============================================

        # ===================================
        # Create fluid
        # ===================================
        fluid = get_particle_array(x=xf,
                                   y=yf,
                                   m=m_fluid,
                                   h=self.h_fluid,
                                   rho=self.fluid_density,
                                   name="fluid")

        # ===================================
        # Create tank
        # ===================================
        tank = get_particle_array(x=xt,
                                  y=yt,
                                  m=m_fluid,
                                  m_fluid=m_fluid,
                                  h=self.h_fluid,
                                  rho=self.fluid_density,
                                  rad_s=self.tank_spacing/2.,
                                  name="tank")

        # =============================================
        # Only structures part particle properties
        # =============================================
        # xp, yp, xw, yw = get_fixed_beam(self.L, self.H, self.H/2.5,
        #                                 self.wall_layers, self.fluid_spacing)

        xp, yp, xw, yw = get_fixed_beam_with_out_clamp(self.gate_length, self.gate_height,
                                                       self.gate_height/2.5,
                                                       self.gate_length * 3.,
                                                       (self.wall_layers + 1) * self.fluid_spacing,
                                                       self.wall_layers,
                                                       self.fluid_spacing)
        # move the wall onto the tank
        scale = max(yw) - min(tank.y) - self.wall_layers * self.fluid_spacing
        yw -= scale
        yp -= scale

        scale = min(xp) - min(fluid.x)
        xp -= scale
        xw -= scale

        m = self.gate_rho0 * self.fluid_spacing**2.

        # ===================================
        # Create elastic gate
        # ===================================
        shift = 0.8 - 0.2 - 0.004
        xp += shift
        gate = get_particle_array(
            x=xp, y=yp, m=m, h=self.h_fluid, rho=self.gate_rho0, name="gate",
            constants={
                'E': self.gate_E,
                'n': 4.,
                'nu': self.gate_nu,
                'spacing0': self.gate_spacing,
                'rho_ref': self.gate_rho0
            })

        # ===================================
        # Create elastic gate support
        # ===================================
        xw += shift
        gate_support = get_particle_array(
            x=xw, y=yw, m=m, h=self.h_fluid, rho=self.gate_rho0, name="gate_support",
            constants={
                'E': self.gate_E,
                'n': 4.,
                'nu': self.gate_nu,
                'spacing0': self.gate_spacing,
                'rho_ref': self.gate_rho0
            })

        # ===========================
        # Adjust the geometry
        # ===========================
        remove_overlap_particles(tank, gate_support, self.fluid_spacing)

        self.scheme.setup_properties([fluid, tank,
                                      gate, gate_support])

        gate.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate.rho_fsi[:] = self.fluid_density

        gate_support.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate_support.rho_fsi[:] = self.fluid_density

        return [fluid, tank, gate, gate_support]

    def create_scheme(self):
        ctvf = FSIETVFScheme(fluids=['fluid'],
                             solids=['tank'],
                             structures=['gate'],
                             structure_solids=['gate_support'],
                             dim=2,
                             h_fluid=0.,
                             c0_fluid=0.,
                             nu_fluid=0.,
                             rho0_fluid=0.,
                             mach_no_fluid=0.,
                             mach_no_structure=0.)

        substep = FSIETVFSubSteppingScheme(fluids=['fluid'],
                                           solids=['tank'],
                                           structures=['gate'],
                                           structure_solids=['gate_support'],
                                           dim=2,
                                           h_fluid=0.,
                                           c0_fluid=0.,
                                           nu_fluid=0.,
                                           rho0_fluid=0.,
                                           mach_no_fluid=0.,
                                           mach_no_structure=0.)

        s = SchemeChooser(default='substep', ctvf=ctvf, substep=substep)

        return s

    def configure_scheme(self):
        dt = self.dt_fluid
        tf = 1.

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=2000)

        self.scheme.configure(
            dim=2,
            h_fluid=self.h_fluid,
            rho0_fluid=self.rho0_fluid,
            pb_fluid=self.pb_fluid,
            c0_fluid=self.c0_fluid,
            nu_fluid=0.0,
            mach_no_fluid=self.mach_no_fluid,
            mach_no_structure=self.mach_no_gate,
            gy=self.gy,
            alpha_fluid=0.1,
            alpha_solid=1.,
            beta_solid=0,
            dt_fluid=self.dt_fluid,
            dt_solid=self.dt_solid
        )

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # print(eqns)
        # equation = eqns.groups[-1][5].equations[4]
        # equation.sources = ["tank", "fluid", "gate", "gate_support"]
        # print(equation)

        return eqns

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
    app = Ng2020DamBreakWithAnElasticStructureDB2()
    app.run()
    # app.post_process(app.info_filename)
