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
from geometry import hydrostatic_tank_2d, create_tank_2d_from_block_2d

from pysph.examples.solid_mech.impact import add_properties
# from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
#                                                                create_fluid,
#                                                                create_sphere)
from pysph.tools.geometry import get_2d_block, rotate

from fsi_coupling import FSIScheme, FSIGTVFScheme
from fsi_substepping import FSISubSteppingScheme
from fsi_coupling_wcsph import FSIWCSPHScheme

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)


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


def get_fixed_beam_without_clamp(beam_length, beam_height, boundary_height, spacing):
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
    xb, yb = get_2d_block(dx=spacing, length=beam_length,
                          height=beam_height)

    # create a (support) block with required number of layers
    xs1, ys1 = get_2d_block(dx=spacing, length=beam_length,
                            height=boundary_height)

    ys1 += np.min(yb) - np.max(ys1) - spacing

    xs = np.concatenate([xs1])
    ys = np.concatenate([ys1])
    # plt.scatter(xs, ys, s=1)
    # plt.scatter(xb, yb, s=1)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig("geometry", dpi=300)
    # plt.show()

    return xb, yb, xs, ys


class ElasticGate(Application):
    def add_user_options(self, group):
        group.add_argument("--rho", action="store", type=float, dest="rho",
                           default=7800.,
                           help="Density of the particle (Defaults to 7800.)")

        group.add_argument(
            "--Vf", action="store", type=float, dest="Vf", default=0.05,
            help="Velocity of the plate (Vf) (Defaults to 0.05)")

        group.add_argument("--length", action="store", type=float,
                           dest="length", default=0.1,
                           help="Length of the plate")

        group.add_argument("--height", action="store", type=float,
                           dest="height", default=0.01,
                           help="height of the plate")

        group.add_argument("--deflection", action="store", type=float,
                           dest="deflection", default=1e-4,
                           help="Deflection of the plate")

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=10,
                           help="No of particles in the height direction")

        group.add_argument("--final-force-time", action="store", type=float,
                           dest="final_force_time", default=1e-3,
                           help="Total time taken to apply the external load")

        group.add_argument("--damping-c", action="store", type=float,
                           dest="damping_c", default=0.1,
                           help="Damping constant in damping force")

        group.add_argument("--material", action="store", type=str,
                           dest="material", default="steel",
                           help="Material of the plate")

        # add_bool_argument(group, 'shepard', dest='use_shepard_correction',
        #                   default=False, help='Use shepard correction')

        # add_bool_argument(group, 'bonet', dest='use_bonet_correction',
        #                   default=False, help='Use Bonet and Lok correction')

        # add_bool_argument(group, 'kgf', dest='use_kgf_correction',
        #                   default=False, help='Use KGF correction')

    def consume_user_options(self):
        self.dim = 2

        # ================================================
        # properties related to the only fluids
        # ================================================
        spacing = 0.001
        self.hdx = 1.0

        self.fluid_length = 0.1
        self.fluid_height = 0.14
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_height = 0.15
        self.tank_length = 0.2
        self.tank_layers = 3
        self.tank_spacing = spacing

        self.h_fluid = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.vref_fluid = np.sqrt(2 * 9.81 * self.fluid_height)
        self.u_max_fluid = self.vref_fluid
        self.c0_fluid = 10 * self.vref_fluid
        print("co is")
        print(self.c0_fluid)
        self.mach_no_fluid = self.vref_fluid / self.c0_fluid
        self.p0_fluid = self.fluid_density * self.c0_fluid**2.
        self.alpha = 0.1
        self.gy = -9.81

        # for boundary particles
        self.seval = None
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "tank", "gate",
                                             "gate_support"],
            boundaries=["tank", "gate", "gate_support"])
        # print(self.boundary_equations)

        # ================================================
        # properties related to the elastic gate
        # ================================================
        # elastic gate is made of rubber
        self.L = 0.005
        self.H = 0.079
        self.gate_spacing = self.fluid_spacing

        self.gate_rho0 = 1100
        self.gate_E = 11958923.292360354
        self.gate_nu = 0.4

        self.c0_gate = get_speed_of_sound(self.gate_E, self.gate_nu,
                                          self.gate_rho0)
        # self.c0 = 5960
        # print("speed of sound is")
        # print(self.c0)
        self.pb_gate = self.gate_rho0 * self.c0_gate**2

        self.edac_alpha = 0.5

        self.edac_nu = self.edac_alpha * self.c0_gate * self.h_fluid / 8

        # attributes for Sun PST technique
        # dummy value, will be updated in consume user options
        self.u_max_gate = 50
        self.mach_no_gate = self.u_max_gate / self.c0_gate

        # for pre step
        # self.seval = None

        # boundary equations
        # self.boundary_equations = get_boundary_identification_etvf_equations(
        #     destinations=["gate"], sources=["gate"])
        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["gate"], sources=["gate", "gate_support"],
            boundaries=["gate_support"])

        self.boundary_equations = self.boundary_equations_1 + self.boundary_equations_2

        self.wall_layers = 2

        self.artificial_stress_eps = 0.3

        self.dt_fluid = 0.25 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        self.dt_solid = 0.25 * self.h_fluid / (
            (self.gate_E / self.gate_rho0)**0.5 + self.u_max_gate)

    def create_particles(self):
        # ===================================
        # Create fluid
        # ===================================
        xf, yf = get_2d_block(dx=self.fluid_spacing, length=self.fluid_length,
                              height=self.fluid_height)

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
        with_out_clamp = True
        if with_out_clamp is True:
            xp, yp, xw, yw = get_fixed_beam_without_clamp(self.H, self.L,
                                                          self.L, self.fluid_spacing)

        else:
            xp, yp, xw, yw = get_fixed_beam(self.H, self.L, self.H/2.5,
                                            self.wall_layers, self.fluid_spacing)

        # make sure that the beam intersection with wall starts at the 0.
        # min_xp = np.min(xp)

        # # add this to the beam and wall
        # xp += abs(min_xp)
        # xw += abs(min_xp)

        # max_xw = np.max(xw)
        # xp -= abs(max_xw)
        # xw -= abs(max_xw)

        m = self.gate_rho0 * self.fluid_spacing**2.

        # ===================================
        # Create elastic gate
        # ===================================
        xp += self.fluid_length
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
        xw += self.fluid_length
        # xw += max(xf) + max(xf) / 2.
        gate_support = get_particle_array(
            x=xw, y=yw, m=m, h=self.h_fluid, rho=self.gate_rho0, name="gate_support",
            constants={
                'E': self.gate_E,
                'n': 4.,
                'nu': self.gate_nu,
                'spacing0': self.gate_spacing,
                'rho_ref': self.gate_rho0
            })
        # ================================
        # Adjust the geometry
        # ================================
        # rotate the particles
        axis = np.array([0.0, 0.0, 1.0])
        angle = -90
        xp, yp, zp = rotate(gate.x, gate.y, gate.z, axis, angle)
        gate.x, gate.y, gate.z = xp[:], yp[:], zp[:]

        xw, yw, zw = rotate(gate_support.x, gate_support.y,
                            gate_support.z, axis, angle)
        gate_support.x, gate_support.y, gate_support.z = xw[:], yw[:], zw[:]

        # translate gate and gate support
        x_translate = (max(fluid.x) - min(gate_support.x)) - self.fluid_spacing * 2.
        gate.x += x_translate
        gate_support.x += x_translate

        y_translate = (max(tank.y) - max(gate_support.y)) + 3. * self.fluid_spacing
        gate.y += y_translate
        gate_support.y += y_translate

        if with_out_clamp is True:
            # set the gate and gate support x variables
            # translate gate and gate support
            x_translate = (max(fluid.x) - min(gate.x)) + self.fluid_spacing
            gate.x += x_translate

            x_translate = (max(fluid.x) - min(gate_support.x)) + self.fluid_spacing
            gate_support.x += x_translate

            y_translate = (min(tank.y) - min(gate.y)) + 3. * self.fluid_spacing
            gate.y += y_translate

            y_translate = (max(gate.y) - min(gate_support.y))
            gate_support.y += y_translate + self.fluid_spacing

        # gate.x[:] += 3. * self.fluid_spacing
        # gate_support.x[:] += 3. * self.fluid_spacing

        # ================================
        # Adjust the geometry ends
        # ================================

        self.scheme.setup_properties([fluid, tank,
                                      gate, gate_support])

        gate.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate.rho_fsi[:] = self.fluid_density

        gate_support.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate_support.rho_fsi[:] = self.fluid_density

        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_density * self.gy * (max(fluid.y) - fluid.y[:])

        return [fluid, tank, gate, gate_support]

    def create_scheme(self):
        etvf = FSIScheme(fluids=['fluid'],
                         solids=['tank'],
                         structures=['gate'],
                         structure_solids=['gate_support'],
                         dim=2,
                         h_fluid=0.,
                         rho0_fluid=0.,
                         pb_fluid=0.,
                         c0_fluid=0.,
                         nu_fluid=0.,
                         mach_no_fluid=0.,
                         mach_no_structure=0.,
                         gy=0.)

        gtvf = FSIGTVFScheme(fluids=['fluid'],
                             solids=['tank'],
                             structures=['gate'],
                             structure_solids=['gate_support'],
                             dim=2,
                             h_fluid=0.,
                             rho0_fluid=0.,
                             pb_fluid=0.,
                             c0_fluid=0.,
                             nu_fluid=0.,
                             mach_no_fluid=0.,
                             mach_no_structure=0.,
                             gy=0.)

        wcsph = FSIWCSPHScheme(fluids=['fluid'],
                               solids=['tank'],
                               structures=['gate'],
                               structure_solids=['gate_support'],
                               dim=2,
                               h_fluid=0.,
                               rho0_fluid=0.,
                               pb_fluid=0.,
                               c0_fluid=0.,
                               nu_fluid=0.,
                               mach_no_fluid=0.,
                               mach_no_structure=0.,
                               gy=0.)

        substep = FSISubSteppingScheme(fluids=['fluid'],
                                       solids=['tank'],
                                       structures=['gate'],
                                       structure_solids=['gate_support'],
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

        s = SchemeChooser(default='etvf', etvf=etvf, gtvf=gtvf, wcsph=wcsph,
                          substep=substep)
        return s

    def configure_scheme(self):
        # dt = 0.125 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        # TODO: This has to be changed for solid
        dt = 0.25 * self.h_fluid / (
            (self.gate_E / self.gate_rho0)**0.5 + (self.u_max_gate/50.))

        self.dt_fluid = 0.25 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        self.dt_solid = 0.25 * self.h_fluid / (
            (self.gate_E / self.gate_rho0)**0.5 + self.u_max_gate)
        if self.options.scheme == "substep":
            dt = self.dt_fluid

        print("DT: %s" % dt)
        tf = 0.2

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

        self.scheme.configure(
            dim=2,
            h_fluid=self.h_fluid,
            rho0_fluid=self.fluid_density,
            pb_fluid=self.p0_fluid,
            c0_fluid=self.c0_fluid,
            nu_fluid=0.00,
            mach_no_fluid=self.mach_no_fluid,
            mach_no_structure=self.mach_no_gate,
            gy=self.gy,
            artificial_vis_alpha=1.,
            alpha=0.1
        )
        if self.options.scheme == 'substep':
            self.scheme.configure(
                dt_fluid=self.dt_fluid,
                dt_solid=self.dt_solid
            )

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # if self.options.scheme == 'etvf':
        #     equation = eqns.groups[-1][5].equations[4]
        #     equation.sources = ["tank", "fluid", "gate", "gate_support"]

        # elif self.options.scheme == 'gtvf':
        #     equation = eqns.groups[-1][4].equations[3]
        #     # print(equation)
        #     equation.sources = ["tank", "fluid", "gate", "gate_support"]

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

    def post_process(self, fname):
        from pysph.solver.utils import iter_output
        from pysph.solver.utils import get_files, load

        files = get_files(fname)

        # initial position of the gate
        index = 479
        data = load(files[0])
        arrays = data['arrays']
        gate = arrays['gate']
        y_initial = gate.y[index]
        x_initial = gate.x[index]

        t, y_amplitude, x_amplitude = [], [], []
        for sd, gate in iter_output(files[::1], 'gate'):
            _t = sd['t']
            t.append(_t)
            y_amplitude.append(gate.y[index] - y_initial)
            x_amplitude.append(gate.x[index] - x_initial)

        import os
        from matplotlib import pyplot as plt

        # gtvf data
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        data = np.loadtxt(os.path.join(directory, 'elastic_dam_break_vertical_displacement_simulated.csv'),
                          delimiter=',')
        t_wcsph, amplitude_wcsph = data[:, 0], data[:, 1]

        data = np.loadtxt(os.path.join(directory, 'elastic_dam_break_vertical_displacement_experimental.csv'),
                          delimiter=',')
        t_exp, amplitude_exp = data[:, 0], data[:, 1]

        plt.clf()
        plt.plot(t_wcsph, amplitude_wcsph, "s-", label='WCSPH Paper')
        plt.plot(t_exp, amplitude_exp, "s-", label='Experiment')
        plt.plot(t, y_amplitude, "-", label='CTVF')

        plt.xlabel('t')
        plt.ylabel('amplitude')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "y_amplitude_with_t.png")
        plt.savefig(fig, dpi=300)

        # x amplitude
        data = np.loadtxt(os.path.join(directory, 'elastic_dam_break_horizontal_displacement_simulated.csv'),
                          delimiter=',')
        t_wcsph, amplitude_wcsph = data[:, 0], data[:, 1]

        data = np.loadtxt(os.path.join(directory, 'elastic_dam_break_horizontal_displacement_experimental.csv'),
                          delimiter=',')
        t_exp, amplitude_exp = data[:, 0], data[:, 1]
        plt.clf()
        plt.plot(t_wcsph, amplitude_wcsph, "s-", label='WCSPH Paper')
        plt.plot(t_exp, amplitude_exp, "s-", label='Experiment')
        plt.plot(t, x_amplitude, "-", label='CTVF')

        plt.xlabel('t')
        plt.ylabel('amplitude')
        plt.legend()
        print(fname)
        fig = os.path.join(os.path.dirname(fname), "x_amplitude_with_t.png")
        print(fig)
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = ElasticGate()
    app.run()
    app.post_process(app.info_filename)
