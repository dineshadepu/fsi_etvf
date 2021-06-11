"""A hydrostatic water column on an elastic plate

https://www.sciencedirect.com/science/article/pii/S0029801820314608#appsec1

Appendix A
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

from fsi_coupling import FSIScheme
from fsi_coupling_wcsph import FSIWCSPHScheme
from fsi_substepping import FSISubSteppingScheme
from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)


def get_hydrostatic_tank_with_fluid(fluid_length=1., fluid_height=2., tank_height=2.3,
                                    tank_layers=2, fluid_spacing=0.1):
    import matplotlib.pyplot as plt

    xf, yf = get_2d_block(dx=fluid_spacing,
                          length=fluid_length+fluid_spacing/2.,
                          height=fluid_height+fluid_spacing/2.)

    xt_1, yt_1 = get_2d_block(dx=fluid_spacing,
                              length=tank_layers*fluid_spacing,
                              height=tank_height+fluid_spacing/2.)
    xt_1 -= max(xt_1) - min(xf) + fluid_spacing
    yt_1 += min(yf) - min(yt_1)

    xt_2, yt_2 = get_2d_block(dx=fluid_spacing,
                              length=tank_layers*fluid_spacing,
                              height=tank_height+fluid_spacing/2.)
    xt_2 += max(xf) - min(xt_2) + fluid_spacing
    yt_2 += min(yf) - min(yt_2)

    xt = np.concatenate([xt_1, xt_2])
    yt = np.concatenate([yt_1, yt_2])

    # plt.scatter(xf, yf, s=10)
    # plt.scatter(xt, yt, s=10)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig("geometry", dpi=300)
    # plt.show()

    return xf, yf, xt, yt


def get_fixed_beam_no_clamp(beam_length, beam_height, beam_inside_length,
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

    # create a (support) block with required number of layers
    xs, ys = get_2d_block(dx=spacing,
                          length=boundary_layers*spacing,
                          height=beam_height + beam_height)

    return xb, yb, xs, ys


class ElasticGate(Application):
    def add_user_options(self, group):
        group.add_argument("--gate-rho", action="store", type=float, dest="gate_rho",
                           default=1500.,
                           help="Density of the gate (Defaults to 1500.)")

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
        self.fluid_length = 0.8
        self.fluid_height = 0.6
        self.fluid_density = 1000.0

        spacing = 0.02 / 4.
        self.hdx = 1.0

        self.fluid_spacing = spacing

        self.tank_height = 0.8
        self.tank_length = 0.6
        self.tank_layers = 3
        self.tank_spacing = spacing

        self.h_fluid = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.vref_fluid = np.sqrt(2 * 9.81 * self.fluid_height)
        self.u_max_fluid = self.vref_fluid
        self.c0_fluid = 10 * self.vref_fluid
        self.mach_no_fluid = self.vref_fluid / self.c0_fluid
        self.p0_fluid = self.fluid_density * self.c0_fluid**2.
        self.alpha = 0.1
        self.gy = -1.

        # for boundary particles
        self.seval = None
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "tank", "gate"],
            boundaries=["tank", "gate"])
        # print(self.boundary_equations)

        # ================================================
        # properties related to the elastic gate
        # ================================================
        # elastic gate is made of rubber
        self.gate_length = 0.35
        self.gate_height = 0.02
        self.gate_spacing = self.fluid_spacing

        self.gate_rho0 = self.options.gate_rho
        self.gate_E = 1.4 * 1e6
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
        self.u_max_gate = 13
        self.mach_no_gate = self.u_max_gate / self.c0_gate

        # for pre step
        # self.seval = None

        # boundary equations
        # self.boundary_equations = get_boundary_identification_etvf_equations(
        #     destinations=["gate"], sources=["gate"])
        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["gate"], sources=["gate"])

        self.boundary_equations = self.boundary_equations_1 + self.boundary_equations_2

        self.wall_layers = 2

        self.artificial_stress_eps = 0.3

        self.dt_fluid = 0.125 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        self.dt_solid = 0.25 * self.h_fluid / (
            (self.gate_E / self.gate_rho0)**0.5 + self.u_max_gate)

    def create_particles(self):
        # ===================================
        # Create fluid
        # ===================================
        xf, yf, xt, yt = hydrostatic_tank_2d(
            self.fluid_length, self.fluid_height, self.tank_height,
            self.tank_layers, self.fluid_spacing, self.fluid_spacing)

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
                                  rad_s=self.fluid_spacing/2.,
                                  name="tank")

        # =============================================
        # Only structures part particle properties
        # =============================================
        xp, yp, xw, yw = get_fixed_beam_no_clamp(self.gate_length,
                                                 self.gate_height,
                                                 self.gate_length/4.,
                                                 self.tank_layers,
                                                 self.fluid_spacing)
        # translate the beam into the tank
        xp += -0.10 - min(xp)
        yp += 0.3 - max(yp)

        m = self.gate_rho0 * self.fluid_spacing**2.

        # ===================================
        # Create elastic gate
        # ===================================
        # xp += self.fluid_length
        gate = get_particle_array(
            x=xp, y=yp, m=m, h=self.h_fluid, rho=self.gate_rho0, name="gate",
            constants={
                'E': self.gate_E,
                'n': 4.,
                'nu': self.gate_nu,
                'spacing0': self.gate_spacing,
                'rho_ref': self.gate_rho0
            })

        # # ===================================
        # # Create elastic gate support
        # # ===================================
        # gate_support = get_particle_array(
        #     x=xw, y=yw, m=m, h=self.h_fluid, rho=self.gate_rho0, name="gate_support",
        #     constants={
        #         'E': self.gate_E,
        #         'n': 4.,
        #         'nu': self.gate_nu,
        #         'spacing0': self.gate_spacing,
        #         'rho_ref': self.gate_rho0
        #     })

        self.scheme.setup_properties([fluid, tank, gate])

        gate.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate.rho_fsi[:] = self.fluid_density

        # gate_support.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        # gate_support.rho_fsi[:] = self.fluid_density

        # gate_support.x[:] -= max(gate_support.x) - min(gate.x) + self.fluid_spacing
        # gate_support.y[:] += max(gate.y) - min(gate_support.y) - self.gate_height - 2. * self.fluid_spacing

        # Remove the fluid particles which are intersecting the gate and
        # gate_support
        # collect the indices which are closer to the stucture
        indices = []
        min_xs = min(gate.x)
        max_xs = max(gate.x)
        min_ys = min(gate.y)
        max_ys = max(gate.y)

        xf = fluid.x
        yf = fluid.y
        for i in range(len(fluid.x)):
            if xf[i] < max_xs + self.fluid_spacing / 2. and xf[i] > min_xs - self.fluid_spacing / 2.:
                if yf[i] < max_ys + self.fluid_spacing / 2. and yf[i] > min_ys - self.fluid_spacing / 2.:
                    indices.append(i)

        fluid.remove_particles(indices)

        gate.x[:] += self.fluid_spacing/2.

        return [fluid, tank, gate]

    def create_scheme(self):
        etvf = FSIScheme(fluids=['fluid'],
                         solids=['tank'],
                         structures=['gate'],
                         structure_solids=None,
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
                                       structure_solids=None,
                                       dt_fluid=self.dt_fluid,
                                       dt_solid=self.dt_solid,
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
                               structure_solids=None,
                               dim=2,
                               h_fluid=0.,
                               rho0_fluid=0.,
                               pb_fluid=0.,
                               c0_fluid=0.,
                               nu_fluid=0.,
                               mach_no_fluid=0.,
                               mach_no_structure=0.,
                               gy=0.)

        s = SchemeChooser(default='etvf', etvf=etvf, wcsph=wcsph,
                          substep=substep)

        return s

    def configure_scheme(self):
        # dt = 0.125 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        # TODO: This has to be changed for solid
        dt = 0.25 * self.h_fluid / (
            (self.gate_E / self.gate_rho0)**0.5 + self.u_max_gate)

        print("DT: %s" % dt)
        tf = 1

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

        self.scheme.configure(
            dim=2,
            h_fluid=self.h_fluid,
            rho0_fluid=self.fluid_density,
            pb_fluid=self.p0_fluid,
            c0_fluid=self.c0_fluid,
            nu_fluid=0.0,
            mach_no_fluid=self.mach_no_fluid,
            mach_no_structure=self.mach_no_gate,
            gy=self.gy,
            artificial_vis_alpha=1.,
            alpha=0.1
        )

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # if self.options.scheme == 'etvf':
        #     equation = eqns.groups[-1][5].equations[4]
        #     equation.sources = ["tank", "fluid", "gate", "gate_support"]
        # # print(equation)

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
        from pysph.solver.utils import iter_output, load
        from pysph.solver.utils import get_files

        files = get_files(fname)

        data = load(files[0])
        # solver_data = data['solver_data']
        arrays = data['arrays']
        pa = arrays['gate']
        y_0 = pa.y[61]

        files = files[0::10]
        # print(len(files))
        t, amplitude = [], []
        for sd, gate in iter_output(files, 'gate'):
            _t = sd['t']
            t.append(_t)
            amplitude.append((y_0 - gate.y[61]) * 1e5)

        # matplotlib.use('Agg')

        import os
        from matplotlib import pyplot as plt

        # res = os.path.join(self.output_dir, "results.npz")
        # np.savez(res, t=t, amplitude=amplitude)

        # gtvf data
        # data = np.loadtxt('./oscillating_plate.csv', delimiter=',')
        # t_gtvf, amplitude_gtvf = data[:, 0], data[:, 1]

        plt.clf()

        # plt.plot(t_gtvf, amplitude_gtvf, "s-", label='GTVF Paper')
        plt.plot(t, amplitude, "-", label='Simulated')

        plt.xlabel('t')
        plt.ylabel('amplitude')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "amplitude_with_t.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = ElasticGate()
    app.run()
    app.post_process(app.info_filename)
    # get_elastic_plate_with_support(1.0, 0.3, 2, 0.05)
    # get_hydrostatic_tank_with_fluid()
