from __future__ import print_function
import numpy as np

# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.sph.equation import Equation

from pysph.solver.application import Application
from pysph.sph.rigid_body import BodyForce

from pysph.sph.scheme import SchemeChooser
from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.wc.gtvf import GTVFIntegrator

# from pysph.dem.common import (EPECIntegratorMultiStage)
from potyondy import (get_particle_array_bonded_dem_potyondy,
                      setup_bc_contacts, PotyondyIPForce3D,
                      LeapFrogStepPotyondy, GlobalDamping,
                      MakeForcesTorqueZero, ResetForceAndMoment)


class ApplyTensionForce(Equation):
    def __init__(self, dest, sources, idx, delta_fx=0, delta_fy=0, delta_fz=0):
        self.force_increment_x = delta_fx
        self.force_increment_y = delta_fy
        self.force_increment_z = delta_fz
        self.idx = idx
        super(ApplyTensionForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz, d_final_force_time,
                   d_total_force_applied_x, d_total_force_applied_y,
                   d_total_force_applied_z, t):
        if t < d_final_force_time[0]:
            d_total_force_applied_x[0] += self.force_increment_x
            d_total_force_applied_y[0] += self.force_increment_y
            d_total_force_applied_z[0] += self.force_increment_z
            # print(d_total_force_applied_x[0])

        if d_idx == self.idx:
            d_fx[d_idx] = d_total_force_applied_x[0]
            d_fy[d_idx] = d_total_force_applied_y[0]
            d_fz[d_idx] = d_total_force_applied_z[0]


class GuoTensionTestValidation(Application):
    def initialize(self):
        self.dt = 1e-4
        self.pfreq = 100
        self.total_time = 2.
        self.final_force_time = 1.
        self.tf = self.total_time
        self.dim = 2
        self.gy = -9.81
        self.seval = None

        self.beam_len = 1.
        self.no_particles = 11
        x = np.linspace(0., self.beam_len, self.no_particles)
        self.radius = (x[1] - x[0]) / 2.
        self.beam_E = 1e7
        self.beam_nu = 0.33
        self.beam_G = self.beam_E / (2. * (1. + self.beam_nu))
        self.rho = 2600
        self.m = self.rho * 4. / 3. * np.pi * self.radius**3.
        print("radius is ")
        print(self.radius)
        self.force_idx = self.no_particles - 1
        self.fx = 100
        self.initial_x_positions = x
        bond_area = np.pi * self.radius**2.
        self.max_displacement = (self.fx * self.beam_len) / (self.beam_E *
                                                             bond_area)
        print("max displacement is ", self.max_displacement)

        # no of time steps used to apply this force
        timesteps = self.final_force_time / self.dt
        self.delta_fx = self.fx / timesteps
        self.delta_fy = 0.
        self.delta_fz = 0.

        tmp = 0.
        force = 0.
        while tmp < self.final_force_time:
            force += self.delta_fx
            tmp += self.dt

        print("force is applied check")
        print(force)
        self.plot_show = False

    def create_particles(self):
        # create a particle
        x = np.linspace(0., self.beam_len, self.no_particles)
        # print(len(x))
        y = np.zeros_like(x)

        rho = self.rho
        m = self.m
        E = self.beam_E
        # this has to be the moment of inertia of a 2d disc
        # please check
        I = 2. / 5. * m * self.radius**2.

        # time step calculation
        dt = np.sqrt(m / E)
        print("timestep is ", dt)

        m_inverse = 1. / m
        I_inverse = 1. / I
        beam = get_particle_array_bonded_dem_potyondy(
            x=x, y=y, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=self.radius, dem_id=1, h=1.2 * self.radius,
            young_mod=self.beam_E, shear_mod=self.beam_G, name="beam")
        setup_bc_contacts(2, beam, 0.1)
        # print(beam.bc_idx)
        print("Total no of contacts are ")
        print(beam.bc_total_contacts)

        zero_force_idx = np.zeros_like(beam.x)
        beam.add_property('zero_force_idx', type='int', data=zero_force_idx)
        # print(beam.zero_force_idx)
        beam.zero_force_idx[0] = 1
        # print(beam.zero_force_idx)

        zero_moment_idx = np.zeros_like(beam.x)
        beam.add_property('zero_moment_idx', type='int', data=zero_moment_idx)
        # print(beam.zero_force_idx)
        beam.zero_moment_idx[0] = 1

        beam.add_constant('final_force_time',
                          np.array([self.final_force_time]))
        beam.add_constant('total_force_applied_x', np.array([0.]))
        beam.add_constant('total_force_applied_y', np.array([0.]))
        beam.add_constant('total_force_applied_z', np.array([0.]))

        return [beam]

    def create_equations(self):
        stage1 = []
        stage2 = [
            Group(equations=[
                MakeForcesTorqueZero(dest='beam', sources=None),
                # BodyForce(dest='beam', sources=None, gx=0.0, gy=-9.81,
                #           gz=0.0),
                ApplyTensionForce(dest='beam', sources=None,
                                  delta_fx=self.delta_fx, delta_fy=0.,
                                  delta_fz=0., idx=self.force_idx),
                PotyondyIPForce3D(dest='beam', sources=None),
                ResetForceAndMoment(dest='beam', sources=None),
            ]),
            Group(equations=[GlobalDamping(dest='beam', sources=None)])
        ]

        return MultiStageEquations([stage1, stage2])

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)

        integrator = GTVFIntegrator(beam=LeapFrogStepPotyondy())

        dt = self.dt
        tf = self.tf
        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=dt, tf=tf)
        solver.set_disable_output(True)
        return solver

    # def customize_output(self):
    #     self._mayavi_config('''
    #     b = particle_arrays['beam']
    #     b.vectors = 'fx, fy, fz'
    #     b.show_vectors = True
    #     b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
    #     b.plot.glyph.glyph_source.glyph_source.radius = {s_rad}
    #     b.scalar = 'fy'
    #     '''.format(s_rad=self.radius/1.25))

    def post_process(self, info_fname):
        # self.read_info(info_fname)
        # if len(self.output_files) == 0:
        #     return

        from pysph.solver.utils import iter_output, load
        import matplotlib.pyplot as plt

        files = self.output_files
        # simulated data
        t, x, u, applied_force = [], [], [], []
        for sd, arrays in iter_output(files):
            beam = arrays['beam']
            t.append(sd['t'])
            x.append(beam.x[2])
            u.append(beam.u[2])
            applied_force.append(beam.total_force_applied_x[0])

        # data = np.loadtxt('ffpw_y.csv', delimiter=',')
        # ta = data[:, 0]
        # ya = data[:, 1]
        # plt.plot(ta, ya)
        plt.plot(t, x)
        plt.savefig('t_vs_x.png')
        if self.plot_show is True:
            plt.show()
        plt.clf()

        # data = np.loadtxt('ffpw_v.csv', delimiter=',')
        # ta = data[:, 0]
        # va = data[:, 1]
        # plt.plot(ta, va)
        plt.plot(t, u)
        if self.plot_show is True:
            plt.show()
        plt.savefig('t_vs_u.png')

        plt.clf()
        plt.plot(t, applied_force)
        # if self.plot_show is True:
        #     plt.show()
        plt.show()
        plt.savefig('t_vs_applied_force.png')

        # ====================
        # dispalcement plot
        # ====================
        files = self.output_files

        final_file = files[-1]
        print(final_file)
        initial_x_positions = self.initial_x_positions
        data = load(final_file)
        arrays = data['arrays']
        beam = arrays['beam']

        # bond_area = np.pi * self.radius**2.
        # max_displacement = (self.fx * max(beam.x)) / (self.beam_E * bond_area)
        # max_displacement = self.max_displacement
        disp = beam.x - initial_x_positions
        print("displacement of the final particle")
        print(disp[-1])
        disp_by_max_disp = disp / self.max_displacement
        x_by_L = initial_x_positions / self.beam_len

        plt.clf()
        plt.plot(x_by_L, disp_by_max_disp)
        plt.show()

        # normal force in the bond check
        # normal_bond_force = np.zeros_like(beam.x)
        # bc_fn_x = beam.bc_fn_x
        # print(bc_fn_x)
        # print(beam.bc_total_contacts)
        # for i in range(len(beam.x)):
        #     j = i * beam.bc_limit[0]
        #     print(j)
        #     print(bc_fn_x[j])

        # print("total force applied x")
        # print(beam.total_force_applied_x)


if __name__ == '__main__':
    app = GuoTensionTestValidation()
    app.run()
    app.post_process(app.info_filename)
