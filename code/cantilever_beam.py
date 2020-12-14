"""This is taken from

`A scale-invariant bonded particle model for simulating large deformation and
failure of continua`, section 3.2

"""
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
                      setup_bc_contacts, PotyondyIPForce2D,
                      PotyondyIPForce,
                      LeapFrogStepPotyondy, GlobalDamping, ResetForceAndMoment,
                      MakeForcesTorqueZero)


class ApplyTensionForce(Equation):
    def __init__(self, dest, sources, fx=0, fy=0, fz=0):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        super(ApplyTensionForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz, d_force_idx):
        if d_force_idx[d_idx] == 1:
            d_fx[d_idx] = self.fx
            d_fy[d_idx] = self.fy
            d_fz[d_idx] = self.fz


class CantileverBeam(Application):
    def initialize(self):
        self.dt = 0.05
        self.tf = 12000
        self.pfreq = 1000
        self.dim = 3
        # friction coefficient
        self.mu = 0.5
        self.gy = -9.81
        self.seval = None
        self.slow_pfreq = 1
        self.slow_dt = 1e-4
        self.post_u = None
        self.scale_factor = 1.1

        self.radius = 2500
        self.no_particles_in_x_direction = 30
        self.fy = 1.5 * 1e12
        self.fx = 1000 * 0.

    def create_particles(self):
        # create a particle
        diameter = self.radius * 2
        x1 = np.arange(0., self.no_particles_in_x_direction * diameter,
                       diameter)
        x2 = np.arange(0., self.no_particles_in_x_direction * diameter,
                       diameter)
        x3 = np.arange(0., self.no_particles_in_x_direction * diameter,
                       diameter)

        y1 = np.ones_like(x1) * -2. * self.radius
        y2 = np.ones_like(x1) * 0.
        y3 = np.ones_like(x1) * 2. * self.radius

        xp = np.concatenate([x1, x2, x3])
        yp = np.concatenate([y1, y2, y3])

        rho = 900
        m = rho * 4. / 3. * np.pi * self.radius**3.
        I = 2. / 5. * m * self.radius**2.
        m_inverse = 1. / m
        I_inverse = 1. / I
        young_mod = 1 * 1e9
        poisson_ratio = 0.3
        shear_mod = young_mod / (2. * (1 + poisson_ratio))
        beam = get_particle_array_bonded_dem_potyondy(
            x=xp, y=yp, m=m, I_inverse=I_inverse, m_inverse=m_inverse,
            rad_s=self.radius, dem_id=0, h=1.2 * self.radius,
            young_mod=young_mod, shear_mod=shear_mod, name="beam")
        setup_bc_contacts(2, beam, 0.5)
        # print(beam.bc_idx)
        print("Total no of contacts are ")
        print(beam.bc_total_contacts)

        # ============================================ #
        # find all the indices which are at the left most
        # ============================================ #
        # min_x = min(xp)
        indices = np.where(xp < 2 * self.radius)
        # print(indices[0])

        zero_force_idx = np.zeros_like(beam.x)
        beam.add_property('zero_force_idx', type='int', data=zero_force_idx)
        # print(beam.zero_force_idx)
        beam.zero_force_idx[indices] = 1
        # print(beam.zero_force_idx)

        zero_moment_idx = np.zeros_like(beam.x)
        beam.add_property('zero_moment_idx', type='int', data=zero_moment_idx)
        # print(beam.zero_force_idx)
        beam.zero_moment_idx[indices] = 1
        # ============================================ #
        # find all the indices which are at the left most ends
        # ============================================ #

        # ============================================ #
        # find all the indices which are at the right most
        # ============================================ #
        max_x = max(xp)
        indices = np.where(xp > max_x - self.radius / 5.)
        print(indices[0])
        force_idx = np.zeros_like(beam.x)
        beam.add_property('force_idx', type='int', data=force_idx)
        # print(beam.zero_force_idx)
        beam.force_idx[indices] = 1

        return [beam]

    def create_equations(self):
        stage1 = []
        stage2 = [
            Group(equations=[
                MakeForcesTorqueZero(dest='beam', sources=None),
                # BodyForce(dest='beam', sources=None, gx=0.0, gy=-9.81,
                #                gz=0.0),
                ApplyTensionForce(dest='beam', sources=None, fx=self.fx,
                                  fy=self.fy),
                PotyondyIPForce(dest='beam', sources=None, dim=self.dim),
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
                        dt=dt, tf=tf, pfreq=self.pfreq)
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
    #     '''.format(s_rad=self.radius/self.scale_factor))

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['beam']
        b.vectors = 'fx, fy, fz'
        b.show_vectors = True
        b.scalar = 'fy'
        '''.format())

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        import matplotlib.pyplot as plt

        files = self.output_files
        # simulated data
        t, y, v = [], [], []
        for sd, arrays in iter_output(files):
            beam = arrays['beam']
            t.append(sd['t'])
            y.append(beam.y[0])
            v.append(beam.v[0])

        data = np.loadtxt('ffpw_y.csv', delimiter=',')
        ta = data[:, 0]
        ya = data[:, 1]
        plt.plot(ta, ya)
        plt.scatter(t, y)
        plt.savefig('t_vs_y.png')
        plt.clf()

        data = np.loadtxt('ffpw_v.csv', delimiter=',')
        ta = data[:, 0]
        va = data[:, 1]
        plt.plot(ta, va)
        plt.scatter(t, v)
        plt.savefig('t_vs_v.png')


if __name__ == '__main__':
    app = CantileverBeam()
    app.run()
