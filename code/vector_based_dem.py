"""Basic Equations for vector based DEM for solid elastic materials
###################################

References
----------

1. Owen, Benjamin, et al. "Vector-based discrete element method for solid
elastic materials." Computer Physics Communications 254 (2020): 107353.

"""

from numpy import sqrt, fabs
import numpy
from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from boundary_particles import (ComputeNormalsEDAC, SmoothNormalsEDAC,
                                IdentifyBoundaryParticleCosAngleEDAC)

from pysph.sph.wc.transport_velocity import SetWallVelocity

from pysph.examples.solid_mech.impact import add_properties
from boundary_particles import (ComputeNormals, SmoothNormals,
                                IdentifyBoundaryParticleCosAngleEDAC)

from pysph.sph.integrator import Integrator

import numpy as np
from math import sqrt, acos, sin, cos
from math import pi as M_PI


class SetContactsVectorDEM(Equation):
    def loop(self, d_idx, d_x, d_y, d_rad, d_cnt_idxs, d_cnt_limits, d_tot_cnts,
             d_no_bonds_limits, d_bc_l0, s_idx, s_x, s_y, s_rad, RIJ, XIJ,
             d_bc_normal_x, d_bc_normal_y,
             d_bc_normal_z, d_bc_normal_x0, d_bc_normal_y0, d_bc_normal_z0,
             d_bc_normal_contact_x, d_bc_normal_contact_y,
             d_bc_normal_contact_z, d_bc_normal_contact_x0,
             d_bc_normal_contact_y0, d_bc_normal_contact_z0,
             d_bc_B1, d_bc_B2, d_bc_B3, d_bc_E, d_bc_nu, d_bc_moi,
             d_rad_s):
        i = declare('int')
        if d_idx != s_idx:
            if RIJ < 2.5 * d_rad_s[d_idx]:
                # add the contact index at the end of the list
                i = d_idx * d_no_bonds_limits[0] + d_tot_cnts[d_idx]
                d_cnt_idxs[i] = s_idx
                d_bc_l0[i] = RIJ

                # increment the total number of contacts
                d_tot_cnts[d_idx] += 1

                # set the no of bonds limit
                d_cnt_limits[2 * d_idx] = d_idx * d_no_bonds_limits[0]
                d_cnt_limits[2 * d_idx + 1] = (
                    d_idx * d_no_bonds_limits[0] + d_tot_cnts[d_idx])

                # set the normals of the bonds
                ni_x = - XIJ[0] / RIJ
                ni_y = - XIJ[1] / RIJ
                ni_z = - XIJ[2] / RIJ

                d_bc_normal_x[i] = ni_x
                d_bc_normal_y[i] = ni_y
                d_bc_normal_z[i] = ni_z

                d_bc_normal_x0[i] = ni_x
                d_bc_normal_y0[i] = ni_y
                d_bc_normal_z0[i] = ni_z

                d_bc_normal_contact_x[i] = -ni_x
                d_bc_normal_contact_y[i] = -ni_y
                d_bc_normal_contact_z[i] = -ni_z

                d_bc_normal_contact_x0[i] = -ni_x
                d_bc_normal_contact_y0[i] = -ni_y
                d_bc_normal_contact_z0[i] = -ni_z

                # compute the bond moment of inertia
                # https://doi.org/10.1016/j.compfluid.2018.11.024
                b = 2. * d_rad_s[d_idx]
                d_bc_moi[i] = b**3. / 12

                # compute stiffness constants
                tmp1 = d_bc_E[i] / sqrt(3)
                CA = tmp1 / (1. - d_bc_nu[i])
                CD = tmp1 * (1. - 3. * d_bc_nu[i]) / (1. - d_bc_nu[i]**2.)
                CB = d_bc_E[i] * d_bc_moi[i] / RIJ

                d_bc_B1[i] = CA
                d_bc_B2[i] = CD * RIJ**2.
                d_bc_B3[i] = CB - d_bc_B2[i] / 4.


class UpdateBondOrientation(Equation):
    def initialize(self, d_idx, d_x, d_y, d_u, d_v, d_wz, d_rad, d_cnt_idxs,
                   d_cnt_limits, d_tot_cnts, d_no_bonds_limits, d_bc_l0,
                   d_fx, d_fy,
                   d_bc_normal_x, d_bc_normal_y,
                   d_bc_normal_contact_x, d_bc_normal_contact_y,
                   d_bc_normal_x0, d_bc_normal_y0,
                   d_bc_normal_contact_x0, d_bc_normal_contact_y0,
                   d_bc_E, d_bc_nu, d_torz, d_rotation_mat, dt):
        i, p, q, sidx = declare('int')
        # particle d_idx has its neighbours information in d_cnt_idxs
        # The range of such is
        p = d_cnt_limits[2 * d_idx]
        q = d_cnt_limits[2 * d_idx + 1]

        # now loop over the neighbours and find the force on particle d_idx
        for i in range(p, q):
            sidx = d_cnt_idxs[i]

            # get the unit normal from particle d_cnt_idxs[i] to particle d_idx
            ni_x0 = d_bc_normal_x0[i]
            ni_y0 = d_bc_normal_y0[i]
            nj_x0 = d_bc_normal_contact_x0[i]
            nj_y0 = d_bc_normal_contact_y0[i]

            Ri_00 = d_rotation_mat[4 * d_idx]
            Ri_01 = d_rotation_mat[4 * d_idx + 1]
            Ri_10 = d_rotation_mat[4 * d_idx + 2]
            Ri_11 = d_rotation_mat[4 * d_idx + 3]

            Rj_00 = d_rotation_mat[4 * sidx]
            Rj_01 = d_rotation_mat[4 * sidx + 1]
            Rj_10 = d_rotation_mat[4 * sidx + 2]
            Rj_11 = d_rotation_mat[4 * sidx + 3]

            d_bc_normal_x[i] = Ri_00 * ni_x0 + Ri_01 * ni_y0
            d_bc_normal_y[i] = Ri_10 * ni_x0 + Ri_11 * ni_y0
            d_bc_normal_contact_x[i] = Rj_00 * nj_x0 + Rj_01 * nj_y0
            d_bc_normal_contact_y[i] = Rj_10 * nj_x0 + Rj_11 * nj_y0


class ComputeForceAndMomentsDueToBonds(Equation):
    def initialize(self, d_idx, d_x, d_y, d_u, d_v, d_wz, d_rad, d_cnt_idxs,
                   d_cnt_limits, d_tot_cnts, d_no_bonds_limits, d_bc_l0, d_fx,
                   d_fy, d_bc_normal_x, d_bc_normal_y, d_bc_normal_contact_x,
                   d_bc_normal_contact_y, d_bc_moi, d_bc_B1, d_bc_B2, d_bc_B3,
                   d_torz, dt):
        i, p, q, sidx = declare('int')
        # particle d_idx has its neighbours information in d_cnt_idxs
        # The range of such is
        p = d_cnt_limits[2 * d_idx]
        q = d_cnt_limits[2 * d_idx + 1]

        # now loop over the neighbours and find the force on particle d_idx
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_torz[d_idx] = 0.

        for i in range(p, q):
            sidx = d_cnt_idxs[i]

            # distance between the particles
            dx = d_x[d_idx] - d_x[sidx]
            dy = d_y[d_idx] - d_y[sidx]
            rij = ((dx)**2. + (dy)**2.)**(0.5)
            # get the unit normal from particle d_cnt_idxs[i] to particle d_idx
            eij_x = - dx / rij
            eij_y = - dy / rij

            # bond orientation vectors
            ni_x = d_bc_normal_x[i]
            ni_y = d_bc_normal_y[i]
            nj_x = d_bc_normal_contact_x[i]
            nj_y = d_bc_normal_contact_y[i]

            # https://math.stackexchange.com/questions/621729/dot-product-between-vector-and-matrix
            eij_00 = eij_x * eij_x
            eij_01 = eij_x * eij_y
            eij_10 = eij_y * eij_x
            eij_11 = eij_y * eij_y

            # Equation 10 (1st term)
            a = d_bc_l0[i]

            B1 = d_bc_B1[i]
            B2 = d_bc_B2[i]
            B3 = d_bc_B3[i]

            fx = 0.
            fy = 0.

            # force equation 10 (first term)
            fx += B1 * (rij - a) * (-dx)
            fy += B1 * (rij - a) * (-dy)

            nji_x = (nj_x - ni_x)
            nji_y = (nj_y - ni_y)
            tmp_1 = nji_x * (1. - eij_00) + nji_y * eij_01
            tmp_2 = nji_x * eij_10 + nji_y * (1. - eij_11)
            tmp_3 = B2 / (2. * rij)

            fx += tmp_3 * tmp_1
            fy += tmp_3 * tmp_2

            eij_cross_ni = eij_x * ni_y - eij_y * ni_x
            nj_cross_ni = nj_x * ni_y - nj_y * ni_x
            torz = - B2 * 0.5 * eij_cross_ni + B3 * nj_cross_ni

            d_fx[d_idx] += fx
            d_fy[d_idx] += fy
            d_torz[d_idx] += torz


class AddGravityToStructure(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(AddGravityToStructure, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


class ApplyBoundaryConditions(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_torz, d_boundary_particle):
        if d_boundary_particle[d_idx] == 1:
            d_fx[d_idx] = 0.
            d_fy[d_idx] = 0.
            d_torz[d_idx] = 0.


class GTVFVectorBasedDEMStep(IntegratorStep):
    """This step follows GTVF paper by Zhang 2017"""
    def stage1(self, d_idx, d_m, d_moi, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_wz, d_torz, dt):
        # this should follow dem principles
        dtb2 = 0.5 * dt

        m_inverse = 1. / d_m[d_idx]
        d_u[d_idx] += dtb2 * d_fx[d_idx] * m_inverse
        d_v[d_idx] += dtb2 * d_fy[d_idx] * m_inverse
        d_w[d_idx] += dtb2 * d_fz[d_idx] * m_inverse

        I_inverse = 1. / d_moi[d_idx]
        d_wz[d_idx] += dtb2 * d_torz[d_idx] * I_inverse

    def stage2(self, d_idx, d_u, d_v, d_w, d_wz, d_x, d_y, d_z,
               d_theta, d_rotation_mat, dt):
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]

        d_theta[d_idx] += dt * d_wz[d_idx]

        d_rotation_mat[4 * d_idx + 0] = cos(d_theta[d_idx])
        d_rotation_mat[4 * d_idx + 1] = - sin(d_theta[d_idx])
        d_rotation_mat[4 * d_idx + 2] = sin(d_theta[d_idx])
        d_rotation_mat[4 * d_idx + 3] = cos(d_theta[d_idx])

    def stage3(self, d_idx, d_m, d_moi, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_wz, d_torz, dt):
        # this should follow dem principles
        dtb2 = 0.5 * dt

        m_inverse = 1. / d_m[d_idx]
        d_u[d_idx] += dtb2 * d_fx[d_idx] * m_inverse
        d_v[d_idx] += dtb2 * d_fy[d_idx] * m_inverse
        d_w[d_idx] += dtb2 * d_fz[d_idx] * m_inverse

        I_inverse = 1. / d_moi[d_idx]
        d_wz[d_idx] += dtb2 * d_torz[d_idx] * I_inverse


class VectorBasedDEM(Scheme):
    def __init__(self, solids, dim, gx=0., gy=0.):
        self.solids = solids

        self.dim = dim

        self.gx = gx
        self.gy = gy

        self.seven_disk_model = True
        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        add_bool_argument(
            group, 'seven-disk', dest='seven_disk_model',
            default=True,
            help='Use seven disk model')

    def consume_user_options(self, options):
        _vars = ['seven_disk_model']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        from pysph.sph.equation import Group

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        # no equations in stage 1
        stage1 = []
        # ------------------------
        # stage 2 equations starts
        # ------------------------

        stage2 = []
        # -------------------
        # Compute forces on the particle due to bonds
        # -------------------
        g4 = []
        for solid in self.solids:
            g4.append(
                UpdateBondOrientation(dest=solid, sources=[solid]))

            g4.append(
                ComputeForceAndMomentsDueToBonds(dest=solid, sources=[solid]))
        stage2.append(Group(g4))

        g9 = []
        for solid in self.solids:
            g9.append(AddGravityToStructure(dest=solid, sources=None,
                                            gx=self.gx,
                                            gy=self.gy,
                                            gz=0.))

            g9.append(ApplyBoundaryConditions(dest=solid, sources=None))

        stage2.append(Group(equations=g9))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.wc.gtvf import GTVFIntegrator

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator

        step_cls = GTVFVectorBasedDEMStep

        for name in self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def setup_properties(self, particles, clean=True):
        pas = dict([(p.name, p) for p in particles])

        for solid in self.solids:
            pa = pas[solid]

            pa.add_property('no_bonds_limits', type='int')
            pa.no_bonds_limits[0] = 8

            # Vector dem properties
            add_properties(pa, 'rad', 'wz', 'fx', 'fy', 'fz', 'torz', 'theta',
                           'boundary_particle')

            # if self.seven_disk_model is True:
            # contact indices
            pa.add_property('cnt_idxs', stride=pa.no_bonds_limits[0],
                            type='int')

            # distance between the particles at the initiation of the contacts
            pa.add_property('bc_l0', stride=pa.no_bonds_limits[0])
            # each particle contact limits
            pa.add_property('cnt_limits', stride=2, type='int')
            # each particle total number of contacts
            pa.add_property('tot_cnts', type='int')
            # each particle total number of contacts
            # this is the intial orientation
            pa.add_property('bc_normal_x0', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_y0', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_z0', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_x', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_y', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_z', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_contact_x0',
                            stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_contact_y0',
                            stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_contact_z0',
                            stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_contact_x',
                            stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_contact_y',
                            stride=pa.no_bonds_limits[0])
            pa.add_property('bc_normal_contact_z',
                            stride=pa.no_bonds_limits[0])

            # Material properties of the bonds
            pa.add_property('bc_E', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_nu', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_moi', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_B1', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_B2', stride=pa.no_bonds_limits[0])
            pa.add_property('bc_B3', stride=pa.no_bonds_limits[0])

            pa.bc_E[:] = pa.E[0]
            pa.bc_nu[:] = pa.nu[0]

            # rotation matrix per particle
            pa.add_property('rotation_mat', stride=4)

            for i in range(len(pa.x)):
                pa.rotation_mat[4 * i] = 1.
                pa.rotation_mat[4 * i + 1] = 0.
                pa.rotation_mat[4 * i + 2] = 1.
                pa.rotation_mat[4 * i + 3] = 0.

            pa.add_property('rotation_mat', stride=4)
            # set the contacts to default values
            pa.cnt_idxs[:] = -1
            pa.cnt_limits[:] = 0
            pa.tot_cnts[:] = 0
            pa.bc_normal_x[:] = 0.
            pa.bc_normal_y[:] = 0.
            pa.bc_normal_z[:] = 0.

            # set the bonds
            equations = [
                Group(
                    equations=[SetContactsVectorDEM(dest=pa.name,
                                                    sources=[pa.name])])
            ]

            sph_eval = SPHEvaluator(arrays=[pa], equations=equations, dim=2,
                                    kernel=CubicSpline(dim=2))

            sph_eval.evaluate(0.1, 0.1)

            # add output arrays
            pa.add_output_arrays(['fx', 'fy', 'torz'])

    def get_solver(self):
        return self.solver
