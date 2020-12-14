from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from compyle.api import declare
from math import sqrt, asin, sin, cos, log
import numpy as np
from pysph.sph.scheme import Scheme
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EulerIntegrator
from pysph.base.kernels import CubicSpline
from math import pi as M_PI


def cross_product(a=[1.0, 0.0], b=[1.0, 0.0], result=[0.0, 0.0]):
    """Cross product between two vectors (a x b)

    Parameters
    ----------

    a: list
    b: list
    result: list
    """
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]


def get_particle_array_bonded_dem_potyondy(constants=None, **props):
    """Return a particle array for a dem particles
    """
    dim = props.pop('dim', None)

    dem_props = [
        'wx', 'wy', 'wz', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'rad_s',
        'm_inverse', 'I_inverse', 'u0', 'v0', 'w0', 'wx0', 'wy0', 'wz0', 'x0',
        'y0', 'z0'
    ]

    dem_id = props.pop('dem_id', None)

    pa = get_particle_array(additional_props=dem_props, **props)

    pa.add_property('dem_id', type='int', data=dem_id)
    # create the array to save the tangential interaction particles
    # index and other variables
    if dim == 3:
        bc_limit = 30
    elif dim == 2 or dim is None:
        bc_limit = 6

    pa.add_constant('bc_limit', bc_limit)
    pa.add_property('bc_idx', stride=bc_limit, type='int')
    pa.bc_idx[:] = -1

    pa.add_property('bc_total_contacts', type='int')
    pa.add_property('bc_rest_len', stride=bc_limit)
    pa.add_property('bc_fn_x', stride=bc_limit)
    pa.add_property('bc_fn_y', stride=bc_limit)
    pa.add_property('bc_fn_z', stride=bc_limit)
    pa.add_property('bc_ft_x', stride=bc_limit)
    pa.add_property('bc_ft_y', stride=bc_limit)
    pa.add_property('bc_ft_z', stride=bc_limit)

    pa.add_property('bc_tort_x', stride=bc_limit)
    pa.add_property('bc_tort_y', stride=bc_limit)
    pa.add_property('bc_tort_z', stride=bc_limit)
    pa.add_property('bc_torn_x', stride=bc_limit)
    pa.add_property('bc_torn_y', stride=bc_limit)
    pa.add_property('bc_torn_z', stride=bc_limit)
    # pa.add_property('bc_ft0_x', stride=bc_limit)
    # pa.add_property('bc_ft0_y', stride=bc_limit)
    # pa.add_property('bc_ft0_z', stride=bc_limit)

    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'wx', 'wy', 'wz', 'm', 'pid', 'tag',
        'gid', 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'rad_s', 'dem_id'
    ])

    return pa


def make_accel_eval(equations, pa_arrays, dim):
    from pysph.tools.sph_evaluator import SPHEvaluator
    kernel = CubicSpline(dim=dim)
    seval = SPHEvaluator(arrays=pa_arrays, equations=equations, dim=dim,
                         kernel=kernel)
    return seval


def setup_bc_contacts(dim, pa, beta):
    eqs1 = [
        Group(equations=[
            SetupContactsBC(dest=pa.name, sources=[pa.name], beta=beta),
        ])
    ]
    arrays = [pa]
    a_eval = make_accel_eval(eqs1, arrays, dim)
    a_eval.evaluate()


class SetupContactsBC(Equation):
    def __init__(self, dest, sources, beta):
        self.beta = beta
        super(SetupContactsBC, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_bc_total_contacts, d_bc_idx, d_bc_limit,
             d_rad_s, d_bc_rest_len, s_rad_s, RIJ):
        p = declare('int')
        p = d_bc_limit[0] * d_idx + d_bc_total_contacts[d_idx]
        fac = RIJ / (d_rad_s[d_idx] + s_rad_s[s_idx])
        if RIJ > 1e-12:
            if 1. - self.beta < fac < 1 + self.beta:
                d_bc_idx[p] = s_idx
                d_bc_total_contacts[d_idx] += 1
                d_bc_rest_len[p] = RIJ


class PotyondyIPForce2D(Equation):
    def __init__(self, dest, sources):
        super(PotyondyIPForce2D, self).__init__(dest, sources)

    def initialize(self, d_idx, d_bc_total_contacts, d_rad_s, d_x, d_y,
                   d_bc_fn_x, d_bc_fn_y, d_bc_fn_z, d_bc_ft_x, d_bc_ft_y,
                   d_bc_ft_z, d_bc_tort_z, d_bc_limit, d_bc_idx, d_bc_rest_len,
                   d_fx, d_fy, d_torz, d_u, d_v, d_wz, d_young_mod,
                   d_shear_mod, dt):
        # find the force on d_idx due to sidx. As in the paper, d_idx is
        # particle b and sidx is paricle a.
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        xij = declare('matrix(3)')
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        for i in range(p, q1):
            sidx = d_bc_idx[i]
            elongation = -1.
            rij = 0.0

            xij[0] = d_x[d_idx] - d_x[sidx]
            xij[1] = d_y[d_idx] - d_y[sidx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])

            # normal vector from j to i (A to B)
            nji_x = xij[0] / rij
            nji_y = xij[1] / rij

            rinv = 1. / rij
            # print("didx")
            # print(d_idx)
            # print("sidx")
            # print(sidx)
            # print("hi")
            # print(d_bc_rest_len[i])
            # check the particles are not on top of each other.
            if rij > 1e-12:
                elongation = rij - d_bc_rest_len[i]

            # print("rij")
            # print(rij)
            # print("overlap")
            # print(overlap)

            # ====================
            # relative velocity of B (d_idx) w.r.t A (sidx)
            # ====================

            # velocity of particle B (d_idx) at contact point
            # vi_x = vi + nji \cross omega_i
            overlap = d_rad_s[d_idx] + d_rad_s[sidx] - rij

            vi_x = d_u[d_idx] + nji_y * d_wz[d_idx] * (d_rad_s[d_idx] -
                                                       overlap / 2.)
            vi_y = d_v[d_idx] + nji_x * d_wz[d_idx] * (d_rad_s[d_idx] -
                                                       overlap / 2.)

            # similarly velocity of particle A (sidx) at contact point
            # negative because the normal is passing the opposite way
            vj_x = d_u[sidx] - nji_y * d_wz[sidx] * (d_rad_s[sidx] -
                                                     overlap / 2.)

            vj_y = d_v[sidx] - nji_x * d_wz[sidx] * (d_rad_s[sidx] -
                                                     overlap / 2.)
            # the relative velocity of particle B (d_idx) w.r.t A (sidx) is
            vr_x = vi_x - vj_x
            vr_y = vi_y - vj_y
            # ====================
            # relative velocity of B (d_idx) w.r.t A (sidx) ends
            # ====================

            # normal velocity magnitude
            vr_dot_nij = vr_x * nji_x + vr_y * nji_y
            vn_x = vr_dot_nij * nji_x
            vn_y = vr_dot_nij * nji_y

            # ---------- force computation starts ------------
            # delta_nx = vn_x * dt
            # delta_ny = vn_y * dt

            # -------------------------------------------------
            # ------------------- normal force ----------------
            # -------------------------------------------------
            r_min = min(d_rad_s[d_idx], d_rad_s[sidx])
            area = 2. * r_min
            kn = d_young_mod[d_idx] * area / rij
            # alpha = 100.
            # d_bc_fn_x[i] += kn * delta_nx
            # d_bc_fn_y[i] += kn * delta_ny

            d_fx[d_idx] += -kn * elongation * nji_x
            d_fy[d_idx] += -kn * elongation * nji_y

            # tangential force
            # ft_x = d_bc_ft_x[i]
            # ft_y = d_bc_ft_y[i]

            # TODO
            # check the coulomb criterion

            # Add the tangential force
            # d_fx[d_idx] += ft_x
            # d_fy[d_idx] += ft_y

            # --------- Tangential force -----------
            # -----increment the tangential force---
            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y

            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

            # get the incremental force
            r_min = min(d_rad_s[d_idx], d_rad_s[sidx])
            area = 2. * r_min
            kt = d_shear_mod[d_idx] * area / rij
            # alpha = 100.
            d_bc_ft_x[i] += -kt * vt_x * dt
            d_bc_ft_y[i] += -kt * vt_y * dt

            d_fx[d_idx] += d_bc_ft_x[i]
            d_fy[d_idx] += d_bc_ft_y[i]

            # in 2d we only have one moment, and that is bending
            dtheta_z = (d_wz[d_idx] - d_wz[sidx]) * dt
            moi = 2. / 3. * r_min * r_min * r_min
            kn = d_young_mod[d_idx] * moi / rij
            d_bc_tort_z[i] -= kn * dtheta_z

            d_torz[d_idx] += d_bc_tort_z[i]

            # and also torque due to shear force
            d_torz[d_idx] -= (d_bc_ft_x[i] * nji_y - d_bc_ft_y[i] * nji_x) * (
                d_rad_s[d_idx] - overlap / 2.)


class PotyondyIPForce(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super(PotyondyIPForce, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [cross_product]

    def initialize(self, d_idx, d_bc_total_contacts, d_rad_s, d_x, d_y, d_z,
                   d_bc_fn_x, d_bc_fn_y, d_bc_fn_z, d_bc_ft_x, d_bc_ft_y,
                   d_bc_ft_z, d_bc_tort_x, d_bc_tort_y, d_bc_tort_z,
                   d_bc_torn_x, d_bc_torn_y, d_bc_torn_z, d_bc_limit, d_bc_idx,
                   d_bc_rest_len, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz,
                   d_u, d_v, d_w, d_wx, d_wy, d_wz, d_young_mod, d_shear_mod,
                   dt, t):
        # find the force on d_idx due to sidx. As in the paper, d_idx is
        # particle b and sidx is paricle a.
        p, q1, tot_ctcs, i, sidx = declare('int', 5)
        xij, nij, omega_i, omega_j, tmp_vec1, tmp_vec2, shear_force_i = declare('matrix(3)', 7)
        # total number of contacts of particle i in destination
        tot_ctcs = d_bc_total_contacts[d_idx]

        # d_idx has a range of tracking indices with sources
        # starting index is p
        p = d_idx * d_bc_limit[0]
        # ending index is q -1
        q1 = p + tot_ctcs

        for i in range(p, q1):
            # print(i)
            sidx = d_bc_idx[i]
            elongation = -1.
            rij = 0.0

            xij[0] = d_x[d_idx] - d_x[sidx]
            xij[1] = d_y[d_idx] - d_y[sidx]
            xij[2] = d_z[d_idx] - d_z[sidx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])

            # normal vector from j to i
            nij[0] = xij[0] / rij
            nij[1] = xij[1] / rij
            nij[2] = xij[2] / rij

            rinv = 1. / rij
            # print("didx")
            # print(d_idx)
            # print("sidx")
            # print(sidx)
            # print("hi")
            # print(d_bc_rest_len[i])
            # check the particles are not on top of each other.
            if rij > 1e-12:
                elongation = rij - d_bc_rest_len[i]

            # print("rij")
            # print(rij)
            # print("overlap")
            # print(overlap)

            # ====================
            # relative velocity of B (d_idx) w.r.t A (sidx)
            # ====================

            # velocity of particle B (d_idx) at contact point
            # vi_x = vi + nji \cross omega_i
            overlap = d_rad_s[d_idx] + d_rad_s[sidx] - rij

            # ==============================================
            # velocity of particle d_idx at contact point is
            omega_i[0] = d_wx[d_idx]
            omega_i[1] = d_wy[d_idx]
            omega_i[2] = d_wz[d_idx]

            omega_j[0] = d_wx[sidx]
            omega_j[1] = d_wy[sidx]
            omega_j[2] = d_wz[sidx]

            cross_product(nij, omega_i, tmp_vec1)

            a_i = (d_rad_s[d_idx] - overlap / 2.)
            vi_x = d_u[d_idx] + tmp_vec1[0] * a_i
            vi_y = d_v[d_idx] + tmp_vec1[1] * a_i
            vi_z = d_w[d_idx] + tmp_vec1[2] * a_i

            cross_product(omega_j, nij, tmp_vec2)
            a_j = (d_rad_s[sidx] - overlap / 2.)
            vj_x = d_u[sidx] + tmp_vec2[0] * a_j
            vj_y = d_v[sidx] + tmp_vec2[1] * a_j
            vj_z = d_w[sidx] + tmp_vec2[2] * a_j

            # the relative velocity of particle B (d_idx) w.r.t A (sidx) is
            vr_x = vi_x - vj_x
            vr_y = vi_y - vj_y
            vr_z = vi_z - vj_z
            # ====================
            # relative velocity of B (d_idx) w.r.t A (sidx) ends
            # ====================

            # normal velocity magnitude
            vr_dot_nij = vr_x * nij[0] + vr_y * nij[1] + vr_z * nij[2]
            vn_x = vr_dot_nij * nij[0]
            vn_y = vr_dot_nij * nij[1]
            vn_z = vr_dot_nij * nij[2]

            # ---------- force computation starts ------------
            # delta_nx = vn_x * dt
            # delta_ny = vn_y * dt

            # -------------------------------------------------
            # ------------------- normal force ----------------
            # -------------------------------------------------
            r_min = min(d_rad_s[d_idx], d_rad_s[sidx])

            area = 2. * r_min
            if self.dim == 3:
                area = M_PI * r_min * r_min

            kn = d_young_mod[d_idx] * area / rij

            d_bc_fn_x[i] = -kn * elongation * nij[0]
            d_bc_fn_y[i] = -kn * elongation * nij[1]
            d_bc_fn_z[i] = -kn * elongation * nij[2]
            # if t > 0.3 - dt and t < 0.3 + dt:
            #     print(d_bc_fn_x[i])
            #     print(d_bc_fn_y[i])
            #     print(d_bc_fn_z[i])

            d_fx[d_idx] += -kn * elongation * nij[0]
            d_fy[d_idx] += -kn * elongation * nij[1]
            d_fz[d_idx] += -kn * elongation * nij[2]

            # --------- Tangential force -----------
            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z

            kt = d_shear_mod[d_idx] * area / rij
            d_bc_ft_x[i] += -kt * vt_x * dt
            d_bc_ft_y[i] += -kt * vt_y * dt
            d_bc_ft_z[i] += -kt * vt_z * dt

            d_fx[d_idx] += d_bc_ft_x[i]
            d_fy[d_idx] += d_bc_ft_y[i]
            d_fz[d_idx] += d_bc_ft_z[i]

            # ==============================================
            # Bending moment and twisting moment contributions
            # ==============================================
            omegar_x = omega_i[0] - omega_j[0]
            omegar_y = omega_i[1] - omega_j[1]
            omegar_z = omega_i[2] - omega_j[2]

            omegar_dot_nij = (omegar_x * nij[0] + omegar_y * nij[1] +
                              omegar_z * nij[2])
            omegan_x = omegar_dot_nij * nij[0]
            omegan_y = omegar_dot_nij * nij[1]
            omegan_z = omegar_dot_nij * nij[2]

            omegat_x = omegar_x - omegan_x
            omegat_y = omegar_y - omegan_y
            omegat_z = omegar_z - omegan_z

            dthetan_x = omegan_x * dt
            dthetan_y = omegan_y * dt
            dthetan_z = omegan_z * dt

            dthetat_x = omegat_x * dt
            dthetat_y = omegat_y * dt
            dthetat_z = omegat_z * dt

            polar_moi = 2. / 3. * r_min * r_min * r_min
            kn = d_young_mod[d_idx] * polar_moi / rij

            d_bc_torn_x[i] -= kn * dthetan_x
            d_bc_torn_y[i] -= kn * dthetan_y
            d_bc_torn_z[i] -= kn * dthetan_z

            area_moi = 2. / 3. * r_min * r_min * r_min
            kt = d_young_mod[d_idx] * area_moi / rij

            d_bc_tort_x[i] -= kt * dthetat_x
            d_bc_tort_y[i] -= kt * dthetat_y
            d_bc_tort_z[i] -= kt * dthetat_z

            d_torx[d_idx] += d_bc_torn_x[i] + d_bc_tort_x[i]
            d_tory[d_idx] += d_bc_torn_y[i] + d_bc_tort_y[i]
            d_torz[d_idx] += d_bc_torn_z[i] + d_bc_tort_z[i]

            # and also torque due to shear force
            # use cross product again
            shear_force_i[0] = d_bc_ft_x[i]
            shear_force_i[1] = d_bc_ft_y[i]
            shear_force_i[2] = d_bc_ft_z[i]
            cross_product(shear_force_i, nij, tmp_vec1)

            a_i = (d_rad_s[d_idx] - overlap / 2.)
            d_torx[d_idx] += tmp_vec1[0] * a_i
            d_tory[d_idx] += tmp_vec1[1] * a_i
            d_torz[d_idx] += tmp_vec1[2] * a_i


class GlobalDamping(Equation):
    def initialize(self, d_idx, d_m, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz,
                   d_u, d_v, d_w, d_wx, d_wy, d_wz):
        # coefficient of global damping is
        alpha = 10.
        C_gd = alpha * d_m[d_idx]

        d_fx[d_idx] -= C_gd * d_u[d_idx]
        d_fy[d_idx] -= C_gd * d_v[d_idx]
        d_fz[d_idx] -= C_gd * d_w[d_idx]

        d_torx[d_idx] -= C_gd * d_wx[d_idx]
        d_tory[d_idx] -= C_gd * d_wy[d_idx]
        d_torz[d_idx] -= C_gd * d_wz[d_idx]


class LeapFrogStepPotyondy(IntegratorStep):
    def stage1(self, d_idx, d_u, d_v, d_w, d_fx, d_fy, d_fz, d_torx, d_tory,
               d_torz, d_wx, d_wy, d_wz, d_m_inverse, d_I_inverse, dt):
        dtb2 = dt / 2.

        d_u[d_idx] += dtb2 * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] += dtb2 * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] += dtb2 * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] += dtb2 * d_torx[d_idx] * d_I_inverse[d_idx]
        d_wy[d_idx] += dtb2 * d_tory[d_idx] * d_I_inverse[d_idx]
        d_wz[d_idx] += dtb2 * d_torz[d_idx] * d_I_inverse[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_fx, d_fy, d_fz,
               d_x0, d_y0, d_z0, d_u0, d_v0, d_w0, d_wx0, d_wy0, d_wz0, d_torx,
               d_tory, d_torz, d_wx, d_wy, d_wz, d_m_inverse, d_I_inverse, dt):
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_fx, d_fy, d_fz, d_torx, d_tory,
               d_torz, d_wx, d_wy, d_wz, d_m_inverse, d_I_inverse, dt):
        dtb2 = dt / 2.

        d_u[d_idx] += dtb2 * d_fx[d_idx] * d_m_inverse[d_idx]
        d_v[d_idx] += dtb2 * d_fy[d_idx] * d_m_inverse[d_idx]
        d_w[d_idx] += dtb2 * d_fz[d_idx] * d_m_inverse[d_idx]

        d_wx[d_idx] += dtb2 * d_torx[d_idx] * d_I_inverse[d_idx]
        d_wy[d_idx] += dtb2 * d_tory[d_idx] * d_I_inverse[d_idx]
        d_wz[d_idx] += dtb2 * d_torz[d_idx] * d_I_inverse[d_idx]


class ResetForceAndMoment(Equation):
    def initialize(self, d_idx, d_x, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz,
                   d_zero_force_idx, d_zero_moment_idx):
        if d_zero_force_idx[d_idx] == 1:
            d_fx[d_idx] = 0.0
            d_fy[d_idx] = 0.0
            d_fz[d_idx] = 0.0

        if d_zero_moment_idx[d_idx] == 1:
            d_torx[d_idx] = 0.0
            d_tory[d_idx] = 0.0
            d_torz[d_idx] = 0.0


class MakeForcesTorqueZero(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz, d_torx, d_tory, d_torz):
        d_fx[d_idx] = 0.0
        d_fy[d_idx] = 0.0
        d_fz[d_idx] = 0.0

        d_torx[d_idx] = 0.0
        d_tory[d_idx] = 0.0
        d_torz[d_idx] = 0.0
