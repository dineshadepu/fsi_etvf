import numpy as np
from pysph.sph.equation import Equation

from numpy import sqrt, log


def set_total_mass(pa):
    # left limit of body i
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.total_mass[i] = np.sum(pa.m[fltr])
        assert pa.total_mass[i] > 0., "Total mass has to be greater than zero"


def set_center_of_mass(pa):
    # loop over all the bodies
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.xcm[3 * i] = np.sum(pa.m[fltr] * pa.x[fltr]) / pa.total_mass[i]
        pa.xcm[3 * i + 1] = np.sum(pa.m[fltr] * pa.y[fltr]) / pa.total_mass[i]
        pa.xcm[3 * i + 2] = np.sum(pa.m[fltr] * pa.z[fltr]) / pa.total_mass[i]


def set_moment_of_inertia_izz(pa):
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        izz = np.sum(pa.m[fltr] * ((pa.x[fltr] - pa.xcm[3 * i])**2. +
                                   (pa.y[fltr] - pa.xcm[3 * i + 1])**2.))
        pa.izz[i] = izz


def set_moment_of_inertia_and_its_inverse(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    # no of bodies
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.xcm[3 * i:3 * i + 3]

        I = np.zeros(9)
        for j in fltr:
            # Ixx
            I[0] += pa.m[j] * ((pa.y[j] - cm_i[1])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.y[j] - cm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
        pa.inertia_tensor_body_frame[9 * i:9 * i + 9] = I[:]

        I_inv = np.linalg.inv(I.reshape(3, 3))
        I_inv = I_inv.ravel()
        pa.inertia_tensor_inverse_body_frame[9 * i:9 * i + 9] = I_inv[:]

        # set the moment of inertia inverse in global frame
        # NOTE: This will be only computed once to compute the angular
        # momentum in the beginning.
        pa.inertia_tensor_global_frame[9 * i:9 * i + 9] = I[:]
        # set the moment of inertia inverse in global frame
        pa.inertia_tensor_inverse_global_frame[9 * i:9 * i + 9] = I_inv[:]


def set_body_frame_position_vectors(pa):
    """Save the position vectors w.r.t body frame"""
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.xcm[3 * i:3 * i + 3]
        for j in fltr:
            pa.dx0[j] = pa.x[j] - cm_i[0]
            pa.dy0[j] = pa.y[j] - cm_i[1]
            pa.dz0[j] = pa.z[j] - cm_i[2]


def set_body_frame_normal_vectors(pa):
    """Save the normal vectors w.r.t body frame"""
    pa.normal0[:] = pa.normal[:]


class BodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy, d_fz):
        d_fx[d_idx] = d_m[d_idx] * self.gx
        d_fy[d_idx] = d_m[d_idx] * self.gy
        d_fz[d_idx] = d_m[d_idx] * self.gz


class SumUpExternalForces(Equation):
    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        xcm = declare('object')
        body_id = declare('object')
        j = declare('int')
        i = declare('int')
        i3 = declare('int')

        frc = dst.force
        trq = dst.torque
        fx = dst.fx
        fy = dst.fy
        fz = dst.fz
        x = dst.x
        y = dst.y
        z = dst.z
        xcm = dst.xcm
        body_id = dst.body_id

        frc[:] = 0
        trq[:] = 0

        for j in range(len(x)):
            i = body_id[j]
            i3 = 3 * i
            frc[i3] += fx[j]
            frc[i3 + 1] += fy[j]
            frc[i3 + 2] += fz[j]

            # torque due to force on particle i
            # (r_i - com) \cross f_i
            dx = x[j] - xcm[i3]
            dy = y[j] - xcm[i3 + 1]
            dz = z[j] - xcm[i3 + 2]

            # torque due to force on particle i
            # dri \cross fi
            trq[i3] += (dy * fz[j] - dz * fy[j])
            trq[i3 + 1] += (dz * fx[j] - dx * fz[j])
            trq[i3 + 2] += (dx * fy[j] - dy * fx[j])


def normalize_R_orientation(orien):
    a1 = np.array([orien[0], orien[3], orien[6]])
    a2 = np.array([orien[1], orien[4], orien[7]])
    a3 = np.array([orien[2], orien[5], orien[8]])
    # norm of col0
    na1 = np.linalg.norm(a1)

    b1 = a1 / na1

    b2 = a2 - np.dot(b1, a2) * b1
    nb2 = np.linalg.norm(b2)
    b2 = b2 / nb2

    b3 = a3 - np.dot(b1, a3) * b1 - np.dot(b2, a3) * b2
    nb3 = np.linalg.norm(b3)
    b3 = b3 / nb3

    orien[0] = b1[0]
    orien[3] = b1[1]
    orien[6] = b1[2]
    orien[1] = b2[0]
    orien[4] = b2[1]
    orien[7] = b2[2]
    orien[2] = b3[0]
    orien[5] = b3[1]
    orien[8] = b3[2]


class RigidBodyLVC(Equation):
    """
    linearViscoelasticContactModelWithCoulombFriction
    """
    def __init__(self, dest, sources, kn=1e5, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(RigidBodyLVC, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_u, d_v, d_w, d_fx, d_fy, d_fz, d_tng_idx,
             d_tng_idx_dem_id, d_tng_fx, d_tng_fy, d_tng_fz,
             d_total_tng_contacts, d_dem_id, d_max_tng_contacts_limit,
             XIJ, RIJ, d_rad_s, s_idx, s_m, s_u, s_v, s_w,
             s_rad_s, s_dem_id, dt, t):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        overlap = -1.

        if d_dem_id[d_idx] != s_dem_id[s_idx]:
            # check the particles are not on top of each other.
            if RIJ > 0:
                overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

            # ---------- force computation starts ------------
            # if particles are overlapping
            if overlap > 0:
                # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
                rinv = 1.0 / RIJ
                # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
                nx = -XIJ[0] * rinv
                ny = -XIJ[1] * rinv
                nz = -XIJ[2] * rinv

                # Now the relative velocity of particle j w.r.t i at the contact
                # point is
                vr_x = s_u[s_idx] - d_u[d_idx]
                vr_y = s_v[s_idx] - d_v[d_idx]
                vr_z = s_w[s_idx] - d_w[d_idx]

                # normal velocity magnitude
                vr_dot_nij = vr_x * nx + vr_y * ny + vr_z * nz
                vn_x = vr_dot_nij * nx
                vn_y = vr_dot_nij * ny
                vn_z = vr_dot_nij * nz

                # tangential velocity
                vt_x = vr_x - vn_x
                vt_y = vr_y - vn_y
                vt_z = vr_z - vn_z
                # magnitude of the tangential velocity
                # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

                m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
                eta_n = self.alpha * sqrt(m_eff)

                ############################
                # normal force computation #
                ############################
                kn_overlap = self.kn * overlap
                fn_x = -kn_overlap * nx + eta_n * vn_x
                fn_y = -kn_overlap * ny + eta_n * vn_y
                fn_z = -kn_overlap * nz + eta_n * vn_z

                #################################
                # tangential force computation  #
                #################################
                # total number of contacts of particle i in destination
                tot_ctcs = d_total_tng_contacts[d_idx]

                # d_idx has a range of tracking indices with sources
                # starting index is p
                p = d_idx * d_max_tng_contacts_limit[0]
                # ending index is q -1
                q1 = p + tot_ctcs

                # check if the particle is in the tracking list
                # if so, then save the location at found_at
                found = 0
                for j in range(p, q1):
                    if s_idx == d_tng_idx[j]:
                        if s_dem_id[s_idx] == d_tng_idx_dem_id[j]:
                            found_at = j
                            found = 1
                            break
                # if the particle is not been tracked then assign an index in
                # tracking history.
                # ft_x = 0.
                # ft_y = 0.
                # ft_z = 0.

                if found == 0:
                    found_at = q1
                    d_tng_idx[found_at] = s_idx
                    d_total_tng_contacts[d_idx] += 1
                    d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]

                # # implies we are tracking the particle
                # else:
                #     ####################################################
                #     # rotate the tangential force to the current plane #
                #     ####################################################
                #     ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2. +
                #                d_tng_fz[found_at]**2.)**0.5
                #     ft_dot_nij = (d_tng_fx[found_at] * nx +
                #                   d_tng_fy[found_at] * ny +
                #                   d_tng_fz[found_at] * nz)
                #     # tangential force projected onto the current normal of the
                #     # contact place
                #     ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
                #     ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
                #     ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

                #     ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
                #     if ftp_magn > 0:
                #         one_by_ftp_magn = 1. / ftp_magn

                #         tx = ft_px * one_by_ftp_magn
                #         ty = ft_py * one_by_ftp_magn
                #         tz = ft_pz * one_by_ftp_magn
                #     else:
                #         if vt_magn > 0.:
                #             tx = -vt_x / vt_magn
                #             ty = -vt_y / vt_magn
                #             tz = -vt_z / vt_magn
                #         else:
                #             tx = 0.
                #             ty = 0.
                #             tz = 0.

                #     # rescale the projection by the magnitude of the
                #     # previous tangential force, which gives the tangential
                #     # force on the current plane
                #     ft_x = ft_magn * tx
                #     ft_y = ft_magn * ty
                #     ft_z = ft_magn * tz

                #     # (*) check against Coulomb criterion
                #     # Tangential force magnitude due to displacement
                #     ftr_magn = (ft_x * ft_x + ft_y * ft_y + ft_z * ft_z)**(0.5)
                #     fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

                #     # we have to compare with static friction, so
                #     # this mu has to be static friction coefficient
                #     fn_mu = self.mu * fn_magn

                #     if ftr_magn >= fn_mu:
                #         # rescale the tangential displacement
                #         d_tng_fx[found_at] = fn_mu * tx
                #         d_tng_fy[found_at] = fn_mu * ty
                #         d_tng_fz[found_at] = fn_mu * tz

                #         # set the tangential force to static friction
                #         # from Coulomb criterion
                #         ft_x = fn_mu * tx
                #         ft_y = fn_mu * ty
                #         ft_z = fn_mu * tz

                d_tng_fx[found_at] -= self.kt * vt_x * dt
                d_tng_fy[found_at] -= self.kt * vt_y * dt
                d_tng_fz[found_at] -= self.kt * vt_z * dt

                # check for Coloumb friction
                fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)
                fn_mu = self.mu * fn_magn

                ft_magn = (d_tng_fx[found_at] * d_tng_fx[found_at] +
                           d_tng_fy[found_at] * d_tng_fy[found_at] +
                           d_tng_fz[found_at] * d_tng_fz[found_at])

                if ft_magn >= fn_magn:
                    d_tng_fx[found_at] = fn_mu * d_tng_fx[found_at] / ft_magn
                    d_tng_fy[found_at] = fn_mu * d_tng_fy[found_at] / ft_magn
                    d_tng_fz[found_at] = fn_mu * d_tng_fz[found_at] / ft_magn

                d_fx[d_idx] += fn_x + d_tng_fx[found_at]
                d_fy[d_idx] += fn_y + d_tng_fy[found_at]
                d_fz[d_idx] += fn_z + d_tng_fz[found_at]

                # # torque = n cross F
                # d_torx[d_idx] += (ny * d_tng_fz[found_at] -
                #                   nz * d_tng_fy[found_at]) * a_i
                # d_tory[d_idx] += (nz * d_tng_fx[found_at] -
                #                   nx * d_tng_fz[found_at]) * a_i
                # d_torz[d_idx] += (nx * d_tng_fy[found_at] -
                #                   ny * d_tng_fx[found_at]) * a_i
