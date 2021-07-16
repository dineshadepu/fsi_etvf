"""
Basic Equations for Solid Mechanics
###################################

References
----------
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
from math import sqrt, acos
from math import pi as M_PI
from solid_mech_common import AddGravityToStructure


class ContinuityEquationUhat(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, d_uhat, d_vhat, d_what, s_idx, s_m, s_uhat,
             s_vhat, s_what, DWIJ, VIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vij[2] = d_what[d_idx] - s_what[s_idx]

        vijdotdwij = DWIJ[0] * vij[0] + DWIJ[1] * vij[1] + DWIJ[2] * vij[2]
        d_arho[d_idx] += s_m[s_idx] * vijdotdwij


class ContinuityEquationETVFCorrection(Equation):
    """
    This is the additional term arriving in the new ETVF continuity equation
    """
    def loop(self, d_idx, d_arho, d_rho, d_u, d_v, d_w, d_uhat, d_vhat, d_what,
             s_idx, s_rho, s_m, s_u, s_v, s_w, s_uhat, s_vhat, s_what, DWIJ):
        tmp0 = s_rho[s_idx] * (s_uhat[s_idx] - s_u[s_idx]) - d_rho[d_idx] * (
            d_uhat[d_idx] - d_u[d_idx])

        tmp1 = s_rho[s_idx] * (s_vhat[s_idx] - s_v[s_idx]) - d_rho[d_idx] * (
            d_vhat[d_idx] - d_v[d_idx])

        tmp2 = s_rho[s_idx] * (s_what[s_idx] - s_w[s_idx]) - d_rho[d_idx] * (
            d_what[d_idx] - d_w[d_idx])

        vijdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)

        d_arho[d_idx] += s_m[s_idx] / s_rho[s_idx] * vijdotdwij


class VelocityGradient2DUhat(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v10, d_v11):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0

    def loop(self, d_uhat, d_vhat, d_what, d_idx, s_idx, s_m, s_rho, d_v00,
             d_v01, d_v10, d_v11, s_uhat, s_vhat, s_what, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vij[2] = d_what[d_idx] - s_what[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]


class VelocityGradient3DUhat(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v02, d_v10, d_v11, d_v12,
                   d_v20, d_v21, d_v22):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v02[d_idx] = 0.0

        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0
        d_v12[d_idx] = 0.0

        d_v20[d_idx] = 0.0
        d_v21[d_idx] = 0.0
        d_v22[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_v00, d_v01, d_v02, d_v10, d_v11,
             d_v12, d_v20, d_v21, d_v22, d_uhat, d_vhat, d_what, s_uhat,
             s_vhat, s_what, DWIJ, VIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vij[2] = d_what[d_idx] - s_what[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]
        d_v02[d_idx] += tmp * -vij[0] * DWIJ[2]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]
        d_v12[d_idx] += tmp * -vij[1] * DWIJ[2]

        d_v20[d_idx] += tmp * -vij[2] * DWIJ[0]
        d_v21[d_idx] += tmp * -vij[2] * DWIJ[1]
        d_v22[d_idx] += tmp * -vij[2] * DWIJ[2]


class SetHIJForInsideParticles(Equation):
    def __init__(self, dest, sources, kernel_factor):
        # depends on the kernel used
        self.kernel_factor = kernel_factor
        super(SetHIJForInsideParticles, self).__init__(dest, sources)

    def initialize(self, d_idx, d_h_b, d_h):
        # back ground pressure h (This will be the usual h value)
        d_h_b[d_idx] = d_h[d_idx]

    def loop_all(self, d_idx, d_x, d_y, d_z, d_rho, d_h, d_is_boundary,
                 d_normal, d_normal_norm, d_h_b, s_m, s_x, s_y, s_z, s_h,
                 s_is_boundary, SPH_KERNEL, NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('int')
        xij = declare('matrix(3)')

        # if the particle is boundary set it's h_b to be zero
        if d_is_boundary[d_idx] == 1:
            d_h_b[d_idx] = 0.
        # if it is not the boundary then set its h_b according to the minimum
        # distance to the boundary particle
        else:
            # get the minimum distance to the boundary particle
            min_dist = 0
            for i in range(N_NBRS):
                s_idx = NBRS[i]

                if s_is_boundary[s_idx] == 1:
                    # find the distance
                    xij[0] = d_x[d_idx] - s_x[s_idx]
                    xij[1] = d_y[d_idx] - s_y[s_idx]
                    xij[2] = d_z[d_idx] - s_z[s_idx]
                    rij = sqrt(xij[0]**2. + xij[1]**2. + xij[2]**2.)

                    if rij > min_dist:
                        min_dist = rij

            # doing this out of desperation
            for i in range(N_NBRS):
                s_idx = NBRS[i]

                if s_is_boundary[s_idx] == 1:
                    # find the distance
                    xij[0] = d_x[d_idx] - s_x[s_idx]
                    xij[1] = d_y[d_idx] - s_y[s_idx]
                    xij[2] = d_z[d_idx] - s_z[s_idx]
                    rij = sqrt(xij[0]**2. + xij[1]**2. + xij[2]**2.)

                    if rij < min_dist:
                        min_dist = rij

            if min_dist > 0.:
                d_h_b[d_idx] = min_dist / self.kernel_factor + min_dist / 50


class MomentumEquationSolids(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_p, s_p, d_s00, d_s01,
             d_s02, d_s11, d_s12, d_s22, s_s00, s_s01, s_s02, s_s11, s_s12,
             s_s22, d_au, d_av, d_aw, WIJ, DWIJ):
        pa = d_p[d_idx]
        pb = s_p[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        rhoa21 = 1. / (rhoa * rhoa)
        rhob21 = 1. / (rhob * rhob)

        s00a = d_s00[d_idx]
        s01a = d_s01[d_idx]
        s02a = d_s02[d_idx]

        s10a = d_s01[d_idx]
        s11a = d_s11[d_idx]
        s12a = d_s12[d_idx]

        s20a = d_s02[d_idx]
        s21a = d_s12[d_idx]
        s22a = d_s22[d_idx]

        s00b = s_s00[s_idx]
        s01b = s_s01[s_idx]
        s02b = s_s02[s_idx]

        s10b = s_s01[s_idx]
        s11b = s_s11[s_idx]
        s12b = s_s12[s_idx]

        s20b = s_s02[s_idx]
        s21b = s_s12[s_idx]
        s22b = s_s22[s_idx]

        # Add pressure to the deviatoric components
        s00a = s00a - pa
        s00b = s00b - pb

        s11a = s11a - pa
        s11b = s11b - pb

        s22a = s22a - pa
        s22b = s22b - pb

        # compute accelerations
        mb = s_m[s_idx]

        d_au[d_idx] += (mb * (s00a * rhoa21 + s00b * rhob21) * DWIJ[0] + mb *
                        (s01a * rhoa21 + s01b * rhob21) * DWIJ[1] + mb *
                        (s02a * rhoa21 + s02b * rhob21) * DWIJ[2])

        d_av[d_idx] += (mb * (s10a * rhoa21 + s10b * rhob21) * DWIJ[0] + mb *
                        (s11a * rhoa21 + s11b * rhob21) * DWIJ[1] + mb *
                        (s12a * rhoa21 + s12b * rhob21) * DWIJ[2])

        d_aw[d_idx] += (mb * (s20a * rhoa21 + s20b * rhob21) * DWIJ[0] + mb *
                        (s21a * rhoa21 + s21b * rhob21) * DWIJ[1] + mb *
                        (s22a * rhoa21 + s22b * rhob21) * DWIJ[2])


class MonaghanArtificialStressCorrection(Equation):
    def loop(self, d_idx, s_idx, s_m, d_r00, d_r01, d_r02, d_r11, d_r12, d_r22,
             s_r00, s_r01, s_r02, s_r11, s_r12, s_r22, d_au, d_av, d_aw,
             d_wdeltap, d_n, WIJ, DWIJ):

        r00a = d_r00[d_idx]
        r01a = d_r01[d_idx]
        r02a = d_r02[d_idx]

        # r10a = d_r01[d_idx]
        r11a = d_r11[d_idx]
        r12a = d_r12[d_idx]

        # r20a = d_r02[d_idx]
        # r21a = d_r12[d_idx]
        r22a = d_r22[d_idx]

        r00b = s_r00[s_idx]
        r01b = s_r01[s_idx]
        r02b = s_r02[s_idx]

        # r10b = s_r01[s_idx]
        r11b = s_r11[s_idx]
        r12b = s_r12[s_idx]

        # r20b = s_r02[s_idx]
        # r21b = s_r12[s_idx]
        r22b = s_r22[s_idx]

        # compute the kernel correction term
        # if wdeltap is less than zero then no correction
        # needed
        if d_wdeltap[0] > 0.:
            fab = WIJ / d_wdeltap[0]
            fab = pow(fab, d_n[0])

            art_stress00 = fab * (r00a + r00b)
            art_stress01 = fab * (r01a + r01b)
            art_stress02 = fab * (r02a + r02b)

            art_stress10 = art_stress01
            art_stress11 = fab * (r11a + r11b)
            art_stress12 = fab * (r12a + r12b)

            art_stress20 = art_stress02
            art_stress21 = art_stress12
            art_stress22 = fab * (r22a + r22b)
        else:
            art_stress00 = 0.0
            art_stress01 = 0.0
            art_stress02 = 0.0

            art_stress10 = art_stress01
            art_stress11 = 0.0
            art_stress12 = 0.0

            art_stress20 = art_stress02
            art_stress21 = art_stress12
            art_stress22 = 0.0

        # compute accelerations
        mb = s_m[s_idx]

        d_au[d_idx] += mb * (art_stress00 * DWIJ[0] + art_stress01 * DWIJ[1] +
                             art_stress02 * DWIJ[2])

        d_av[d_idx] += mb * (art_stress10 * DWIJ[0] + art_stress11 * DWIJ[1] +
                             art_stress12 * DWIJ[2])

        d_aw[d_idx] += mb * (art_stress20 * DWIJ[0] + art_stress21 * DWIJ[1] +
                             art_stress22 * DWIJ[2])


class ComputeAuHatETVF(Equation):
    def __init__(self, dest, sources, pb):
        self.pb = pb
        super(ComputeAuHatETVF, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_m, d_h_b, d_auhat, d_avhat, d_awhat,
             WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ):
        dwijhat = declare('matrix(3)')

        rhoa = d_rho[d_idx]

        rhoa21 = 1. / (rhoa * rhoa)

        # add the background pressure acceleration
        if d_h_b[d_idx] > 0.:
            tmp = -self.pb * s_m[s_idx] * rhoa21
            SPH_KERNEL.gradient(XIJ, RIJ, d_h_b[d_idx], dwijhat)

            d_auhat[d_idx] += tmp * dwijhat[0]
            d_avhat[d_idx] += tmp * dwijhat[1]
            d_awhat[d_idx] += tmp * dwijhat[2]

    def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
                  d_awhat, d_is_boundary):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        if d_is_boundary[d_idx] == 1:
            # since it is boundary make its shifting acceleration zero
            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.


class ComputeAuHatETVFTangentialCorrection(Equation):
    """
    This equation computes the background pressure force for ETVF.

    FIXME: This is not compatible for variable smoothing lengths
    """
    def __init__(self, dest, sources, pb):
        self.pb = pb
        super(ComputeAuHatETVFTangentialCorrection,
              self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_m, d_h, d_auhat, d_avhat, d_awhat,
             WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ):
        dwijhat = declare('matrix(3)')

        rhoa = d_rho[d_idx]

        rhoa21 = 1. / (rhoa * rhoa)

        # add the background pressure acceleration
        tmp = -self.pb * s_m[s_idx] * rhoa21
        SPH_KERNEL.gradient(XIJ, RIJ, d_h[d_idx], dwijhat)

        d_auhat[d_idx] += tmp * dwijhat[0]
        d_avhat[d_idx] += tmp * dwijhat[1]
        d_awhat[d_idx] += tmp * dwijhat[2]

    def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
                  d_awhat, d_is_boundary):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        if d_h_b[d_idx] < d_h[d_idx]:
            if d_is_boundary[d_idx] == 1:
                # since it is boundary make its shifting acceleration zero
                d_auhat[d_idx] = 0.
                d_avhat[d_idx] = 0.
                d_awhat[d_idx] = 0.
            else:
                # implies this is a particle adjacent to boundary particle so
                # nullify the normal component
                au_dot_normal = (d_auhat[d_idx] * d_normal[idx3] +
                                 d_avhat[d_idx] * d_normal[idx3 + 1] +
                                 d_awhat[d_idx] * d_normal[idx3 + 2])

                # remove the normal acceleration component
                d_auhat[d_idx] -= au_dot_normal * d_normal[idx3]
                d_avhat[d_idx] -= au_dot_normal * d_normal[idx3 + 1]
                d_awhat[d_idx] -= au_dot_normal * d_normal[idx3 + 2]


class ComputeKappaSun2019PST(Equation):
    def __init__(self, dest, sources, limit_angle):
        self.limit_angle = limit_angle
        super(ComputeKappaSun2019PST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kappa):
        # back ground pressure h (This will be the usual h value)
        d_kappa[d_idx] = 1.

    def loop_all(self, d_idx, d_x, d_y, d_z, d_h, d_normal, s_normal, s_x, s_y,
                 s_z, s_normal_norm, d_h_b, d_kappa, NBRS, N_NBRS):
        i, didx3, sidx3 = declare('int', 3)
        s_idx = declare('int')

        didx3 = 3 * d_idx

        # if d_idx == 239:
        #     print(d_normal[didx3])
        #     print(d_normal[didx3 + 1])
        #     print(d_normal[didx3 + 2])

        max_angle = 0.

        if d_h_b[d_idx] < d_h[d_idx]:
            for i in range(N_NBRS):
                s_idx = NBRS[i]

                dx = d_x[d_idx] - s_x[s_idx]
                dy = d_y[d_idx] - s_y[s_idx]
                dz = d_z[d_idx] - s_z[s_idx]

                rij = (dx * dx + dy * dy + dz * dz)**0.5
                if rij > 1e-12:
                    if s_normal_norm[d_idx] == 1e-8:
                        sidx3 = 3 * s_idx

                        angle = acos(d_normal[didx3] * s_normal[sidx3] +
                                     d_normal[didx3 + 1] *
                                     s_normal[sidx3 + 1] +
                                     d_normal[didx3 + 2] *
                                     s_normal[sidx3 + 2]) * 180. / M_PI

                        if angle > max_angle:
                            max_angle = angle

                        # cos(15) (15 degrees)
                        if max_angle > self.limit_angle:
                            d_kappa[d_idx] = 0.
                            break
                        # find the angle between the normals
        else:
            d_kappa[d_idx] = 1.


class ComputeAuHatETVFSun2019(Equation):
    def __init__(self, dest, sources, mach_no, dim=2):
        self.mach_no = mach_no
        self.dim = dim
        super(ComputeAuHatETVFSun2019, self).__init__(dest, sources)

    def initialize(self, d_idx, d_div_r, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0
        d_div_r[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_h, d_auhat, d_avhat, d_awhat,
             d_div_r, d_c0_ref, d_wdeltap, d_n, WIJ, SPH_KERNEL, DWIJ, XIJ,
             RIJ, dt):
        fab = 0.
        # this value is directly taken from the paper
        R = 0.2

        if d_wdeltap[0] > 0.:
            fab = WIJ / d_wdeltap[0]
            fab = pow(fab, d_n[0])

        tmp = self.mach_no * d_c0_ref[0] * 2. * d_h[d_idx] / dt

        tmp1 = s_m[s_idx] / s_rho[s_idx]

        d_auhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[0]
        d_avhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[1]
        d_awhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[2]

        d_div_r[d_idx] -= tmp1 * (XIJ[0] * DWIJ[0] +
                                  XIJ[1] * DWIJ[1] +
                                  XIJ[2] * DWIJ[2])

    def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
                  d_awhat, d_div_r, d_is_boundary):
        """Save the auhat avhat awhat
        First we make all the particles with div_r < dim - 0.5 as zero.

        Now if the particle is a free surface particle and not a free particle,
        which identified through our normal code (d_h_b < d_h), we cut off the
        normal component

        """
        idx3 = declare('int')
        idx3 = 3 * d_idx

        auhat = d_auhat[d_idx]
        avhat = d_avhat[d_idx]
        awhat = d_awhat[d_idx]

        if d_div_r[d_idx] < self.dim - 0.5:
            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.

        # Now apply the filter for boundary particles and adjacent particles
        if d_h_b[d_idx] < d_h[d_idx]:
            if d_is_boundary[d_idx] == 1:
                # since it is boundary make its shifting acceleration zero
                d_auhat[d_idx] = 0.
                d_avhat[d_idx] = 0.
                d_awhat[d_idx] = 0.
            else:
                # implies this is a particle adjacent to boundary particle

                # check if the particle is going away from the continuum
                # or into the continuum
                au_dot_normal = (auhat * d_normal[idx3] +
                                 avhat * d_normal[idx3 + 1] +
                                 awhat * d_normal[idx3 + 2])

                # if it is going away from the continuum then nullify the
                # normal component.
                if au_dot_normal > 0.:
                    d_auhat[d_idx] = auhat - au_dot_normal * d_normal[idx3]
                    d_avhat[d_idx] = avhat - au_dot_normal * d_normal[idx3 + 1]
                    d_awhat[d_idx] = awhat - au_dot_normal * d_normal[idx3 + 2]


class ComputeAuHatETVFSun2019Solid(Equation):
    def __init__(self, dest, sources, mach_no):
        self.mach_no = mach_no
        super(ComputeAuHatETVFSun2019Solid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_h, d_auhat, d_avhat, d_awhat,
             d_c0_ref, d_wdeltap, d_n, WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ, dt):
        fab = 0.
        # this value is directly taken from the paper
        R = 0.2

        if d_wdeltap[0] > 0.:
            fab = WIJ / d_wdeltap[0]
            fab = pow(fab, d_n[0])

        tmp = self.mach_no * d_c0_ref[0] * 2. * d_h[d_idx] / dt
        # tmp = self.mach_no * d_c0_ref[0] * 2. * d_h[d_idx]

        tmp1 = s_m[s_idx] / s_rho[s_idx]

        d_auhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[0]
        d_avhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[1]
        d_awhat[d_idx] -= tmp * tmp1 * (1. + R * fab) * DWIJ[2]

    def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
                  d_awhat, d_is_boundary, dt):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        # first put a clearance
        magn_auhat = sqrt(d_auhat[d_idx] * d_auhat[d_idx] +
                          d_avhat[d_idx] * d_avhat[d_idx] +
                          d_awhat[d_idx] * d_awhat[d_idx])

        if magn_auhat > 1e-12:
            # tmp = min(magn_auhat, self.u_max * 0.5 / dt)
            # d_auhat[d_idx] = tmp * d_auhat[d_idx] / magn_auhat
            # d_avhat[d_idx] = tmp * d_avhat[d_idx] / magn_auhat
            # d_awhat[d_idx] = tmp * d_awhat[d_idx] / magn_auhat

            # Now apply the filter for boundary particles and adjacent particles
            if d_h_b[d_idx] < d_h[d_idx]:
                if d_is_boundary[d_idx] == 1:
                    # since it is boundary make its shifting acceleration zero
                    d_auhat[d_idx] = 0.
                    d_avhat[d_idx] = 0.
                    d_awhat[d_idx] = 0.
                else:
                    # implies this is a particle adjacent to boundary particle

                    # check if the particle is going away from the continuum
                    # or into the continuum
                    au_dot_normal = (d_auhat[d_idx] * d_normal[idx3] +
                                     d_avhat[d_idx] * d_normal[idx3 + 1] +
                                     d_awhat[d_idx] * d_normal[idx3 + 2])

                    # remove the normal acceleration component
                    if au_dot_normal > 0.:
                        d_auhat[d_idx] -= au_dot_normal * d_normal[idx3]
                        d_avhat[d_idx] -= au_dot_normal * d_normal[idx3 + 1]
                        d_awhat[d_idx] -= au_dot_normal * d_normal[idx3 + 2]
                    # d_auhat[d_idx] -= au_dot_normal * d_normal[idx3]
                    # d_avhat[d_idx] -= au_dot_normal * d_normal[idx3 + 1]
                    # d_awhat[d_idx] -= au_dot_normal * d_normal[idx3 + 2]


########################
# IPST equations start #
########################
def setup_ipst(pa, kernel):
    props = 'ipst_x ipst_y ipst_z ipst_dx ipst_dy ipst_dz'.split()

    for prop in props:
        pa.add_property(prop)

    pa.add_constant('ipst_chi0', 0.)
    pa.add_property('ipst_chi')

    equations = [
        Group(
            equations=[
                CheckUniformityIPST(dest=pa.name, sources=[pa.name]),
            ], real=False),
    ]

    sph_eval = SPHEvaluator(arrays=[pa], equations=equations, dim=2,
                            kernel=kernel(dim=2))

    sph_eval.evaluate(dt=0.1)

    pa.ipst_chi0[0] = min(pa.ipst_chi)


class SavePositionsIPSTBeforeMoving(Equation):
    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z):
        d_ipst_x[d_idx] = d_x[d_idx]
        d_ipst_y[d_idx] = d_y[d_idx]
        d_ipst_z[d_idx] = d_z[d_idx]


class AdjustPositionIPST(Equation):
    def __init__(self, dest, sources, u_max):
        self.u_max = u_max
        super(AdjustPositionIPST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ipst_dx, d_ipst_dy, d_ipst_dz):
        d_ipst_dx[d_idx] = 0.0
        d_ipst_dy[d_idx] = 0.0
        d_ipst_dz[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_h, d_ipst_dx, d_ipst_dy,
             d_ipst_dz, WIJ, XIJ, RIJ, dt):
        tmp = self.u_max * dt
        Vj = s_m[s_idx] / s_rho[s_idx]  # volume of j

        nij_x = 0.
        nij_y = 0.
        nij_z = 0.
        if RIJ > 1e-12:
            nij_x = XIJ[0] / RIJ
            nij_y = XIJ[1] / RIJ
            nij_z = XIJ[2] / RIJ

        d_ipst_dx[d_idx] += tmp * Vj * nij_x * WIJ
        d_ipst_dy[d_idx] += tmp * Vj * nij_y * WIJ
        d_ipst_dz[d_idx] += tmp * Vj * nij_z * WIJ

    def post_loop(self, d_idx, d_x, d_y, d_z, d_ipst_dx, d_ipst_dy, d_ipst_dz,
                  d_normal):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        # before adding the correction of position, cut off the normal component
        dr_dot_normal = (d_ipst_dx[d_idx] * d_normal[idx3] +
                         d_ipst_dy[d_idx] * d_normal[idx3 + 1] +
                         d_ipst_dz[d_idx] * d_normal[idx3 + 2])

        # if it is going away from the continuum then nullify the
        # normal component.
        if dr_dot_normal > 0.:
            # remove the normal acceleration component
            d_ipst_dx[d_idx] -= dr_dot_normal * d_normal[idx3]
            d_ipst_dy[d_idx] -= dr_dot_normal * d_normal[idx3 + 1]
            d_ipst_dz[d_idx] -= dr_dot_normal * d_normal[idx3 + 2]

        d_x[d_idx] = d_x[d_idx] + d_ipst_dx[d_idx]
        d_y[d_idx] = d_y[d_idx] + d_ipst_dy[d_idx]
        d_z[d_idx] = d_z[d_idx] + d_ipst_dz[d_idx]


class CheckUniformityIPST(Equation):
    """
    For this specific equation one has to update the NNPS

    """
    def __init__(self, dest, sources, tolerance=0.2, debug=False):
        self.inhomogenity = 0.0
        self.debug = debug
        self.tolerance = tolerance
        super(CheckUniformityIPST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ipst_chi):
        d_ipst_chi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_ipst_chi, d_h, WIJ, XIJ, RIJ,
             dt):
        d_ipst_chi[d_idx] += d_h[d_idx] * d_h[d_idx] * WIJ

    def reduce(self, dst, t, dt):
        chi_max = serial_reduce_array(dst.ipst_chi, 'min')
        self.inhomogenity = fabs(chi_max - dst.ipst_chi0[0])

    def converged(self):
        debug = self.debug
        inhomogenity = self.inhomogenity

        if inhomogenity > self.tolerance:
            if debug:
                print("Not converged:", inhomogenity)
            return -1.0
        else:
            if debug:
                print("Converged:", inhomogenity)
            return 1.0


class ComputeAuhatETVFIPST(Equation):
    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z,
                   d_auhat, d_avhat, d_awhat, dt):
        # FIXME: this 2 is specific to the integrator.
        dt_square_inv = 2. / (dt * dt)
        d_auhat[d_idx] = (d_x[d_idx] - d_ipst_x[d_idx]) * dt_square_inv
        d_avhat[d_idx] = (d_y[d_idx] - d_ipst_y[d_idx]) * dt_square_inv
        d_awhat[d_idx] = (d_z[d_idx] - d_ipst_z[d_idx]) * dt_square_inv

    # def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
    #               d_awhat, d_is_boundary):
    #     idx3 = declare('int')
    #     idx3 = 3 * d_idx

    #     # first put a clearance
    #     magn_auhat = sqrt(d_auhat[d_idx] * d_auhat[d_idx] +
    #                       d_avhat[d_idx] * d_avhat[d_idx] +
    #                       d_awhat[d_idx] * d_awhat[d_idx])

    #     if magn_auhat > 1e-12:
    #         # tmp = min(magn_auhat, self.u_max * 0.5)
    #         tmp = magn_auhat
    #         d_auhat[d_idx] = tmp * d_auhat[d_idx] / magn_auhat
    #         d_avhat[d_idx] = tmp * d_avhat[d_idx] / magn_auhat
    #         d_awhat[d_idx] = tmp * d_awhat[d_idx] / magn_auhat

    #         # Now apply the filter for boundary particles and adjacent particles
    #         if d_h_b[d_idx] < d_h[d_idx]:
    #             if d_is_boundary[d_idx] == 1:
    #                 # since it is boundary make its shifting acceleration zero
    #                 d_auhat[d_idx] = 0.
    #                 d_avhat[d_idx] = 0.
    #                 d_awhat[d_idx] = 0.
    #             else:
    #                 # implies this is a particle adjacent to boundary particle

    #                 # check if the particle is going away from the continuum
    #                 # or into the continuum
    #                 au_dot_normal = (d_auhat[d_idx] * d_normal[idx3] +
    #                                  d_avhat[d_idx] * d_normal[idx3 + 1] +
    #                                  d_awhat[d_idx] * d_normal[idx3 + 2])

    #                 # if it is going away from the continuum then nullify the
    #                 # normal component.
    #                 if au_dot_normal > 0.:
    #                     # remove the normal acceleration component
    #                     d_auhat[d_idx] -= au_dot_normal * d_normal[idx3]
    #                     d_avhat[d_idx] -= au_dot_normal * d_normal[idx3 + 1]
    #                     d_awhat[d_idx] -= au_dot_normal * d_normal[idx3 + 2]


class ResetParticlePositionsIPST(Equation):
    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z):
        d_x[d_idx] = d_ipst_x[d_idx]
        d_y[d_idx] = d_ipst_y[d_idx]
        d_z[d_idx] = d_ipst_z[d_idx]


######################
# IPST equations end #
######################


class GTVFEOS(Equation):
    def initialize(self, d_idx, d_rho, d_p, d_c0_ref, d_rho_ref):
        d_p[d_idx] = d_c0_ref[0] * d_c0_ref[0] * (d_rho[d_idx] - d_rho_ref[0])

    def post_loop(self, d_idx, d_rho, d_p0, d_p, d_p_ref):
        d_p0[d_idx] = min(10. * abs(d_p[d_idx]), d_p_ref[0])


class GTVFSetP0(Equation):
    def initialize(self, d_idx, d_rho, d_p0, d_p, d_p_ref):
        d_p0[d_idx] = min(10. * abs(d_p[d_idx]), d_p_ref[0])


class ComputeAuHatGTVF(Equation):
    def __init__(self, dest, sources):
        super(ComputeAuHatGTVF, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_p0, s_rho, s_m, d_auhat, d_avhat,
             d_awhat, WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ, HIJ):
        dwijhat = declare('matrix(3)')

        rhoa = d_rho[d_idx]

        rhoa21 = 1. / (rhoa * rhoa)

        # add the background pressure acceleration
        tmp = -d_p0[d_idx] * s_m[s_idx] * rhoa21

        SPH_KERNEL.gradient(XIJ, RIJ, 0.5 * HIJ, dwijhat)

        d_auhat[d_idx] += tmp * dwijhat[0]
        d_avhat[d_idx] += tmp * dwijhat[1]
        d_awhat[d_idx] += tmp * dwijhat[2]


class AdamiBoundaryConditionExtrapolateNoSlip(Equation):
    """
    Taken from

    [1] A numerical study on ice failure process and ice-ship interactions by
    Smoothed Particle Hydrodynamics
    [2] Adami 2012 boundary conditions paper.
    [3] LOQUAT: an open-source GPU-accelerated SPH solver for geotechnical modeling

    """
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(AdamiBoundaryConditionExtrapolateNoSlip, self).__init__(
            dest, sources)

    def initialize(self, d_idx, d_p, d_wij, d_s00, d_s01, d_s02, d_s11, d_s12,
                   d_s22):
        d_s00[d_idx] = 0.0
        d_s01[d_idx] = 0.0
        d_s02[d_idx] = 0.0
        d_s11[d_idx] = 0.0
        d_s12[d_idx] = 0.0
        d_s22[d_idx] = 0.0
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_wij, d_p, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
             d_au, d_av, d_aw, s_idx, s_s00, s_s01, s_s02, s_s11, s_s12, s_s22,
             s_p, s_rho, WIJ, XIJ):
        d_s00[d_idx] += s_s00[s_idx] * WIJ
        d_s01[d_idx] += s_s01[s_idx] * WIJ
        d_s02[d_idx] += s_s02[s_idx] * WIJ
        d_s11[d_idx] += s_s11[s_idx] * WIJ
        d_s12[d_idx] += s_s12[s_idx] * WIJ
        d_s22[d_idx] += s_s22[s_idx] * WIJ

        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p[d_idx] += s_p[s_idx] * WIJ + s_rho[s_idx]*gdotxij*WIJ

        # denominator of Eq. (27)
        d_wij[d_idx] += WIJ

    def post_loop(self, d_wij, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                  d_p):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_s00[d_idx] /= d_wij[d_idx]
            d_s01[d_idx] /= d_wij[d_idx]
            d_s02[d_idx] /= d_wij[d_idx]
            d_s11[d_idx] /= d_wij[d_idx]
            d_s12[d_idx] /= d_wij[d_idx]
            d_s22[d_idx] /= d_wij[d_idx]

            d_p[d_idx] /= d_wij[d_idx]


class AdamiBoundaryConditionExtrapolateFreeSlip(Equation):
    """
    Taken from

    [1] A numerical study on ice failure process and ice-ship interactions by
    Smoothed Particle Hydrodynamics
    [2] Adami 2012 boundary conditions paper.
    [3] LOQUAT: an open-source GPU-accelerated SPH solver for geotechnical modeling

    """
    def initialize(self, d_idx, d_p, d_wij, d_s00, d_s01, d_s02, d_s11, d_s12,
                   d_s22):
        d_s00[d_idx] = 0.0
        d_s01[d_idx] = 0.0
        d_s02[d_idx] = 0.0
        d_s11[d_idx] = 0.0
        d_s12[d_idx] = 0.0
        d_s22[d_idx] = 0.0
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_wij, d_p, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
             s_idx, s_s00, s_s01, s_s02, s_s11, s_s12, s_s22, s_p, WIJ):
        d_s00[d_idx] += s_s00[s_idx] * WIJ
        d_s01[d_idx] -= s_s01[s_idx] * WIJ
        d_s02[d_idx] -= s_s02[s_idx] * WIJ
        d_s11[d_idx] += s_s11[s_idx] * WIJ
        d_s12[d_idx] -= s_s12[s_idx] * WIJ
        d_s22[d_idx] += s_s22[s_idx] * WIJ

        d_p[d_idx] += s_p[s_idx] * WIJ

        # denominator of Eq. (27)
        d_wij[d_idx] += WIJ

    def post_loop(self, d_wij, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                  d_p):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_s00[d_idx] /= d_wij[d_idx]
            d_s01[d_idx] /= d_wij[d_idx]
            d_s02[d_idx] /= d_wij[d_idx]
            d_s11[d_idx] /= d_wij[d_idx]
            d_s12[d_idx] /= d_wij[d_idx]
            d_s22[d_idx] /= d_wij[d_idx]

            d_p[d_idx] /= d_wij[d_idx]


class ComputePrincipalStress2D(Equation):
    def initialize(self, d_idx, d_sigma_1, d_sigma_2, d_sigma00, d_sigma01,
                   d_sigma02, d_sigma11, d_sigma12, d_sigma22):
        # https://www.ecourses.ou.edu/cgi-bin/eBook.cgi?doc=&topic=me&chap_sec=07.2&page=theory
        tmp1 = (d_sigma00[d_idx] + d_sigma11[d_idx]) / 2

        tmp2 = (d_sigma00[d_idx] - d_sigma11[d_idx]) / 2

        tmp3 = sqrt(tmp2**2. + d_sigma01[d_idx]**2.)

        d_sigma_1[d_idx] = tmp1 + tmp3
        d_sigma_2[d_idx] = tmp1 - tmp3


class ComputeDivVelocity(Equation):
    def initialize(self, d_idx, d_div_vel):
        d_div_vel[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_div_vel, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_u, s_v, s_w, s_uhat, s_vhat, s_what, DWIJ):

        tmp = s_m[s_idx] / s_rho[s_idx]

        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - d_u[d_idx] - (s_uhat[s_idx] - s_u[s_idx])
        vij[1] = d_vhat[d_idx] - d_v[d_idx] - (s_vhat[s_idx] - s_v[s_idx])
        vij[2] = d_what[d_idx] - d_w[d_idx] - (s_what[s_idx] - s_w[s_idx])

        d_div_vel[d_idx] += tmp * -(vij[0] * DWIJ[0] + vij[1] * DWIJ[1] +
                                    vij[2] * DWIJ[2])


class ComputeDivDeviatoricStressOuterVelocity(Equation):
    def initialize(self, d_idx, d_s00u_x, d_s00v_y, d_s00w_z, d_s01u_x,
                   d_s01v_y, d_s01w_z, d_s02u_x, d_s02v_y, d_s02w_z, d_s11u_x,
                   d_s11v_y, d_s11w_z, d_s12u_x, d_s12v_y, d_s12w_z, d_s22u_x,
                   d_s22v_y, d_s22w_z):
        d_s00u_x[d_idx] = 0.0
        d_s00v_y[d_idx] = 0.0
        d_s00w_z[d_idx] = 0.0

        d_s01u_x[d_idx] = 0.0
        d_s01v_y[d_idx] = 0.0
        d_s01w_z[d_idx] = 0.0

        d_s02u_x[d_idx] = 0.0
        d_s02v_y[d_idx] = 0.0
        d_s02w_z[d_idx] = 0.0

        d_s11u_x[d_idx] = 0.0
        d_s11v_y[d_idx] = 0.0
        d_s11w_z[d_idx] = 0.0

        d_s12u_x[d_idx] = 0.0
        d_s12v_y[d_idx] = 0.0
        d_s12w_z[d_idx] = 0.0

        d_s22u_x[d_idx] = 0.0
        d_s22v_y[d_idx] = 0.0
        d_s22w_z[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_u, d_v, d_w, d_uhat, d_vhat, d_what, d_s00,
             d_s01, d_s02, d_s11, d_s12, d_s22, d_s00u_x, d_s00v_y, d_s00w_z,
             d_s01u_x, d_s01v_y, d_s01w_z, d_s02u_x, d_s02v_y, d_s02w_z,
             d_s11u_x, d_s11v_y, d_s11w_z, d_s12u_x, d_s12v_y, d_s12w_z,
             d_s22u_x, d_s22v_y, d_s22w_z, s_m, s_rho, s_u, s_v, s_w, s_uhat,
             s_vhat, s_what, s_s00, s_s01, s_s02, s_s11, s_s12, s_s22, DWIJ):

        tmp = s_m[s_idx] / s_rho[s_idx]

        ud = d_uhat[d_idx] - d_u[d_idx]
        vd = d_vhat[d_idx] - d_v[d_idx]
        wd = d_what[d_idx] - d_w[d_idx]

        us = s_uhat[s_idx] - s_u[s_idx]
        vs = s_vhat[s_idx] - s_v[s_idx]
        ws = s_what[s_idx] - s_w[s_idx]

        d_s00u_x[d_idx] += tmp * -(d_s00[d_idx] * ud -
                                   s_s00[s_idx] * us) * DWIJ[0]
        d_s00v_y[d_idx] += tmp * -(d_s00[d_idx] * vd -
                                   s_s00[s_idx] * vs) * DWIJ[1]
        d_s00w_z[d_idx] += tmp * -(d_s00[d_idx] * wd -
                                   s_s00[s_idx] * ws) * DWIJ[2]

        d_s01u_x[d_idx] += tmp * -(d_s01[d_idx] * ud -
                                   s_s01[s_idx] * us) * DWIJ[0]
        d_s01v_y[d_idx] += tmp * -(d_s01[d_idx] * vd -
                                   s_s01[s_idx] * vs) * DWIJ[1]
        d_s01w_z[d_idx] += tmp * -(d_s01[d_idx] * wd -
                                   s_s01[s_idx] * ws) * DWIJ[2]

        d_s02u_x[d_idx] += tmp * -(d_s02[d_idx] * ud -
                                   s_s02[s_idx] * us) * DWIJ[0]
        d_s02v_y[d_idx] += tmp * -(d_s02[d_idx] * vd -
                                   s_s02[s_idx] * vs) * DWIJ[1]
        d_s02w_z[d_idx] += tmp * -(d_s02[d_idx] * wd -
                                   s_s02[s_idx] * ws) * DWIJ[2]

        d_s11u_x[d_idx] += tmp * -(d_s11[d_idx] * ud -
                                   s_s11[s_idx] * us) * DWIJ[0]
        d_s11v_y[d_idx] += tmp * -(d_s11[d_idx] * vd -
                                   s_s11[s_idx] * vs) * DWIJ[1]
        d_s11w_z[d_idx] += tmp * -(d_s11[d_idx] * wd -
                                   s_s11[s_idx] * ws) * DWIJ[2]

        d_s12u_x[d_idx] += tmp * -(d_s12[d_idx] * ud -
                                   s_s12[s_idx] * us) * DWIJ[0]
        d_s12v_y[d_idx] += tmp * -(d_s12[d_idx] * vd -
                                   s_s12[s_idx] * vs) * DWIJ[1]
        d_s12w_z[d_idx] += tmp * -(d_s12[d_idx] * wd -
                                   s_s12[s_idx] * ws) * DWIJ[2]

        d_s22u_x[d_idx] += tmp * -(d_s22[d_idx] * ud -
                                   s_s22[s_idx] * us) * DWIJ[0]
        d_s22v_y[d_idx] += tmp * -(d_s22[d_idx] * vd -
                                   s_s22[s_idx] * vs) * DWIJ[1]
        d_s22w_z[d_idx] += tmp * -(d_s22[d_idx] * wd -
                                   s_s22[s_idx] * ws) * DWIJ[2]


class HookesDeviatoricStressRateETVFCorrection(Equation):
    def initialize(self, d_idx, d_as00, d_as01, d_as02, d_as11, d_as12, d_as22,
                   d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_s00u_x,
                   d_s00v_y, d_s00w_z, d_s01u_x, d_s01v_y, d_s01w_z, d_s02u_x,
                   d_s02v_y, d_s02w_z, d_s11u_x, d_s11v_y, d_s11w_z, d_s12u_x,
                   d_s12v_y, d_s12w_z, d_s22u_x, d_s22v_y, d_s22w_z,
                   d_div_vel):
        d_as00[d_idx] += (d_s00u_x[d_idx] + d_s00v_y[d_idx] + d_s00w_z[d_idx] +
                          d_s00[d_idx] * d_div_vel[d_idx])
        d_as01[d_idx] += (d_s01u_x[d_idx] + d_s01v_y[d_idx] + d_s01w_z[d_idx] +
                          d_s01[d_idx] * d_div_vel[d_idx])
        d_as02[d_idx] += (d_s02u_x[d_idx] + d_s02v_y[d_idx] + d_s02w_z[d_idx] +
                          d_s02[d_idx] * d_div_vel[d_idx])

        d_as11[d_idx] += (d_s11u_x[d_idx] + d_s11v_y[d_idx] + d_s11w_z[d_idx] +
                          d_s11[d_idx] * d_div_vel[d_idx])
        d_as12[d_idx] += (d_s12u_x[d_idx] + d_s12v_y[d_idx] + d_s12w_z[d_idx] +
                          d_s12[d_idx] * d_div_vel[d_idx])

        d_as22[d_idx] += (d_s22u_x[d_idx] + d_s22v_y[d_idx] + d_s22w_z[d_idx] +
                          d_s22[d_idx] * d_div_vel[d_idx])


class EDACEquation(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(EDACEquation, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_c0_ref, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_p, s_m, s_rho, d_ap, DWIJ, XIJ, s_uhat, s_vhat,
             s_what, s_u, s_v, s_w, R2IJ, VIJ, EPS):
        vhatij = declare('matrix(3)')
        vhatij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vhatij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vhatij[2] = d_what[d_idx] - s_what[s_idx]

        cs2 = d_c0_ref[0] * d_c0_ref[0]

        rhoj1 = 1.0 / s_rho[s_idx]
        Vj = s_m[s_idx] * rhoj1
        rhoi = d_rho[d_idx]
        pi = d_p[d_idx]
        rhoj = s_rho[s_idx]
        pj = s_p[s_idx]

        vij_dot_dwij = -(VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] +
                         VIJ[2] * DWIJ[2])

        vhatij_dot_dwij = -(vhatij[0] * DWIJ[0] + vhatij[1] * DWIJ[1] +
                            vhatij[2] * DWIJ[2])

        # vhatij_dot_dwij = (VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] +
        #                    VIJ[2]*DWIJ[2])

        #######################################################
        # first term on the rhs of Eq 23 of the current paper #
        #######################################################
        d_ap[d_idx] += (pi - rhoi * cs2) * Vj * vij_dot_dwij

        #######################################################
        # second term on the rhs of Eq 23 of the current paper #
        #######################################################
        d_ap[d_idx] += -pi * Vj * vhatij_dot_dwij

        ########################################################
        # third term on the rhs of Eq 19 of the current paper #
        ########################################################
        tmp0 = pj * (s_uhat[s_idx] - s_u[s_idx]) - pi * (d_uhat[d_idx] -
                                                         d_u[d_idx])

        tmp1 = pj * (s_vhat[s_idx] - s_v[s_idx]) - pi * (d_vhat[d_idx] -
                                                         d_v[d_idx])

        tmp2 = pj * (s_what[s_idx] - s_w[s_idx]) - pi * (d_what[d_idx] -
                                                         d_w[d_idx])

        tmpdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)
        d_ap[d_idx] += -Vj * tmpdotdwij

        #######################################################
        # fourth term on the rhs of Eq 19 of the current paper #
        #######################################################
        rhoij = d_rho[d_idx] + s_rho[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p[s_idx])


class MakeSurfaceParticlesPressureApZero(Equation):
    def initialize(self, d_idx, d_is_boundary, d_p, d_ap):
        # if the particle is boundary set it's h_b to be zero
        if d_is_boundary[d_idx] == 1:
            d_ap[d_idx] = 0.
            d_p[d_idx] = 0.


class MakeSurfaceParticlesPressureApZeroEDACUpdated(Equation):
    def initialize(self, d_idx, d_edac_is_boundary, d_p, d_ap):
        # if the particle is boundary set it's h_b to be zero
        if d_edac_is_boundary[d_idx] == 1:
            d_ap[d_idx] = 0.
            d_p[d_idx] = 0.


class SolidMechStep(IntegratorStep):
    """This step follows GTVF paper by Zhang 2017"""
    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_uhat, d_vhat,
               d_what, d_auhat, d_avhat, d_awhat, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]

    def stage2(self, d_idx, d_uhat, d_vhat, d_what, d_x, d_y, d_z, d_rho,
               d_arho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_as00,
               d_as01, d_as02, d_as11, d_as12, d_as22, d_sigma00, d_sigma01,
               d_sigma02, d_sigma11, d_sigma12, d_sigma22, d_p, dt):
        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

        # update deviatoric stress components
        d_s00[d_idx] = d_s00[d_idx] + dt * d_as00[d_idx]
        d_s01[d_idx] = d_s01[d_idx] + dt * d_as01[d_idx]
        d_s02[d_idx] = d_s02[d_idx] + dt * d_as02[d_idx]
        d_s11[d_idx] = d_s11[d_idx] + dt * d_as11[d_idx]
        d_s12[d_idx] = d_s12[d_idx] + dt * d_as12[d_idx]
        d_s22[d_idx] = d_s22[d_idx] + dt * d_as22[d_idx]

        # update sigma
        d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
        d_sigma01[d_idx] = d_s01[d_idx]
        d_sigma02[d_idx] = d_s02[d_idx]
        d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
        d_sigma12[d_idx] = d_s12[d_idx]
        d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]

        d_rho[d_idx] += dt * d_arho[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]


class SolidMechStepEDAC(SolidMechStep):
    """This step follows GTVF paper by Zhang 2017"""
    def stage2(self, d_idx, d_m, d_uhat, d_vhat, d_what, d_x, d_y, d_z, d_rho,
               d_arho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_as00,
               d_as01, d_as02, d_as11, d_as12, d_as22, d_sigma00, d_sigma01,
               d_sigma02, d_sigma11, d_sigma12, d_sigma22, d_p, d_ap, dt):
        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

        # update deviatoric stress components
        d_s00[d_idx] = d_s00[d_idx] + dt * d_as00[d_idx]
        d_s01[d_idx] = d_s01[d_idx] + dt * d_as01[d_idx]
        d_s02[d_idx] = d_s02[d_idx] + dt * d_as02[d_idx]
        d_s11[d_idx] = d_s11[d_idx] + dt * d_as11[d_idx]
        d_s12[d_idx] = d_s12[d_idx] + dt * d_as12[d_idx]
        d_s22[d_idx] = d_s22[d_idx] + dt * d_as22[d_idx]

        # update sigma
        d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
        d_sigma01[d_idx] = d_s01[d_idx]
        d_sigma02[d_idx] = d_s02[d_idx]
        d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
        d_sigma12[d_idx] = d_s12[d_idx]
        d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]

        d_rho[d_idx] += dt * d_arho[d_idx]

        d_p[d_idx] += dt * d_ap[d_idx]


class EDACIntegrator(Integrator):
    def initial_acceleration(self, t, dt):
        pass

    def one_timestep(self, t, dt):
        self.compute_accelerations(0, update_nnps=False)
        self.stage1()
        self.do_post_stage(dt, 1)
        self.update_domain()

        self.compute_accelerations(1)

        self.stage2()
        self.do_post_stage(dt, 2)


class SolidMechETVFEDACIntegStep(IntegratorStep):
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_au, d_av, d_aw,
               d_uhat, d_vhat, d_what, d_auhat, d_avhat, d_awhat, dt):
        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_w[d_idx] += dt * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dt * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dt * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dt * d_awhat[d_idx]

        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

    def stage2(self, d_idx, d_m, d_uhat, d_vhat, d_what, d_rho, d_vol, d_ap,
               d_avol, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_as00,
               d_as01, d_as02, d_as11, d_as12, d_as22, d_sigma00, d_sigma01,
               d_sigma02, d_sigma11, d_sigma12, d_sigma22, d_p, dt):
        d_vol[d_idx] += dt * d_avol[d_idx]

        # set the density from the volume
        d_rho[d_idx] = d_m[d_idx] / d_vol[d_idx]

        d_p[d_idx] += dt * d_ap[d_idx]

        # update deviatoric stress components
        d_s00[d_idx] = d_s00[d_idx] + dt * d_as00[d_idx]
        d_s01[d_idx] = d_s01[d_idx] + dt * d_as01[d_idx]
        d_s02[d_idx] = d_s02[d_idx] + dt * d_as02[d_idx]
        d_s11[d_idx] = d_s11[d_idx] + dt * d_as11[d_idx]
        d_s12[d_idx] = d_s12[d_idx] + dt * d_as12[d_idx]
        d_s22[d_idx] = d_s22[d_idx] + dt * d_as22[d_idx]

        # update sigma
        d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
        d_sigma01[d_idx] = d_s01[d_idx]
        d_sigma02[d_idx] = d_s02[d_idx]
        d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
        d_sigma12[d_idx] = d_s12[d_idx]
        d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]


class StiffEOS(Equation):
    def __init__(self, dest, sources, gamma):
        self.gamma = gamma
        super(StiffEOS, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_p, d_c0_ref, d_rho_ref):
        tmp = d_rho[d_idx] / d_rho_ref[0]
        tmp1 = d_rho_ref[0] * d_c0_ref[0] * d_c0_ref[0] / self.gamma
        d_p[d_idx] = tmp1 * (pow(tmp, self.gamma) - 1.)


class MakeAuhatZero(Equation):
    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.
        d_avhat[d_idx] = 0.
        d_awhat[d_idx] = 0.


class ContinuityEquationSolid(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, d_u, d_v, d_w, s_ub, s_vb, s_wb,
             DWIJ):
        vij = declare('matrix(3)')

        vij[0] = d_u[d_idx] - s_ub[s_idx]
        vij[1] = d_v[d_idx] - s_vb[s_idx]
        vij[2] = d_w[d_idx] - s_wb[s_idx]

        vijdotdwij = DWIJ[0]*vij[0] + DWIJ[1]*vij[1] + DWIJ[2]*vij[2]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij


class ContinuityEquationUhatSolid(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, d_uhat, d_vhat, d_what, s_idx, s_m, s_ubhat,
             s_vbhat, s_wbhat, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_ubhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vbhat[s_idx]
        vij[2] = d_what[d_idx] - s_wbhat[s_idx]

        vijdotdwij = DWIJ[0] * vij[0] + DWIJ[1] * vij[1] + DWIJ[2] * vij[2]
        d_arho[d_idx] += s_m[s_idx] * vijdotdwij


class ContinuityEquationETVFCorrectionSolid(Equation):
    """
    This is the additional term arriving in the new ETVF continuity equation
    """
    def loop(self, d_idx, d_arho, d_rho, d_u, d_v, d_w, d_uhat, d_vhat, d_what,
             s_idx, s_rho, s_m, s_ub, s_vb, s_wb, s_ubhat, s_vbhat, s_wbhat,
             DWIJ):
        tmp0 = s_rho[s_idx] * (s_ubhat[s_idx] - s_ub[s_idx]) - d_rho[d_idx] * (
            d_uhat[d_idx] - d_u[d_idx])

        tmp1 = s_rho[s_idx] * (s_vbhat[s_idx] - s_vb[s_idx]) - d_rho[d_idx] * (
            d_vhat[d_idx] - d_v[d_idx])

        tmp2 = s_rho[s_idx] * (s_wbhat[s_idx] - s_wb[s_idx]) - d_rho[d_idx] * (
            d_what[d_idx] - d_w[d_idx])

        vijdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)

        d_arho[d_idx] += s_m[s_idx] / s_rho[s_idx] * vijdotdwij


class VelocityGradient2DSolid(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v10, d_v11):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_v00, d_v01, d_v10, d_v11, d_u,
             d_v, d_w, s_ub, s_vb, s_wb, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ub[s_idx]
        vij[1] = d_v[d_idx] - s_vb[s_idx]
        vij[2] = d_w[d_idx] - s_wb[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]


class VelocityGradient3DSolid(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v02, d_v10, d_v11, d_v12,
                   d_v20, d_v21, d_v22):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v02[d_idx] = 0.0

        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0
        d_v12[d_idx] = 0.0

        d_v20[d_idx] = 0.0
        d_v21[d_idx] = 0.0
        d_v22[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_v00, d_v01, d_v02, d_v10, d_v11,
             d_v12, d_v20, d_v21, d_v22, d_u, d_v, d_w, s_ub, s_vb, s_wb,
             DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ub[s_idx]
        vij[1] = d_v[d_idx] - s_vb[s_idx]
        vij[2] = d_w[d_idx] - s_wb[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]
        d_v02[d_idx] += tmp * -vij[0] * DWIJ[2]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]
        d_v12[d_idx] += tmp * -vij[1] * DWIJ[2]

        d_v20[d_idx] += tmp * -vij[2] * DWIJ[0]
        d_v21[d_idx] += tmp * -vij[2] * DWIJ[1]
        d_v22[d_idx] += tmp * -vij[2] * DWIJ[2]


class VelocityGradient2DUhatSolid(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v10, d_v11):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0

    def loop(self, d_uhat, d_vhat, d_what, d_idx, s_idx, s_m, s_rho, d_v00,
             d_v01, d_v10, d_v11, s_ubhat, s_vbhat, s_wbhat, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_ubhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vbhat[s_idx]
        vij[2] = d_what[d_idx] - s_wbhat[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]


class VelocityGradient3DUhatSolid(Equation):
    def initialize(self, d_idx, d_v00, d_v01, d_v02, d_v10, d_v11, d_v12,
                   d_v20, d_v21, d_v22):
        d_v00[d_idx] = 0.0
        d_v01[d_idx] = 0.0
        d_v02[d_idx] = 0.0

        d_v10[d_idx] = 0.0
        d_v11[d_idx] = 0.0
        d_v12[d_idx] = 0.0

        d_v20[d_idx] = 0.0
        d_v21[d_idx] = 0.0
        d_v22[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, s_rho, d_v00, d_v01, d_v02, d_v10, d_v11,
             d_v12, d_v20, d_v21, d_v22, d_uhat, d_vhat, d_what, s_ubhat,
             s_vbhat, s_wbhat, DWIJ, VIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_ubhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vbhat[s_idx]
        vij[2] = d_what[d_idx] - s_wbhat[s_idx]

        tmp = s_m[s_idx] / s_rho[s_idx]

        d_v00[d_idx] += tmp * -vij[0] * DWIJ[0]
        d_v01[d_idx] += tmp * -vij[0] * DWIJ[1]
        d_v02[d_idx] += tmp * -vij[0] * DWIJ[2]

        d_v10[d_idx] += tmp * -vij[1] * DWIJ[0]
        d_v11[d_idx] += tmp * -vij[1] * DWIJ[1]
        d_v12[d_idx] += tmp * -vij[1] * DWIJ[2]

        d_v20[d_idx] += tmp * -vij[2] * DWIJ[0]
        d_v21[d_idx] += tmp * -vij[2] * DWIJ[1]
        d_v22[d_idx] += tmp * -vij[2] * DWIJ[2]


class SetWallVelocityNoSlipUSolidMech(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_ub, d_vb, d_wb, d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

        d_ub[d_idx] = 0.0
        d_vb[d_idx] = 0.0
        d_wb[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_uf, d_vf, d_wf, s_u, s_v, s_w,
             d_wij, WIJ):
        # normalisation factor is different from 'V' as the particles
        # near the boundary do not have full kernel support
        d_wij[d_idx] += WIJ

        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_u[s_idx] * WIJ
        d_vf[d_idx] += s_v[s_idx] * WIJ
        d_wf[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ub, d_vb, d_wb, d_u,
                  d_v, d_w, d_normal):
        idx3 = declare('int', 1)
        idx3 = 3 * d_idx
        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # For No slip boundary conditions
        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ub[d_idx] = 2 * d_u[d_idx] - d_uf[d_idx]
        d_vb[d_idx] = 2 * d_v[d_idx] - d_vf[d_idx]
        d_wb[d_idx] = 2 * d_w[d_idx] - d_wf[d_idx]

        vn = (d_ub[d_idx]*d_normal[idx3] + d_vb[d_idx]*d_normal[idx3+1]
              + d_wb[d_idx]*d_normal[idx3+2])
        if vn < 0:
            d_ub[d_idx] -= vn*d_normal[idx3]
            d_vb[d_idx] -= vn*d_normal[idx3+1]
            d_wb[d_idx] -= vn*d_normal[idx3+2]


class SetWallVelocityNoSlipUhatSolidMech(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf,
                   d_ubhat, d_vbhat, d_wbhat,
                   d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

        d_ubhat[d_idx] = 0.0
        d_vbhat[d_idx] = 0.0
        d_wbhat[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_uf, d_vf, d_wf, s_u, s_v, s_w,
             d_ubhat, d_vbhat, d_wbhat, s_uhat, s_vhat, s_what,
             d_wij, WIJ):
        # normalisation factor is different from 'V' as the particles
        # near the boundary do not have full kernel support
        d_wij[d_idx] += WIJ

        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_uhat[s_idx] * WIJ
        d_vf[d_idx] += s_vhat[s_idx] * WIJ
        d_wf[d_idx] += s_what[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ubhat, d_vbhat,
                  d_wbhat, d_uhat, d_vhat, d_what, d_normal):
        idx3 = declare('int', 1)
        idx3 = 3 * d_idx
        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # For No slip boundary conditions
        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ubhat[d_idx] = 2 * d_uhat[d_idx] - d_uf[d_idx]
        d_vbhat[d_idx] = 2 * d_vhat[d_idx] - d_vf[d_idx]
        d_wbhat[d_idx] = 2 * d_what[d_idx] - d_wf[d_idx]

        vn = (d_ubhat[d_idx]*d_normal[idx3] + d_vbhat[d_idx]*d_normal[idx3+1]
              + d_wbhat[d_idx]*d_normal[idx3+2])
        if vn < 0:
            d_ubhat[d_idx] -= vn*d_normal[idx3]
            d_vbhat[d_idx] -= vn*d_normal[idx3+1]
            d_wbhat[d_idx] -= vn*d_normal[idx3+2]


class SolidsScheme(Scheme):
    """

    There are three schemes

    1. GRAY
    2. GTVF
    3. ETVF

    ETVF scheme in particular has 2 PST techniques

    1. SUN2019
    2. IPST


    Using the following commands one can use these schemes

    1. GRAY

    python file_name.py --no-volume-correction --no-uhat --no-shear-tvf-correction --pst gray -d filename_pst_gray_output

    2. GTVF

    python file_name.py --no-volume-correction --uhat --no-shear-tvf-correction --pst gtvf -d filename_pst_gtvf_output

    3. ETVF

    python file_name.py --volume-correction --no-uhat --shear-tvf-correction --pst $(sun2019 or ipst) -d filename_etvf_pst_$(sun2019_or_ipst)_output


    ipst has additional arguments such as `ipst_max_iterations`, this can be
    changed using command line arguments


    # Note

    Additionally one can go for EDAC option

    python file_name.py --edac --surface-p-zero $(rest_of_the_arguments)

    """
    def __init__(self, solids, boundaries, dim, pb, edac_nu, mach_no,
                 hdx, ipst_max_iterations=10, ipst_min_iterations=0,
                 ipst_tolerance=0.2, ipst_interval=1, use_uhat_velgrad=False,
                 use_uhat_cont=False, artificial_vis_alpha=1.0,
                 artificial_vis_beta=0.0, artificial_stress_eps=0.3,
                 continuity_tvf_correction=False,
                 shear_stress_tvf_correction=False, kernel_choice="1",
                 stiff_eos=False, gamma=7., pst="sun2019",
                 gx=0., gy=0., gz=0.):
        self.solids = solids
        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        self.dim = dim

        # TODO: if the kernel is adaptive this will fail
        self.hdx = hdx

        # for Monaghan stress
        self.artificial_stress_eps = artificial_stress_eps

        # TODO: kernel_fac will change with kernel. This should change
        self.kernel_choice = "1"
        self.kernel = QuinticSpline
        self.kernel_factor = 2

        self.use_uhat_cont = use_uhat_cont
        self.use_uhat_velgrad = use_uhat_velgrad

        self.pb = pb

        self.no_boundaries = len(self.boundaries)

        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        self.continuity_tvf_correction = continuity_tvf_correction
        self.shear_stress_tvf_correction = shear_stress_tvf_correction

        self.edac_nu = edac_nu
        self.surf_p_zero = True
        self.edac = False

        self.pst = pst

        # attributes for P Sun 2019 PST technique
        self.mach_no = mach_no

        # attributes for IPST technique
        self.ipst_max_iterations = ipst_max_iterations
        self.ipst_min_iterations = ipst_min_iterations
        self.ipst_tolerance = ipst_tolerance
        self.ipst_interval = ipst_interval

        self.debug = False

        self.stiff_eos = stiff_eos
        self.gamma = gamma

        # boundary conditions
        self.adami_velocity_extrapolate = False
        self.no_slip = False
        self.free_slip = False

        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        add_bool_argument(
            group, 'surf-p-zero', dest='surf_p_zero', default=True,
            help='Make the surface pressure and acceleration to be zero')

        add_bool_argument(group, 'uhat-cont', dest='use_uhat_cont',
                          default=False,
                          help='Use Uhat in continuity equation')

        add_bool_argument(group, 'uhat-velgrad', dest='use_uhat_velgrad',
                          default=False,
                          help='Use Uhat in velocity gradient computation')

        group.add_argument("--artificial-vis-alpha", action="store",
                           dest="artificial_vis_alpha", default=1.0,
                           type=float,
                           help="Artificial viscosity coefficients")

        group.add_argument("--artificial-vis-beta", action="store",
                           dest="artificial_vis_beta", default=1.0, type=float,
                           help="Artificial viscosity coefficients, beta")

        add_bool_argument(
            group, 'continuity-tvf-correction',
            dest='continuity_tvf_correction', default=True,
            help='Add the extra continuty term arriving due to TVF')

        add_bool_argument(
            group, 'shear-stress-tvf-correction',
            dest='shear_stress_tvf_correction', default=True,
            help='Add the extra shear stress rate term arriving due to TVF')

        add_bool_argument(group, 'edac', dest='edac', default=True,
                          help='Use pressure evolution equation EDAC')

        add_bool_argument(group, 'adami-velocity-extrapolate',
                          dest='adami_velocity_extrapolate', default=False,
                          help='Use adami velocity extrapolation')

        add_bool_argument(group, 'no-slip', dest='no_slip', default=False,
                          help='No slip bc')

        add_bool_argument(group, 'free-slip', dest='free_slip', default=False,
                          help='Free slip bc')

        choices = ['sun2019', 'ipst', 'gray', 'gtvf', 'none']
        group.add_argument(
            "--pst", action="store", dest='pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use (one of %s)." % choices)

        group.add_argument("--ipst-max-iterations", action="store",
                           dest="ipst_max_iterations", default=10, type=int,
                           help="Max iterations of IPST")

        group.add_argument("--ipst-min-iterations", action="store",
                           dest="ipst_min_iterations", default=5, type=int,
                           help="Min iterations of IPST")

        group.add_argument("--ipst-interval", action="store",
                           dest="ipst_interval", default=1, type=int,
                           help="Frequency at which IPST is to be done")

        group.add_argument("--ipst-tolerance", action="store", type=float,
                           dest="ipst_tolerance", default=None,
                           help="Tolerance limit of IPST")

        add_bool_argument(group, 'debug', dest='debug', default=False,
                          help='Check if the IPST converged')

        choices = ["1", "2", "3", "4", "5", "6", "7", "8"]
        group.add_argument(
            "--kernel-choice", action="store", dest='kernel_choice',
            default="1", choices=choices,
            help="""Specify what kernel to use (one of %s).
                           1. QuinticSpline
                           2. WendlandQuintic
                           3. CubicSpline
                           4. WendlandQuinticC4
                           5. Gaussian
                           6. SuperGaussian
                           7. Gaussian
                           8. Gaussian""" % choices)

        add_bool_argument(group, 'stiff-eos', dest='stiff_eos', default=False,
                          help='use stiff equation of state')

    def consume_user_options(self, options):
        _vars = [
            'surf_p_zero', 'use_uhat_cont', 'use_uhat_velgrad',
            'artificial_vis_alpha', 'shear_stress_tvf_correction', 'edac',
            'pst', 'debug', 'ipst_max_iterations', 'ipst_tolerance',
            'ipst_interval', 'kernel_choice', 'stiff_eos',
            'continuity_tvf_correction', 'adami_velocity_extrapolate',
            'no_slip', 'free_slip'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def attributes_changed(self):
        if self.kernel_choice == "1":
            self.kernel = QuinticSpline
            self.kernel_factor = 3
        elif self.kernel_choice == "2":
            self.kernel = WendlandQuintic
            self.kernel_factor = 2
        elif self.kernel_choice == "3":
            self.kernel = CubicSpline
            self.kernel_factor = 2
        elif self.kernel_choice == "4":
            self.kernel = WendlandQuinticC4
            self.kernel_factor = 2
            self.h = self.h / self.hdx * 2.0
        elif self.kernel_choice == "5":
            self.kernel = Gaussian
            self.kernel_factor = 3
        elif self.kernel_choice == "6":
            self.kernel = SuperGaussian
            self.kernel_factor = 3

    def check_ipst_time(self, t, dt):
        if int(t / dt) % self.ipst_interval == 0:
            return True
        else:
            return False

    def get_equations(self):
        # from fluids import (SetWallVelocityFreeSlip, ContinuitySolidEquation,
        #                     ContinuitySolidEquationGTVF,
        #                     ContinuitySolidEquationETVFCorrection)

        from pysph.sph.equation import Group, MultiStageEquations
        from pysph.sph.basic_equations import (ContinuityEquation,
                                               MonaghanArtificialViscosity,
                                               VelocityGradient3D,
                                               VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (IsothermalEOS,
                                                HookesDeviatoricStressRate,
                                                MonaghanArtificialStress)

        stage1 = []
        g1 = []
        all = list(set(self.solids + self.boundaries))

        # ------------------------
        # stage 1 equations starts
        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #
        tmp = []
        if len(self.boundaries) > 0:
            for boundary in self.boundaries:
                # if self.free_slip is True:
                #     tmp.append(
                #         SetWallVelocityFreeSlipSolidMech(dest=boundary,
                #                                          sources=self.solids))

                # else:
                tmp.append(
                    SetWallVelocityNoSlipUhatSolidMech(dest=boundary,
                                                       sources=self.solids))
            stage1.append(Group(equations=tmp))

        tmp = []
        if len(self.boundaries) > 0:
            for boundary in self.boundaries:
                # if self.free_slip is True:
                #     tmp.append(
                #         SetWallVelocityFreeSlipSolidMech(dest=boundary,
                #                                          sources=self.solids))

                # else:
                tmp.append(
                    SetWallVelocityNoSlipUSolidMech(dest=boundary,
                                                    sources=self.solids))
            stage1.append(Group(equations=tmp))
        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #

        for solid in self.solids:
            if self.use_uhat_cont is True:
                g1.append(ContinuityEquationUhat(dest=solid, sources=self.solids))
            else:
                g1.append(ContinuityEquation(dest=solid, sources=self.solids))

            if self.continuity_tvf_correction is True:
                g1.append(
                    ContinuityEquationETVFCorrection(dest=solid,
                                                     sources=self.solids))

            if self.use_uhat_velgrad is True:
                if self.dim == 2:
                    g1.append(
                        VelocityGradient2DUhat(dest=solid, sources=self.solids))
                elif self.dim == 3:
                    g1.append(
                        VelocityGradient3DUhat(dest=solid, sources=self.solids))
            else:
                if self.dim == 2:
                    g1.append(VelocityGradient2D(dest=solid, sources=self.solids))
                elif self.dim == 3:
                    g1.append(VelocityGradient3D(dest=solid, sources=self.solids))

            if len(self.boundaries) > 0:
                if self.use_uhat_cont is True:
                    g1.append(
                        ContinuityEquationUhatSolid(dest=solid,
                                                    sources=self.boundaries))
                else:
                    g1.append(
                        ContinuityEquationSolid(dest=solid,
                                                sources=self.boundaries))

                if self.continuity_tvf_correction is True:
                    g1.append(
                        ContinuityEquationETVFCorrectionSolid(
                            dest=solid, sources=self.boundaries))

                if self.use_uhat_velgrad is True:
                    if self.dim == 2:
                        g1.append(
                            VelocityGradient2DUhatSolid(
                                dest=solid, sources=self.boundaries))
                    elif self.dim == 3:
                        g1.append(
                            VelocityGradient3DUhatSolid(
                                dest=solid, sources=self.boundaries))
                else:
                    if self.dim == 2:
                        g1.append(
                            VelocityGradient2DSolid(dest=solid,
                                                    sources=self.boundaries))
                    elif self.dim == 3:
                        g1.append(
                            VelocityGradient3DSolid(dest=solid,
                                                    sources=self.boundaries))

            if self.pst == "gray":
                g1.append(
                    MonaghanArtificialStress(dest=solid, sources=None,
                                             eps=self.artificial_stress_eps))

        stage1.append(Group(equations=g1))

        # --------------------
        # solid equations
        # --------------------
        g2 = []
        for solid in self.solids:
            g2.append(HookesDeviatoricStressRate(dest=solid, sources=None))

        stage1.append(Group(equations=g2))

        # edac pressure evolution equation
        if self.edac is True:
            gtmp = []
            for solid in self.solids:
                gtmp.append(
                    EDACEquation(dest=solid, sources=all, nu=self.edac_nu))

            stage1.append(Group(gtmp))

        if self.surf_p_zero is True:
            # -------------------
            # find the surface particles whose pressure has to be zero
            # -------------------
            g4 = []
            g5 = []
            g6 = []
            for pa in self.solids:
                g4.append(ComputeNormalsEDAC(dest=pa, sources=all))
                g5.append(SmoothNormalsEDAC(dest=pa, sources=all))
                g6.append(
                    IdentifyBoundaryParticleCosAngleEDAC(dest=pa, sources=all))

            stage1.append(Group(equations=g4))
            stage1.append(Group(equations=g5))
            stage1.append(Group(equations=g6))

        # ------------------------
        # stage 2 equations starts
        # ------------------------

        stage2 = []
        g1 = []
        g2 = []
        g3 = []
        g4 = []

        if self.pst in ["sun2019", "ipst"]:
            for solid in self.solids:
                g1.append(
                    SetHIJForInsideParticles(dest=solid, sources=[solid],
                                             kernel_factor=self.kernel_factor))
            stage2.append(Group(g1))

        if self.edac is False:
            if self.pst in ["gray", "ipst", "sun2019"]:
                for solid in self.solids:
                    if self.stiff_eos is True:
                        g2.append(
                            StiffEOS(solid, sources=None, gamma=self.gamma))
                    else:
                        g2.append(IsothermalEOS(solid, sources=None))

            elif self.pst == "gtvf":
                for solid in self.solids:
                    g2.append(GTVFEOS(solid, sources=None))

            if len(g2) > 0:
                stage2.append(Group(g2))
        else:
            if self.pst == "gtvf":
                for solid in self.solids:
                    g2.append(GTVFSetP0(solid, sources=None))

                stage2.append(Group(g2))

        # make the acceleration of pressure and pressure of boundary
        # particles zero
        if self.surf_p_zero is True:
            g2_tmp = []
            for pa in self.solids:
                g2_tmp.append(
                    MakeSurfaceParticlesPressureApZeroEDACUpdated(
                        dest=pa, sources=None))

            stage2.append(Group(equations=g2_tmp))

        # -------------------
        # boundary conditions
        # -------------------
        for boundary in self.boundaries:
            if self.free_slip is True:
                g3.append(
                    AdamiBoundaryConditionExtrapolateFreeSlip(
                        dest=boundary, sources=self.solids))
            else:
                g3.append(
                    AdamiBoundaryConditionExtrapolateNoSlip(
                        dest=boundary, sources=self.solids,
                        gx=self.gx, gy=self.gy, gz=self.gz
                    ))
        if len(g3) > 0:
            stage2.append(Group(g3))

        # -------------------
        # solve momentum equation for solid
        # -------------------
        g4 = []
        for solid in self.solids:
            # add only if there is some positive value
            if self.artificial_vis_alpha > 0. or self.artificial_vis_beta > 0.:
                g4.append(
                    MonaghanArtificialViscosity(
                        dest=solid, sources=all,
                        alpha=self.artificial_vis_alpha,
                        beta=self.artificial_vis_beta))

            g4.append(MomentumEquationSolids(dest=solid, sources=all))

            if self.pst == "sun2019":
                g4.append(
                    ComputeAuHatETVFSun2019Solid(
                        dest=solid, sources=[solid] + self.boundaries,
                        mach_no=self.mach_no))
            elif self.pst == "gtvf":
                g4.append(
                    ComputeAuHatGTVF(dest=solid,
                                     sources=[solid] + self.boundaries))

            elif self.pst == "gray":
                g4.append(
                    MonaghanArtificialStressCorrection(dest=solid,
                                                       sources=[solid]))

        stage2.append(Group(g4))

        # this PST is handled separately
        if self.pst == "ipst":
            g5 = []
            g6 = []
            g7 = []
            g8 = []

            # make auhat zero before computation of ipst force
            eqns = []
            for solid in self.solids:
                eqns.append(MakeAuhatZero(dest=solid, sources=None))

            stage2.append(Group(eqns))

            for solid in self.solids:
                g5.append(
                    SavePositionsIPSTBeforeMoving(dest=solid, sources=None))

                # these two has to be in the iterative group and the nnps has to
                # be updated
                # ---------------------------------------
                g6.append(
                    AdjustPositionIPST(dest=solid,
                                       sources=[solid] + self.boundaries,
                                       u_max=self.u_max))

                g7.append(
                    CheckUniformityIPST(dest=solid,
                                        sources=[solid] + self.boundaries,
                                        debug=self.debug))
                # ---------------------------------------

                g8.append(ComputeAuhatETVFIPST(dest=solid, sources=None))
                g8.append(ResetParticlePositionsIPST(dest=solid, sources=None))

            stage2.append(Group(g5, condition=self.check_ipst_time))

            # this is the iterative group
            stage2.append(
                Group(equations=[Group(equations=g6),
                                 Group(equations=g7)], iterate=True,
                      max_iterations=self.ipst_max_iterations,
                      min_iterations=self.ipst_min_iterations,
                      condition=self.check_ipst_time))

            stage2.append(Group(g8, condition=self.check_ipst_time))

        g9 = []
        for solid in self.solids:
            g9.append(AddGravityToStructure(dest=solid, sources=None,
                                            gx=self.gx,
                                            gy=self.gy,
                                            gz=self.gz))

        stage2.append(Group(equations=g9))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        """TODO: Fix the integrator of the boundary. If it is solve_tau then solve for
        deviatoric stress or else no integrator has to be used
        """
        kernel = self.kernel(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.wc.gtvf import GTVFIntegrator

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator

        if self.edac is True:
            step_cls = SolidMechStepEDAC
        else:
            step_cls = SolidMechStep

        for name in self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        # for name in self.boundaries:
        #     if name not in steppers:
        #         steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties

        pas = dict([(p.name, p) for p in particles])

        for solid in self.solids:
            # we expect the solid to have Young's modulus, Poisson ration as
            # given
            pa = pas[solid]

            # add the properties that are used by all the schemes
            add_properties(pa, 'cs', 'v00', 'v01', 'v02', 'v10', 'v11', 'v12',
                           'v20', 'v21', 'v22', 's00', 's01', 's02', 's11',
                           's12', 's22', 'as00', 'as01', 'as02', 'as11',
                           'as12', 'as22', 'arho', 'au', 'av', 'aw')

            # this will change
            kernel = self.kernel(dim=2)
            wdeltap = kernel.kernel(rij=pa.spacing0[0], h=pa.h[0])
            pa.add_constant('wdeltap', wdeltap)

            # set the shear modulus G
            G = get_shear_modulus(pa.E[0], pa.nu[0])
            pa.add_constant('G', G)

            # set the speed of sound
            cs = np.ones_like(pa.x) * get_speed_of_sound(
                pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.cs[:] = cs[:]

            c0_ref = get_speed_of_sound(pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.add_constant('c0_ref', c0_ref)

            # auhat properties are needed for gtvf, etvf but not for gray. But
            # for the compatability with the integrator we will add
            add_properties(pa, 'auhat', 'avhat', 'awhat', 'uhat', 'vhat',
                           'what')

            add_properties(pa, 'sigma00', 'sigma01', 'sigma02', 'sigma11',
                           'sigma12', 'sigma22')

            # output arrays
            pa.add_output_arrays(['sigma00', 'sigma01', 'sigma11'])

            # now add properties specific to the scheme and PST
            if self.pst == "gray":
                add_properties(pa, 'r02', 'r11', 'r22', 'r01', 'r00', 'r12')

            if self.pst == "gtvf":
                add_properties(pa, 'p0')

                if 'p_ref' not in pa.constants:
                    pa.add_constant('p_ref', 0.)

                if 'b_mod' not in pa.constants:
                    pa.add_constant('b_mod', 0.)

                pa.b_mod[0] = get_bulk_mod(pa.G[0], pa.nu[0])
                pa.p_ref[0] = pa.b_mod[0]

            if self.pst == "sun2019" or "ipst":
                # for boundary identification and for sun2019 pst
                pa.add_property('normal', stride=3)
                pa.add_property('normal_tmp', stride=3)
                pa.add_property('normal_norm')

                # check for boundary particle
                pa.add_property('is_boundary', type='int')

                # used to set the particles near the boundary
                pa.add_property('h_b')

            # if the PST is IPST
            if self.pst == "ipst":
                setup_ipst(pa, self.kernel)

            # for edac
            if self.edac == True:
                add_properties(pa, 'ap')

            if self.surf_p_zero == True:
                pa.add_property('edac_normal', stride=3)
                pa.add_property('edac_normal_tmp', stride=3)
                pa.add_property('edac_normal_norm')

                # check for edac boundary particle
                pa.add_property('edac_is_boundary', type='int')

                pa.add_property('ap')

            # add the corrected shear stress rate
            if self.shear_stress_tvf_correction == True:
                add_properties(pa, 'div_vel')
                add_properties(pa, 's11v_y', 's01v_y', 's11w_z', 's00w_z',
                               's12w_z', 's12v_y', 's02u_x', 's22w_z',
                               's11u_x', 's22u_x', 's00u_x', 's02w_z',
                               's02v_y', 's00v_y', 's01w_z', 's22v_y',
                               's12u_x', 's01u_x')

            # update the h if using wendlandquinticc4
            if self.kernel_choice == "4":
                pa.h[:] = pa.h[:] / self.hdx * 2.

            pa.add_output_arrays(['p'])

        for boundary in self.boundaries:
            pa = pas[boundary]
            pa.add_property('wij')

            # for adami boundary condition

            add_properties(pa, 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 's00',
                           's01', 's02', 's11', 's12', 's22', 'cs', 'uhat',
                           'vhat', 'what', 'ub', 'vb', 'wb', 'ubhat', 'vbhat',
                           'wbhat')

            cs = np.ones_like(pa.x) * get_speed_of_sound(
                pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.cs[:] = cs[:]

            if self.continuity_tvf_correction == True:
                pa.add_property('ughat')
                pa.add_property('vghat')
                pa.add_property('wghat')

            if self.surf_p_zero == True:
                pa.add_property('edac_normal', stride=3)
                pa.add_property('edac_normal_tmp', stride=3)
                pa.add_property('edac_normal_norm')

            # now add properties specific to the scheme and PST
            if self.pst == "gray":
                add_properties(pa, 'r02', 'r11', 'r22', 'r01', 'r00', 'r12')

            if self.pst == "gtvf":
                add_properties(pa, 'uhat', 'vhat', 'what')

            if self.kernel_choice == "4":
                pa.h[:] = pa.h[:] / self.hdx * 2.

            pa.add_property('normal', stride=3)
            pa.add_output_arrays(['normal'])
            pa.add_property('normal_tmp', stride=3)

            name = pa.name
            props = ['m', 'rho', 'h']
            for p in props:
                x = pa.get(p)
                if numpy.all(x < 1e-12):
                    msg = f'WARNING: cannot compute normals "{p}" is zero'
                    print(msg)

            seval = SPHEvaluator(
                arrays=[pa], equations=[
                    Group(
                        equations=[ComputeNormals(dest=name, sources=[name])]),
                    Group(
                        equations=[SmoothNormals(dest=name, sources=[name])]),
                ], dim=self.dim)
            seval.evaluate()

    def get_solver(self):
        return self.solver


# ---------------------------
# Plasticity code
# ---------------------------
def setup_elastic_plastic_johnsoncook_model(pa):
    props = 'ipst_x ipst_y ipst_z ipst_dx ipst_dy ipst_dz'.split()

    for prop in props:
        pa.add_property(prop)

    pa.add_constant('ipst_chi0', 0.)
    pa.add_property('ipst_chi')

    equations = [
        Group(
            equations=[
                CheckUniformityIPST(dest=pa.name, sources=[pa.name]),
            ], real=False),
    ]

    sph_eval = SPHEvaluator(arrays=[pa], equations=equations, dim=2,
                            kernel=QuinticSpline(dim=2))

    sph_eval.evaluate(dt=0.1)

    pa.ipst_chi0[0] = min(pa.ipst_chi)


def get_poisson_ratio_from_E_G(e, g):
    return e / (2. * g) - 1.


def add_plasticity_properties(pa):
    pa.add_property('plastic_limit')
    pa.add_property('J2')


class ComputeJ2(Equation):
    def initialize(self, d_idx, d_p, s_p, d_s00, d_s01, d_s02, d_s11, d_s12,
                   d_s22, d_J2, d_plastic_limit):
        # J2 = tr(A^T, B)
        # compute the second invariant of deviatoric stress
        d_J2[d_idx] = (
            d_s00[d_idx] * d_s00[d_idx] + d_s01[d_idx] * d_s01[d_idx] +
            d_s02[d_idx] * d_s02[d_idx] + d_s01[d_idx] * d_s01[d_idx] +
            d_s11[d_idx] * d_s11[d_idx] + d_s12[d_idx] * d_s12[d_idx] +
            d_s02[d_idx] * d_s02[d_idx] + d_s12[d_idx] * d_s12[d_idx] +
            d_s22[d_idx] * d_s22[d_idx]) / 2.


class ComputeJohnsonCookYieldStress(Equation):
    """

    Params:

    a, b, c, n, m: material constants
    T_m: melting temperature
    T_tr: transition temperature

    """
    def __init__(self, dest, sources, a, b, c, n, m, T_m, T_tr,
                 plastic_strain_rate_0):
        self.a = a
        self.b = b
        self.c = c
        self.n = n
        self.m = m
        self.T_m = T_m
        self.T_tr = T_tr
        self.plastic_strain_rate_0 = plastic_strain_rate_0
        super(ComputeJohnsonCookYieldStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, s_p, d_s00, d_s01, d_s02, d_s11, d_s12,
                   d_s22, d_J2, d_plastic_strain, d_plastic_strain_rate,
                   d_sigma_yield):
        tmp1 = (self.a + self.b * pow(d_epsilon_p[d_idx], n))
        tmp2 = 1. + self.c * ln(
            d_plastic_strain_rate[d_idx] / self.plastic_strain_rate_0)

        tmp3_1 = d_T[d_idx] - self.T_tr
        tmp3_2 = self.T_m - self.T_tr
        tmp3 = (1. - pow(tmp3_1 / tmp3_2), self.m)

        d_sigma_yield[d_idx] = tmp1 * tmp2 * tmp3


class MieGruneisenEOS(Equation):
    """
    3.2 in [1]
    3.12 in [2]
    [1] The study on performances of kernel types in solid dynamic problems
    by smoothed particle hydrodynamics

    [2] 3D smooth particle hydrodynamics modeling for high velocity
    penetrating impact using GPU: Application to a blunt projectile
    penetrating thin steel plates
    """
    def __init__(self, dest, sources, c0, s, reference_rho, gamma_0):
        self.c0 = c0
        self.s = s
        self.rho0 = reference_rho
        self.gamma_0 = gamma_0
        super(MieGruneisenEOS, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho, d_e, d_p, d_c0_ref, d_rho_ref):
        # 3.12 in [2]
        eta = (d_rho[d_idx] / self.rho0) - 1
        if eta > 0.:
            c1 = self.rho0 * self.c0 * self.c0
            s1 = self.s - 1.
            c2 = c1 * (1. + 2. * s1)
            c3 = c1 * (2. * s1 + 3. * s1 * s1)
            p_H = c1 * eta + c2 * eta * eta + c3 * eta * eta * eta
        else:
            p_H = c1 * eta

        tmp = self.gamma_0 * d_rho[d_idx] * d_e[d_idx]
        d_p[d_idx] = (1. - 0.5 * self.gamma_0 * eta) * p_H + tmp


class LimitDeviatoricStress(Equation):
    def __init__(self, dest, sources, yield_modulus):
        self.yield_modulus = yield_modulus
        super(LimitDeviatoricStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_J2,
                   d_plastic_limit):
        # J2 = tr(A^T, B)
        # compute the second invariant of deviatoric stress
        d_J2[d_idx] = (
            d_s00[d_idx] * d_s00[d_idx] + d_s01[d_idx] * d_s01[d_idx] +
            d_s02[d_idx] * d_s02[d_idx] + d_s01[d_idx] * d_s01[d_idx] +
            d_s11[d_idx] * d_s11[d_idx] + d_s12[d_idx] * d_s12[d_idx] +
            d_s02[d_idx] * d_s02[d_idx] + d_s12[d_idx] * d_s12[d_idx] +
            d_s22[d_idx] * d_s22[d_idx])
        # this is just to check if it is greater than zero. It can be 1e-12
        # because we are dividing it by J2
        if d_J2[d_idx] > 0.1:
            # d_plastic_limit[d_idx] = min(
            #     self.yield_modulus * self.yield_modulus / (3. * d_J2[d_idx]),
            #     1.)
            d_plastic_limit[d_idx] = min(
                self.yield_modulus / sqrt(3. * d_J2[d_idx]), 1.)

        d_s00[d_idx] = d_plastic_limit[d_idx] * d_s00[d_idx]
        d_s01[d_idx] = d_plastic_limit[d_idx] * d_s01[d_idx]
        d_s02[d_idx] = d_plastic_limit[d_idx] * d_s02[d_idx]
        d_s11[d_idx] = d_plastic_limit[d_idx] * d_s11[d_idx]
        d_s12[d_idx] = d_plastic_limit[d_idx] * d_s12[d_idx]
        d_s22[d_idx] = d_plastic_limit[d_idx] * d_s22[d_idx]
