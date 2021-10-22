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


class ElasticSolidContinuityEquation(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, d_u, d_v, d_w, s_idx, s_m, s_u, s_v, s_w,
             DWIJ, VIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_u[s_idx]
        vij[1] = d_v[d_idx] - s_v[s_idx]
        vij[2] = d_w[d_idx] - s_w[s_idx]

        vijdotdwij = DWIJ[0] * vij[0] + DWIJ[1] * vij[1] + DWIJ[2] * vij[2]
        d_arho[d_idx] += s_m[s_idx] * vijdotdwij


class ElasticSolidContinuityEquationUhat(Equation):
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


class ElasticSolidContinuityEquationETVFCorrection(Equation):
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


class ElasticSolidMomentumEquation(Equation):
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


class ElasticSolidComputeAuHatETVFSun2019(Equation):
    def __init__(self, dest, sources, mach_no):
        self.mach_no = mach_no
        super(ElasticSolidComputeAuHatETVFSun2019, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_h, d_auhat, d_avhat, d_awhat,
             d_cs, d_wdeltap, d_n, WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ, dt):
        fab = 0.
        # this value is directly taken from the paper
        R = 0.2

        if d_wdeltap[0] > 0.:
            fab = WIJ / d_wdeltap[0]
            fab = pow(fab, d_n[0])

        tmp = self.mach_no * d_cs[d_idx] * 2. * d_h[d_idx] / dt

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


class HookesDeviatoricStressRate(Equation):
    def initialize(self, d_idx, d_as00, d_as01, d_as02, d_as11, d_as12,
                   d_as22):
        d_as00[d_idx] = 0.0
        d_as01[d_idx] = 0.0
        d_as02[d_idx] = 0.0

        d_as11[d_idx] = 0.0
        d_as12[d_idx] = 0.0

        d_as22[d_idx] = 0.0

    def loop(self, d_idx, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22, d_v00,
             d_v01, d_v02, d_v10, d_v11, d_v12, d_v20, d_v21, d_v22, d_as00,
             d_as01, d_as02, d_as11, d_as12, d_as22, d_G):

        v00 = d_v00[d_idx]
        v01 = d_v01[d_idx]
        v02 = d_v02[d_idx]

        v10 = d_v10[d_idx]
        v11 = d_v11[d_idx]
        v12 = d_v12[d_idx]

        v20 = d_v20[d_idx]
        v21 = d_v21[d_idx]
        v22 = d_v22[d_idx]

        s00 = d_s00[d_idx]
        s01 = d_s01[d_idx]
        s02 = d_s02[d_idx]

        s10 = d_s01[d_idx]
        s11 = d_s11[d_idx]
        s12 = d_s12[d_idx]

        s20 = d_s02[d_idx]
        s21 = d_s12[d_idx]
        s22 = d_s22[d_idx]

        # strain rate tensor is symmetric
        eps00 = v00
        eps01 = 0.5 * (v01 + v10)
        eps02 = 0.5 * (v02 + v20)

        eps10 = eps01
        eps11 = v11
        eps12 = 0.5 * (v12 + v21)

        eps20 = eps02
        eps21 = eps12
        eps22 = v22

        # rotation tensor is asymmetric
        omega00 = 0.0
        omega01 = 0.5 * (v01 - v10)
        omega02 = 0.5 * (v02 - v20)

        omega10 = -omega01
        omega11 = 0.0
        omega12 = 0.5 * (v12 - v21)

        omega20 = -omega02
        omega21 = -omega12
        omega22 = 0.0

        tmp = 2.0 * d_G[d_idx]
        trace = 1.0 / 3.0 * (eps00 + eps11 + eps22)

        # S_00
        d_as00[d_idx] = tmp*( eps00 - trace ) + \
                        ( s00*omega00 + s01*omega01 + s02*omega02) + \
                        ( s00*omega00 + s10*omega01 + s20*omega02)

        # S_01
        d_as01[d_idx] = tmp*(eps01) + \
                        ( s00*omega10 + s01*omega11 + s02*omega12) + \
                        ( s01*omega00 + s11*omega01 + s21*omega02)

        # S_02
        d_as02[d_idx] = tmp*eps02 + \
                        (s00*omega20 + s01*omega21 + s02*omega22) + \
                        (s02*omega00 + s12*omega01 + s22*omega02)

        # S_11
        d_as11[d_idx] = tmp*( eps11 - trace ) + \
                        (s10*omega10 + s11*omega11 + s12*omega12) + \
                        (s01*omega10 + s11*omega11 + s21*omega12)

        # S_12
        d_as12[d_idx] = tmp*eps12 + \
                        (s10*omega20 + s11*omega21 + s12*omega22) + \
                        (s02*omega10 + s12*omega11 + s22*omega12)

        # S_22
        d_as22[d_idx] = tmp*(eps22 - trace) + \
                        (s20*omega20 + s21*omega21 + s22*omega22) + \
                        (s02*omega20 + s12*omega21 + s22*omega22)


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


class ElasticSolidEDACEquation(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(ElasticSolidEDACEquation, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_cs, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_p, s_m, s_rho, d_ap, DWIJ, XIJ, s_uhat, s_vhat,
             s_what, s_u, s_v, s_w, R2IJ, VIJ, EPS):
        vhatij = declare('matrix(3)')
        vhatij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vhatij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vhatij[2] = d_what[d_idx] - s_what[s_idx]

        cs2 = d_cs[d_idx] * d_cs[d_idx]

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


class ElasticSolidEDACEquationSolid(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(ElasticSolidEDACEquationSolid, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_cs, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_p, s_m, s_rho, d_ap, DWIJ, XIJ, s_ubhat,
             s_vbhat, s_wbhat, s_ub, s_vb, s_wb, R2IJ, VIJ, EPS):
        vhatij = declare('matrix(3)')
        vhatij[0] = d_uhat[d_idx] - s_ubhat[s_idx]
        vhatij[1] = d_vhat[d_idx] - s_vbhat[s_idx]
        vhatij[2] = d_what[d_idx] - s_wbhat[s_idx]

        cs2 = d_cs[d_idx] * d_cs[d_idx]

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
        tmp0 = pj * (s_ubhat[s_idx] - s_ub[s_idx]) - pi * (d_uhat[d_idx] -
                                                           d_u[d_idx])

        tmp1 = pj * (s_vbhat[s_idx] - s_vb[s_idx]) - pi * (d_vhat[d_idx] -
                                                           d_v[d_idx])

        tmp2 = pj * (s_wbhat[s_idx] - s_wb[s_idx]) - pi * (d_what[d_idx] -
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


class GTVFSolidMechStepEDAC(SolidMechStep):
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


class RK2SolidMechStepEDAC(IntegratorStep):
    """Predictor corrector Integrator for solid mechanics problems"""
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho,
                   d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                   d_s000, d_s010, d_s020, d_s110, d_s120, d_s220):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

        d_s000[d_idx] = d_s00[d_idx]
        d_s010[d_idx] = d_s01[d_idx]
        d_s020[d_idx] = d_s02[d_idx]
        d_s110[d_idx] = d_s11[d_idx]
        d_s120[d_idx] = d_s12[d_idx]
        d_s220[d_idx] = d_s22[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
               d_aw, d_arho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
               d_s000, d_s010, d_s020, d_s110, d_s120, d_s220,
               d_as00, d_as01, d_as02, d_as11, d_as12, d_as22,
               d_uhat, d_auhat, d_vhat, d_avhat, d_what, d_awhat,
               d_sigma00, d_sigma01,
               d_sigma02, d_sigma11, d_sigma12, d_sigma22, d_p, dt):
        dtb2 = 0.5*dt

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_what[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

        # update deviatoric stress components
        d_s00[d_idx] = d_s000[d_idx] + dtb2 * d_as00[d_idx]
        d_s01[d_idx] = d_s010[d_idx] + dtb2 * d_as01[d_idx]
        d_s02[d_idx] = d_s020[d_idx] + dtb2 * d_as02[d_idx]
        d_s11[d_idx] = d_s110[d_idx] + dtb2 * d_as11[d_idx]
        d_s12[d_idx] = d_s120[d_idx] + dtb2 * d_as12[d_idx]
        d_s22[d_idx] = d_s220[d_idx] + dtb2 * d_as22[d_idx]

        # update sigma
        d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
        d_sigma01[d_idx] = d_s01[d_idx]
        d_sigma02[d_idx] = d_s02[d_idx]
        d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
        d_sigma12[d_idx] = d_s12[d_idx]
        d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw, d_arho, d_s00,
               d_s01, d_s02, d_s11, d_s12, d_s22, d_s000, d_s010, d_s020,
               d_s110, d_s120, d_s220, d_as00, d_as01, d_as02, d_as11, d_as12,
               d_as22, d_uhat, d_auhat, d_vhat, d_avhat, d_what, d_awhat,
               d_sigma00, d_sigma01, d_sigma02, d_sigma11, d_sigma12,
               d_sigma22, d_p, dt):
        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dt * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dt * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dt * d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_what[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]

        # update deviatoric stress components
        d_s00[d_idx] = d_s000[d_idx] + dt * d_as00[d_idx]
        d_s01[d_idx] = d_s010[d_idx] + dt * d_as01[d_idx]
        d_s02[d_idx] = d_s020[d_idx] + dt * d_as02[d_idx]
        d_s11[d_idx] = d_s110[d_idx] + dt * d_as11[d_idx]
        d_s12[d_idx] = d_s120[d_idx] + dt * d_as12[d_idx]
        d_s22[d_idx] = d_s220[d_idx] + dt * d_as22[d_idx]

        # update sigma
        d_sigma00[d_idx] = d_s00[d_idx] - d_p[d_idx]
        d_sigma01[d_idx] = d_s01[d_idx]
        d_sigma02[d_idx] = d_s02[d_idx]
        d_sigma11[d_idx] = d_s11[d_idx] - d_p[d_idx]
        d_sigma12[d_idx] = d_s12[d_idx]
        d_sigma22[d_idx] = d_s22[d_idx] - d_p[d_idx]


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


class ElasticSolidContinuityEquationUhatSolid(Equation):
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


class ElasticSolidContinuityEquationETVFCorrectionSolid(Equation):
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


class ElasticSolidSetWallVelocityNoSlipU(Equation):
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


class ElasticSolidSetWallVelocityNoSlipUhat(Equation):
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


class MonaghanArtificialViscosity(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        super(MonaghanArtificialViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, VIJ, XIJ, HIJ, R2IJ, RHOIJ1, EPS, DWIJ):

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = d_cs[d_idx]

            muij = (HIJ * vijdotxij)/(R2IJ + EPS)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

        d_au[d_idx] += -s_m[s_idx] * piij * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * piij * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * piij * DWIJ[2]


class ElasticSolidMonaghanArtificialViscositySolid(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        super(ElasticSolidMonaghanArtificialViscositySolid, self).__init__(
            dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, d_u, d_v, d_w, s_ub, s_vb, s_wb,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ub[s_idx]
        vij[1] = d_v[d_idx] - s_vb[s_idx]
        vij[2] = d_w[d_idx] - s_wb[s_idx]

        vijdotxij = vij[0]*XIJ[0] + vij[1]*XIJ[1] + vij[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = d_cs[d_idx]

            muij = (HIJ * vijdotxij)/(R2IJ + EPS)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

        d_au[d_idx] += -s_m[s_idx] * piij * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * piij * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * piij * DWIJ[2]


class ElasticSolidMonaghanArtificialViscosityUhatSolid(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        super(ElasticSolidMonaghanArtificialViscosityUhatSolid, self).__init__(
            dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_au, d_av, d_aw, s_m, s_rho,
             s_cs, d_uhat, d_vhat, d_what, s_ubhat, s_vbhat, s_wbhat, XIJ, HIJ,
             R2IJ, RHOIJ1, EPS, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_ubhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vbhat[s_idx]
        vij[2] = d_what[d_idx] - s_wbhat[s_idx]

        vijdotxij = vij[0]*XIJ[0] + vij[1]*XIJ[1] + vij[2]*XIJ[2]

        piij = 0.0
        if vijdotxij < 0:
            cij = d_cs[d_idx]

            muij = (HIJ * vijdotxij)/(R2IJ + EPS)

            piij = -self.alpha*cij*muij + self.beta*muij*muij
            piij = piij*RHOIJ1

        d_au[d_idx] += -s_m[s_idx] * piij * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * piij * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * piij * DWIJ[2]


class MonaghanArtificialViscosityModified(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        super(MonaghanArtificialViscosityModified, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, VIJ, XIJ, HIJ, R2IJ, RHOIJ1, EPS, DWIJ):

        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        piij = 0.0

        cij = d_cs[d_idx]

        muij = (HIJ * vijdotxij)/(R2IJ + EPS)

        piij = -self.alpha*cij*muij + self.beta*muij*muij
        piij = piij*RHOIJ1

        d_au[d_idx] += -s_m[s_idx] * piij * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * piij * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * piij * DWIJ[2]


class ElasticSolidMonaghanArtificialViscositySolidModified(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        super(ElasticSolidMonaghanArtificialViscositySolidModified,
              self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_au, d_av, d_aw, s_m,
             s_rho, s_cs, d_u, d_v, d_w, s_ub, s_vb, s_wb,
             XIJ, HIJ, R2IJ, RHOIJ1, EPS, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ub[s_idx]
        vij[1] = d_v[d_idx] - s_vb[s_idx]
        vij[2] = d_w[d_idx] - s_wb[s_idx]

        vijdotxij = vij[0]*XIJ[0] + vij[1]*XIJ[1] + vij[2]*XIJ[2]

        piij = 0.0
        cij = d_cs[d_idx]

        muij = (HIJ * vijdotxij)/(R2IJ + EPS)

        piij = -self.alpha*cij*muij + self.beta*muij*muij
        piij = piij*RHOIJ1

        d_au[d_idx] += -s_m[s_idx] * piij * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * piij * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * piij * DWIJ[2]


class ElasticSolidMonaghanArtificialViscosityUhatSolidModified(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        super(ElasticSolidMonaghanArtificialViscosityUhatSolidModified,
              self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_cs, d_au, d_av, d_aw, s_m, s_rho,
             s_cs, d_uhat, d_vhat, d_what, s_ubhat, s_vbhat, s_wbhat, XIJ, HIJ,
             R2IJ, RHOIJ1, EPS, DWIJ):
        vij = declare('matrix(3)')
        vij[0] = d_uhat[d_idx] - s_ubhat[s_idx]
        vij[1] = d_vhat[d_idx] - s_vbhat[s_idx]
        vij[2] = d_what[d_idx] - s_wbhat[s_idx]

        vijdotxij = vij[0]*XIJ[0] + vij[1]*XIJ[1] + vij[2]*XIJ[2]

        piij = 0.0
        cij = d_cs[d_idx]

        muij = (HIJ * vijdotxij)/(R2IJ + EPS)

        piij = -self.alpha*cij*muij + self.beta*muij*muij
        piij = piij*RHOIJ1

        d_au[d_idx] += -s_m[s_idx] * piij * DWIJ[0]
        d_av[d_idx] += -s_m[s_idx] * piij * DWIJ[1]
        d_aw[d_idx] += -s_m[s_idx] * piij * DWIJ[2]


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


class BuiFukagawaDampingGraularSPH(Equation):
    def __init__(self, dest, sources, damping_coeff=0.02):
        self.damping_coeff = damping_coeff
        super(BuiFukagawaDampingGraularSPH, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_h, d_rho,
                   d_E, d_u, d_v, d_w):
        tmp1 = d_rho[d_idx] * d_h[d_idx]**2.
        tmp = self.damping_coeff * (d_E[d_idx] / tmp1)**0.5

        d_au[d_idx] -= tmp * d_u[d_idx]
        d_av[d_idx] -= tmp * d_v[d_idx]
        d_aw[d_idx] -= tmp * d_w[d_idx]


class IsothermalEOS(Equation):
    def loop(self, d_idx, d_rho, d_p, d_cs, d_rho_ref):
        d_p[d_idx] = d_cs[d_idx] * d_cs[d_idx] * (d_rho[d_idx] -
                                                  d_rho_ref[d_idx])


class SolidsScheme(Scheme):
    def __init__(self, solids, boundaries, dim, pb, edac_nu, mach_no, hdx,
                 artificial_vis_alpha=1.0, artificial_vis_beta=0.0,
                 artificial_stress_eps=0.3, gamma=7., pst="sun2019", gx=0.,
                 gy=0., gz=0.):
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
        self.kernel = QuinticSpline
        self.kernel_factor = 2

        self.pb = pb

        self.no_boundaries = len(self.boundaries)

        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        self.edac_nu = edac_nu
        self.surf_p_zero = True
        self.edac = False

        self.pst = pst

        # attributes for P Sun 2019 PST technique
        self.mach_no = mach_no

        self.debug = False

        self.gamma = gamma

        # boundary conditions
        self.adami_velocity_extrapolate = False
        self.no_slip = False
        self.free_slip = False
        self.edac = True
        self.solid_velocity_bc = False
        self.solid_stress_bc = False
        self.wall_pst = False
        self.continuity_correction = True
        self.uhat_vgrad = False
        self.integrator = "gtvf"
        self.modified_artificial_viscosity = False
        self.damping = False

        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--artificial-vis-alpha", action="store",
                           dest="artificial_vis_alpha", default=2.0,
                           type=float,
                           help="Artificial viscosity coefficients")

        group.add_argument("--artificial-vis-beta", action="store",
                           dest="artificial_vis_beta", default=0.0, type=float,
                           help="Artificial viscosity coefficients, beta")

        add_bool_argument(
            group, 'solid-velocity-bc', dest='solid_velocity_bc',
            default=False,
            help='Use velocity bc for solid')

        add_bool_argument(
            group, 'solid-stress-bc', dest='solid_stress_bc', default=False,
            help='Use stress bc for solid')

        choices = ['sun2019', 'none']
        group.add_argument(
            "--pst", action="store", dest='pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use (one of %s)." % choices)

        add_bool_argument(
            group, 'edac', dest='edac',
            default=True,
            help='Use edac for pressure computation')

        add_bool_argument(
            group, 'continuity-correction', dest='continuity_correction',
            default=True,
            help='Use correction in continuity equation')

        add_bool_argument(
            group, 'uhat-vgrad', dest='uhat_vgrad',
            default=False,
            help='Use uhat in strain rate computation')

        add_bool_argument(group, 'wall-pst', dest='wall_pst',
                          default=False, help='Add wall as PST source')

        choices = ['rk2', 'gtvf']
        group.add_argument(
            "--integrator", action="store", dest='integrator', default="gtvf",
            choices=choices,
            help="Specify what integrator to use (one of %s)." % choices)

        add_bool_argument(group, 'modified-artificial-viscosity',
                          dest='modified_artificial_viscosity',
                          default=False,
                          help='Use modified artificial viscosity')

        add_bool_argument(group, 'damping',
                          dest='damping',
                          default=False,
                          help='Use damping')

    def consume_user_options(self, options):
        _vars = ['artificial_vis_alpha', 'artificial_vis_beta',
                 'solid_velocity_bc', 'solid_stress_bc', 'pst', 'edac',
                 'wall_pst', 'continuity_correction', 'uhat_vgrad',
                 'integrator', 'modified_artificial_viscosity',
                 'damping']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        if self.integrator == "gtvf":
            return self._get_gtvf_equation()

        if self.integrator == "rk2":
            return self._get_rk2_equation()

    def _get_gtvf_equation(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import (VelocityGradient2D)

        stage1 = []
        g1 = []
        all = list(set(self.solids + self.boundaries))

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #
        if self.solid_velocity_bc is True:
            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipUhat(
                            dest=boundary, sources=self.solids))
                stage1.append(Group(equations=tmp))

            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipU(
                            dest=boundary, sources=self.solids))
                stage1.append(Group(equations=tmp))
        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #

        for solid in self.solids:
            g1.append(ElasticSolidContinuityEquationUhat(
                dest=solid, sources=self.solids))

            if self.continuity_correction is True:
                g1.append(ElasticSolidContinuityEquationETVFCorrection(
                    dest=solid, sources=self.solids))

            if self.edac is True:
                g1.append(ElasticSolidEDACEquation(dest=solid,
                                                   sources=self.solids,
                                                   nu=self.edac_nu))

            if self.uhat_vgrad is True:
                g1.append(VelocityGradient2DUhat(dest=solid, sources=self.solids))
            else:
                g1.append(VelocityGradient2D(dest=solid, sources=self.solids))

            if len(self.boundaries) > 0:
                g1.append(
                    ElasticSolidContinuityEquationUhatSolid(dest=solid,
                                                            sources=self.boundaries))

                if self.continuity_correction is True:
                    g1.append(
                        ElasticSolidContinuityEquationETVFCorrectionSolid(
                            dest=solid, sources=self.boundaries))

                if self.edac is True:
                    g1.append(
                        ElasticSolidEDACEquationSolid(dest=solid,
                                                      sources=self.boundaries,
                                                      nu=self.edac_nu))

            if self.uhat_vgrad is True:
                g1.append(VelocityGradient2DUhatSolid(dest=solid,
                                                      sources=self.boundaries))
            else:
                g1.append(
                    VelocityGradient2DSolid(dest=solid, sources=self.boundaries))

        stage1.append(Group(equations=g1))

        # --------------------
        # solid equations
        # --------------------
        g2 = []
        for solid in self.solids:
            g2.append(HookesDeviatoricStressRate(dest=solid, sources=None))

        stage1.append(Group(equations=g2))

        # ------------------------
        # stage 2 equations starts
        # ------------------------

        stage2 = []
        g1 = []
        g2 = []
        g3 = []
        g4 = []

        if self.pst in ["sun2019"]:
            for solid in self.solids:
                g1.append(
                    SetHIJForInsideParticles(dest=solid, sources=[solid],
                                             kernel_factor=self.kernel_factor))
            stage2.append(Group(g1))

        tmp = []
        if self.edac is False:
            for solid in self.solids:
                tmp.append(IsothermalEOS(dest=solid, sources=None))
            stage2.append(Group(tmp))
        # -------------------
        # boundary conditions
        # -------------------
        if self.solid_stress_bc is True:
            for boundary in self.boundaries:
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
            if self.modified_artificial_viscosity is False:
                if self.artificial_vis_alpha > 0. or self.artificial_vis_beta > 0.:
                    g4.append(
                        MonaghanArtificialViscosity(
                            dest=solid, sources=[solid],
                            alpha=self.artificial_vis_alpha,
                            beta=self.artificial_vis_beta))

                    if len(self.boundaries) > 0:
                        if self.uhat_vgrad is True:
                            g4.append(
                                ElasticSolidMonaghanArtificialViscosityUhatSolid(
                                    dest=solid, sources=self.boundaries,
                                    alpha=self.artificial_vis_alpha,
                                    beta=self.artificial_vis_beta))
                        else:
                            g4.append(
                                ElasticSolidMonaghanArtificialViscositySolid(
                                    dest=solid, sources=self.boundaries,
                                    alpha=self.artificial_vis_alpha,
                                    beta=self.artificial_vis_beta))

            elif self.modified_artificial_viscosity is True:
                if self.artificial_vis_alpha > 0. or self.artificial_vis_beta > 0.:
                    g4.append(
                        MonaghanArtificialViscosityModified(
                            dest=solid, sources=[solid],
                            alpha=self.artificial_vis_alpha,
                            beta=self.artificial_vis_beta))

                    if len(self.boundaries) > 0:
                        if self.uhat_vgrad is True:
                            g4.append(
                                ElasticSolidMonaghanArtificialViscosityUhatSolidModified(
                                    dest=solid, sources=self.boundaries,
                                    alpha=self.artificial_vis_alpha,
                                    beta=self.artificial_vis_beta))
                        else:
                            g4.append(
                                ElasticSolidMonaghanArtificialViscositySolidModified(
                                    dest=solid, sources=self.boundaries,
                                    alpha=self.artificial_vis_alpha,
                                    beta=self.artificial_vis_beta))

            g4.append(ElasticSolidMomentumEquation(dest=solid, sources=all))

            if self.pst == "sun2019":
                if self.wall_pst is True:
                    g4.append(
                        ElasticSolidComputeAuHatETVFSun2019(
                            dest=solid, sources=[solid] + self.boundaries,
                            mach_no=self.mach_no))
                else:
                    g4.append(
                        ElasticSolidComputeAuHatETVFSun2019(
                            dest=solid, sources=[solid],
                            mach_no=self.mach_no))

        stage2.append(Group(g4))

        g9 = []
        for solid in self.solids:
            g9.append(AddGravityToStructure(dest=solid, sources=None,
                                            gx=self.gx,
                                            gy=self.gy,
                                            gz=self.gz))

            if self.damping == True:
                g9.append(
                    BuiFukagawaDampingGraularSPH(
                        dest=solid, sources=None, damping_coeff=0.02))

        stage2.append(Group(equations=g9))

        return MultiStageEquations([stage1, stage2])

    def _get_rk2_equation(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import (VelocityGradient2D)

        stage1 = []
        g1 = []
        all = list(set(self.solids + self.boundaries))

        if self.solid_velocity_bc is True:
            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipUhat(
                            dest=boundary, sources=self.solids))
                stage1.append(Group(equations=tmp))

            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipU(
                            dest=boundary, sources=self.solids))
                stage1.append(Group(equations=tmp))

        if self.solid_stress_bc is True:
            tmp = []
            for boundary in self.boundaries:
                tmp.append(
                    AdamiBoundaryConditionExtrapolateNoSlip(
                        dest=boundary, sources=self.solids,
                        gx=self.gx, gy=self.gy, gz=self.gz
                    ))
            if len(tmp) > 0:
                stage1.append(Group(tmp))

        tmp = []
        for solid in self.solids:
            tmp.append(IsothermalEOS(dest=solid, sources=None))
        stage1.append(Group(tmp))

        g1 = []
        if self.pst in ["sun2019"]:
            for solid in self.solids:
                g1.append(
                    SetHIJForInsideParticles(dest=solid, sources=[solid],
                                             kernel_factor=self.kernel_factor))
            stage1.append(Group(g1))

        g1 = []
        for solid in self.solids:
            g1.append(ElasticSolidContinuityEquationUhat(
                dest=solid, sources=self.solids))

            if self.continuity_correction is True:
                g1.append(ElasticSolidContinuityEquationETVFCorrection(
                    dest=solid, sources=self.solids))

            if self.edac is True:
                g1.append(ElasticSolidEDACEquation(dest=solid,
                                                   sources=self.solids,
                                                   nu=self.edac_nu))

            if self.uhat_vgrad is True:
                g1.append(VelocityGradient2DUhat(dest=solid, sources=self.solids))
            else:
                g1.append(VelocityGradient2D(dest=solid, sources=self.solids))

            if len(self.boundaries) > 0:
                g1.append(
                    ElasticSolidContinuityEquationUhatSolid(dest=solid,
                                                            sources=self.boundaries))

                if self.continuity_correction is True:
                    g1.append(
                        ElasticSolidContinuityEquationETVFCorrectionSolid(
                            dest=solid, sources=self.boundaries))

                if self.edac is True:
                    g1.append(
                        ElasticSolidEDACEquationSolid(dest=solid,
                                                      sources=self.boundaries,
                                                      nu=self.edac_nu))

            if self.uhat_vgrad is True:
                g1.append(VelocityGradient2DUhatSolid(dest=solid,
                                                      sources=self.boundaries))
            else:
                g1.append(
                    VelocityGradient2DSolid(dest=solid, sources=self.boundaries))

            # add only if there is some positive value
            if self.artificial_vis_alpha > 0. or self.artificial_vis_beta > 0.:
                g1.append(
                    MonaghanArtificialViscosity(
                        dest=solid, sources=[solid],
                        alpha=self.artificial_vis_alpha,
                        beta=self.artificial_vis_beta))

                if len(self.boundaries) > 0:
                    if self.uhat_vgrad is True:
                        g1.append(
                            ElasticSolidMonaghanArtificialViscosityUhatSolid(
                                dest=solid, sources=self.boundaries,
                                alpha=self.artificial_vis_alpha,
                                beta=self.artificial_vis_beta))
                    else:
                        g1.append(
                            ElasticSolidMonaghanArtificialViscositySolid(
                                dest=solid, sources=self.boundaries,
                                alpha=self.artificial_vis_alpha,
                                beta=self.artificial_vis_beta))

            g1.append(ElasticSolidMomentumEquation(dest=solid, sources=all))

            if self.pst == "sun2019":
                if self.wall_pst is True:
                    g1.append(
                        ElasticSolidComputeAuHatETVFSun2019(
                            dest=solid, sources=[solid] + self.boundaries,
                            mach_no=self.mach_no))
                else:
                    g1.append(
                        ElasticSolidComputeAuHatETVFSun2019(
                            dest=solid, sources=[solid],
                            mach_no=self.mach_no))

        stage1.append(Group(equations=g1))

        g9 = []
        for solid in self.solids:
            g9.append(AddGravityToStructure(dest=solid, sources=None,
                                            gx=self.gx,
                                            gy=self.gy,
                                            gz=self.gz))

        stage1.append(Group(equations=g9))

        # --------------------
        # solid equations
        # --------------------
        g2 = []
        for solid in self.solids:
            g2.append(HookesDeviatoricStressRate(dest=solid, sources=None))

        stage1.append(Group(equations=g2))

        return stage1

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
        from pysph.sph.integrator import EPECIntegrator

        if self.integrator == "gtvf":
            cls = integrator_cls if integrator_cls is not None else GTVFIntegrator
            step_cls = GTVFSolidMechStepEDAC

        elif self.integrator == "rk2":
            cls = integrator_cls if integrator_cls is not None else EPECIntegrator
            step_cls = RK2SolidMechStepEDAC

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
            # we expect the solid to have Young's modulus, Poisson ration as
            # given
            pa = pas[solid]

            # add the properties that are used by all the schemes
            add_properties(pa, 'cs', 'v00', 'v01', 'v02', 'v10', 'v11', 'v12',
                           'v20', 'v21', 'v22', 's00', 's01', 's02', 's11',
                           's12', 's22', 'as00', 'as01', 'as02', 'as11',
                           'as12', 'as22', 'arho', 'au', 'av', 'aw', 'ap',
                           'rho_ref')

            if self.integrator == "rk2":
                add_properties(pa, 'rho0', 's000', 's010', 's020', 's110',
                               's120', 's220', 'x0', 'y0', 'z0', 'u0', 'v0',
                               'w0')

            # for isothermal eqn
            pa.rho_ref[:] = pa.rho[:]

            # this will change
            kernel = self.kernel(dim=2)
            wdeltap = kernel.kernel(rij=pa.spacing0[0], h=pa.h[0])
            pa.add_constant('wdeltap', wdeltap)

            # set the shear modulus G
            pa.add_property('G')

            # set the speed of sound
            for i in range(len(pa.x)):
                cs = get_speed_of_sound(pa.E[i], pa.nu[i], pa.rho_ref[i])
                G = get_shear_modulus(pa.E[i], pa.nu[i])

                pa.G[i] = G
                pa.cs[i] = cs

            # auhat properties are needed for gtvf, etvf but not for gray. But
            # for the compatability with the integrator we will add
            add_properties(pa, 'auhat', 'avhat', 'awhat', 'uhat', 'vhat',
                           'what')

            add_properties(pa, 'sigma00', 'sigma01', 'sigma02', 'sigma11',
                           'sigma12', 'sigma22')

            # output arrays
            pa.add_output_arrays(['sigma00', 'sigma01', 'sigma11'])

            # for boundary identification and for sun2019 pst
            pa.add_property('normal', stride=3)
            pa.add_property('normal_tmp', stride=3)
            pa.add_property('normal_norm')

            # check for boundary particle
            pa.add_property('is_boundary', type='int')

            # used to set the particles near the boundary
            pa.add_property('h_b')

            # for edac
            add_properties(pa, 'ap')

            pa.add_output_arrays(['p'])

        for boundary in self.boundaries:
            pa = pas[boundary]
            pa.add_property('wij')

            # for adami boundary condition

            add_properties(pa, 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 's00',
                           's01', 's02', 's11', 's12', 's22', 'cs', 'uhat',
                           'vhat', 'what', 'ub', 'vb', 'wb', 'ubhat', 'vbhat',
                           'wbhat')

            pa.add_property('ughat')
            pa.add_property('vghat')
            pa.add_property('wghat')

            if self.surf_p_zero == True:
                pa.add_property('edac_normal', stride=3)
                pa.add_property('edac_normal_tmp', stride=3)
                pa.add_property('edac_normal_norm')

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


class SolidsSchemeGray(Scheme):
    def __init__(self, solids, boundaries, dim, pb, edac_nu, mach_no, hdx,
                 artificial_vis_alpha=1.0, artificial_vis_beta=0.0,
                 artificial_stress_eps=0.3, gamma=7., pst="sun2019", gx=0.,
                 gy=0., gz=0.):
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
        self.kernel = QuinticSpline
        self.kernel_factor = 2

        self.pb = pb

        self.no_boundaries = len(self.boundaries)

        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        self.edac_nu = edac_nu
        self.surf_p_zero = True
        self.edac = False

        self.pst = pst

        # attributes for P Sun 2019 PST technique
        self.mach_no = mach_no

        self.debug = False

        self.gamma = gamma

        # boundary conditions
        self.adami_velocity_extrapolate = False
        self.no_slip = False
        self.free_slip = False
        self.edac = True
        self.solid_velocity_bc = False
        self.solid_stress_bc = False
        self.wall_pst = False

        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--artificial-vis-alpha", action="store",
                           dest="artificial_vis_alpha", default=2.0,
                           type=float,
                           help="Artificial viscosity coefficients")

        group.add_argument("--artificial-vis-beta", action="store",
                           dest="artificial_vis_beta", default=0.0, type=float,
                           help="Artificial viscosity coefficients, beta")

        add_bool_argument(
            group, 'solid-velocity-bc', dest='solid_velocity_bc',
            default=False,
            help='Use velocity bc for solid')

        add_bool_argument(
            group, 'solid-stress-bc', dest='solid_stress_bc', default=False,
            help='Use stress bc for solid')

        choices = ['sun2019', 'none']
        group.add_argument(
            "--pst", action="store", dest='pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use (one of %s)." % choices)

        add_bool_argument(
            group, 'edac', dest='edac',
            default=True,
            help='Use edac for pressure computation')

        add_bool_argument(group, 'wall-pst', dest='wall_pst',
                          default=False, help='Add wall as PST source')

    def consume_user_options(self, options):
        _vars = ['artificial_vis_alpha', 'artificial_vis_beta',
                 'solid_velocity_bc', 'solid_stress_bc', 'pst', 'edac',
                 'wall_pst']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        from pysph.sph.basic_equations import (
            ContinuityEquation, MonaghanArtificialViscosity, XSPHCorrection,
            VelocityGradient2D)

        from pysph.sph.solid_mech.basic import (
            MomentumEquationWithStress,
            HookesDeviatoricStressRate, MonaghanArtificialStress)

        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import (VelocityGradient2D)

        stage1 = []
        g1 = []
        all = list(set(self.solids + self.boundaries))

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #
        if self.solid_velocity_bc is True:
            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        SetWallVelocityNoSlipUhatSolidMech(dest=boundary,
                                                           sources=self.solids))
                stage1.append(Group(equations=tmp))

            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        SetWallVelocityNoSlipUSolidMech(dest=boundary,
                                                        sources=self.solids))
                stage1.append(Group(equations=tmp))
        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #

        for solid in self.solids:
            g1.append(ContinuityEquation(dest=solid,
                                         sources=self.solids+self.boundaries))

            if self.edac is True:
                g1.append(EDACEquation(dest=solid,
                                       sources=self.solids+self.boundaries,
                                       nu=self.edac_nu))

            g1.append(VelocityGradient2D(dest=solid,
                                         sources=self.solids+self.boundaries))

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

        # ------------------------
        # stage 2 equations starts
        # ------------------------

        stage2 = []
        g1 = []
        g2 = []
        g3 = []
        g4 = []

        tmp = []
        if self.edac is False:
            for solid in self.solids:
                tmp.append(IsothermalEOS(dest=solid, sources=None))
            stage2.append(Group(tmp))
        # -------------------
        # boundary conditions
        # -------------------
        if self.solid_stress_bc is True:
            for boundary in self.boundaries:
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
                        dest=solid, sources=[solid],
                        alpha=self.artificial_vis_alpha,
                        beta=self.artificial_vis_beta))

            g4.append(MomentumEquationWithStress(dest=solid, sources=all))
        stage2.append(Group(equations=g4))

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

        step_cls = SolidMechStepEDAC

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

        pas = dict([(p.name, p) for p in particles])

        for solid in self.solids:
            # we expect the solid to have Young's modulus, Poisson ration as
            # given
            pa = pas[solid]

            # add the properties that are used by all the schemes
            add_properties(pa, 'cs', 'v00', 'v01', 'v02', 'v10', 'v11', 'v12',
                           'v20', 'v21', 'v22', 's00', 's01', 's02', 's11',
                           's12', 's22', 'as00', 'as01', 'as02', 'as11',
                           'as12', 'as22', 'arho', 'au', 'av', 'aw', 'ap',
                           'rho_ref', 'r00', 'r11', 'r02', 'r01', 'r12',
                           'r22')

            # for isothermal eqn
            pa.rho_ref[:] = pa.rho[:]

            # this will change
            kernel = self.kernel(dim=2)
            wdeltap = kernel.kernel(rij=pa.spacing0[0], h=pa.h[0])
            pa.add_constant('wdeltap', wdeltap)

            # set the shear modulus G
            pa.add_property('G')

            # set the speed of sound
            for i in range(len(pa.x)):
                cs = get_speed_of_sound(pa.E[i], pa.nu[i], pa.rho_ref[i])
                G = get_shear_modulus(pa.E[i], pa.nu[i])

                pa.G[i] = G
                pa.cs[i] = cs

            # auhat properties are needed for gtvf, etvf but not for gray. But
            # for the compatability with the integrator we will add
            add_properties(pa, 'auhat', 'avhat', 'awhat', 'uhat', 'vhat',
                           'what')

            add_properties(pa, 'sigma00', 'sigma01', 'sigma02', 'sigma11',
                           'sigma12', 'sigma22')

            # output arrays
            pa.add_output_arrays(['sigma00', 'sigma01', 'sigma11'])

            # for boundary identification and for sun2019 pst
            pa.add_property('normal', stride=3)
            pa.add_property('normal_tmp', stride=3)
            pa.add_property('normal_norm')

            # check for boundary particle
            pa.add_property('is_boundary', type='int')

            # used to set the particles near the boundary
            pa.add_property('h_b')

            # for edac
            add_properties(pa, 'ap')

            pa.add_output_arrays(['p'])

        for boundary in self.boundaries:
            pa = pas[boundary]
            pa.add_property('wij')

            # for adami boundary condition

            add_properties(pa, 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 's00',
                           's01', 's02', 's11', 's12', 's22', 'cs', 'uhat',
                           'vhat', 'what', 'ub', 'vb', 'wb', 'ubhat', 'vbhat',
                           'wbhat', 'r00', 'r11', 'r02', 'r01', 'r12', 'r22')

            pa.add_property('ughat')
            pa.add_property('vghat')
            pa.add_property('wghat')

            if self.surf_p_zero == True:
                pa.add_property('edac_normal', stride=3)
                pa.add_property('edac_normal_tmp', stride=3)
                pa.add_property('edac_normal_norm')

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
