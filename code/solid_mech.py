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


class MakeAuhatZero(Equation):
    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.
        d_avhat[d_idx] = 0.
        d_awhat[d_idx] = 0.


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
    def __init__(self, solids, boundaries, dim, mach_no,
                 artificial_vis_alpha=1.0, artificial_vis_beta=0.0,
                 pst="sun2019", gx=0., gy=0., gz=0.):
        # Particle arrays
        self.solids = solids
        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        # general parameters
        self.dim = dim

        self.kernel = QuinticSpline
        self.kernel_factor = 2

        # parameters required by equations
        # Artificial viscosity
        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        # Homogenization force
        self.pst = pst
        self.mach_no = mach_no

        # boundary conditions
        self.solid_velocity_bc = False
        self.solid_stress_bc = False
        self.wall_pst = False
        self.damping = False
        self.damping_coeff = 0.002

        # Gravity equation
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

        add_bool_argument(group, 'wall-pst', dest='wall_pst',
                          default=False, help='Add wall as PST source')

        add_bool_argument(group, 'damping',
                          dest='damping',
                          default=False,
                          help='Use damping')

        group.add_argument("--damping-coeff", action="store",
                           dest="damping_coeff", default=0.002, type=float,
                           help="Damping coefficient for Bui")

    def consume_user_options(self, options):
        _vars = ['artificial_vis_alpha', 'artificial_vis_beta',
                 'solid_velocity_bc', 'solid_stress_bc', 'pst',
                 'wall_pst', 'damping', 'damping_coeff']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        return self._get_gtvf_equation()

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
                        ElasticSolidSetWallVelocityNoSlipU(
                            dest=boundary, sources=self.solids))
                stage1.append(Group(equations=tmp))

            tmp = []
            if len(self.boundaries) > 0:
                for boundary in self.boundaries:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipUhat(
                            dest=boundary, sources=self.solids))
                stage1.append(Group(equations=tmp))

        # =================================== #
        # solid velocity extrapolation ends
        # =================================== #

        for solid in self.solids:
            g1.append(ElasticSolidContinuityEquationUhat(
                dest=solid, sources=self.solids))

            g1.append(ElasticSolidContinuityEquationETVFCorrection(
                dest=solid, sources=self.solids))

            g1.append(VelocityGradient2D(dest=solid, sources=self.solids))

            if len(self.boundaries) > 0:
                g1.append(
                    ElasticSolidContinuityEquationUhatSolid(dest=solid,
                                                            sources=self.boundaries))

                g1.append(
                    ElasticSolidContinuityEquationETVFCorrectionSolid(
                        dest=solid, sources=self.boundaries))

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
                        dest=solid, sources=[solid]+self.boundaries,
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
                        dest=solid, sources=None,
                        damping_coeff=self.damping_coeff))

        stage2.append(Group(equations=g9))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        kernel = self.kernel(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.wc.gtvf import GTVFIntegrator

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator
        step_cls = GTVFSolidMechStepEDAC

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
