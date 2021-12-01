"""
Run it by

python lid_driven_cavity.py --openmp --scheme etvf --integrator pec --internal-flow --pst sun2019 --re 100 --tf 25 --nx 50 --no-edac -d lid_driven_cavity_scheme_etvf_integrator_pec_pst_sun2019_re_100_nx_50_no_edac_output --detailed-output --pfreq 100

"""
import numpy
import numpy as np

from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)

from pysph.sph.wc.edac import (SolidWallPressureBC)
from pysph.sph.wc.edac import ClampWallPressure

from pysph.sph.integrator import PECIntegrator
from boundary_particles import (add_boundary_identification_properties)

from boundary_particles import (ComputeNormals, SmoothNormals,
                                IdentifyBoundaryParticleCosAngleEDAC)

from common import EDACIntegrator
from pysph.examples.solid_mech.impact import add_properties
from pysph.sph.wc.linalg import mat_vec_mult
from pysph.sph.basic_equations import (MonaghanArtificialViscosity,
                                       VelocityGradient3D, VelocityGradient2D)
from pysph.sph.solid_mech.basic import (IsothermalEOS,
                                        HookesDeviatoricStressRate)
from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from pysph.sph.wc.transport_velocity import (MomentumEquationArtificialViscosity)
from pysph.sph.integrator import Integrator

from pysph.sph.wc.basic import TaitEOS


class SetMassFracRhoFracPressureFrac(Equation):
    """
    This equation is applied to structure and structure solids
    """
    def initialize(self, d_m_frac, d_rho_frac, d_p_frac, d_m, d_m_fsi, d_rho,
                   d_rho_fsi, d_p_fsi, d_p, d_idx):
        d_m_frac[d_idx] = d_m_fsi[d_idx] / d_m[d_idx]
        d_rho_frac[d_idx] = d_rho_fsi[d_idx] / d_rho[d_idx]
        d_p_frac[d_idx] = d_p_fsi[d_idx] / d_p[d_idx]


class FluidContinuityEquationGTVFFSI(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m_fsi, d_rho, s_rho_fsi, d_uhat, d_vhat,
             d_what, s_uhat, s_vhat, s_what, d_arho, DWIJ):
        uhatij = d_uhat[d_idx] - s_uhat[s_idx]
        vhatij = d_vhat[d_idx] - s_vhat[s_idx]
        whatij = d_what[d_idx] - s_what[s_idx]

        udotdij = DWIJ[0]*uhatij + DWIJ[1]*vhatij + DWIJ[2]*whatij
        fac = d_rho[d_idx] * s_m_fsi[s_idx] / s_rho_fsi[s_idx]
        d_arho[d_idx] += fac * udotdij


class FluidContinuityEquationETVFCorrectionFSI(Equation):
    def loop(self, d_idx, d_arho, d_rho, d_u, d_v, d_w, d_uhat, d_vhat, d_what,
             s_idx, s_rho_fsi, s_m_fsi, s_u, s_v, s_w, s_uhat, s_vhat, s_what,
             DWIJ):
        tmp0 = s_rho_fsi[s_idx] * (s_uhat[s_idx] -
                                   s_u[s_idx]) - d_rho[d_idx] * (
                                       d_uhat[d_idx] - d_u[d_idx])

        tmp1 = s_rho_fsi[s_idx] * (s_vhat[s_idx] -
                                   s_v[s_idx]) - d_rho[d_idx] * (
                                       d_vhat[d_idx] - d_v[d_idx])

        tmp2 = s_rho_fsi[s_idx] * (s_what[s_idx] -
                                   s_w[s_idx]) - d_rho[d_idx] * (
                                       d_what[d_idx] - d_w[d_idx])

        vijdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)

        d_arho[d_idx] += s_m_fsi[s_idx] / s_rho_fsi[s_idx] * vijdotdwij


class FluidContinuityEquationGTVFFSISolid(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m_fsi, d_rho, s_rho_fsi, d_uhat, d_vhat,
             d_what, s_ughatfs, s_vghatfs, s_wghatfs, d_arho, DWIJ):
        uhatij = d_uhat[d_idx] - s_ughatfs[s_idx]
        vhatij = d_vhat[d_idx] - s_vghatfs[s_idx]
        whatij = d_what[d_idx] - s_wghatfs[s_idx]

        udotdij = DWIJ[0] * uhatij + DWIJ[1] * vhatij + DWIJ[2] * whatij
        fac = d_rho[d_idx] * s_m_fsi[s_idx] / s_rho_fsi[s_idx]
        d_arho[d_idx] += fac * udotdij


class FluidContinuityEquationETVFCorrectionFSISolid(Equation):
    def loop(self, d_idx, d_arho, d_rho, d_u, d_v, d_w, d_uhat, d_vhat, d_what,
             s_idx, s_rho_fsi, s_m_fsi, s_ugfs, s_vgfs, s_wgfs, s_ughatfs,
             s_vghatfs, s_wghatfs, DWIJ):
        tmp0 = s_rho_fsi[s_idx] * (s_ughatfs[s_idx] -
                                   s_ugfs[s_idx]) - d_rho[d_idx] * (
                                       d_uhat[d_idx] - d_u[d_idx])

        tmp1 = s_rho_fsi[s_idx] * (s_vghatfs[s_idx] -
                                   s_vgfs[s_idx]) - d_rho[d_idx] * (
                                       d_vhat[d_idx] - d_v[d_idx])

        tmp2 = s_rho_fsi[s_idx] * (s_wghatfs[s_idx] -
                                   s_wgfs[s_idx]) - d_rho[d_idx] * (
                                       d_what[d_idx] - d_w[d_idx])

        vijdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)

        d_arho[d_idx] += s_m_fsi[s_idx] / s_rho_fsi[s_idx] * vijdotdwij


class FluidEDACEquationFSI(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidEDACEquationFSI, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_cs, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_p_fsi, s_m_fsi, s_rho_fsi, d_ap, DWIJ, XIJ,
             s_uhat, s_vhat, s_what, s_u, s_v, s_w, R2IJ, VIJ, EPS):
        vhatij = declare('matrix(3)')
        vhatij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vhatij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vhatij[2] = d_what[d_idx] - s_what[s_idx]

        cs2 = d_cs[d_idx] * d_cs[d_idx]

        rhoj1 = 1.0 / s_rho_fsi[s_idx]
        Vj = s_m_fsi[s_idx] * rhoj1
        rhoi = d_rho[d_idx]
        pi = d_p[d_idx]
        rhoj = s_rho_fsi[s_idx]
        pj = s_p_fsi[s_idx]

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
        rhoij = d_rho[d_idx] + s_rho_fsi[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p_fsi[s_idx])


class FluidEDACEquationFSISolid(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidEDACEquationFSISolid, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_c0_ref, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_p_fsi, s_m_fsi, s_rho_fsi, d_ap, DWIJ, XIJ,
             s_ughatfs, s_vghatfs, s_wghatfs, s_ugfs, s_vgfs, s_wgfs, R2IJ,
             EPS):
        vhatij = declare('matrix(3)')
        vij = declare('matrix(3)')
        vhatij[0] = d_uhat[d_idx] - s_ughatfs[s_idx]
        vhatij[1] = d_vhat[d_idx] - s_vghatfs[s_idx]
        vhatij[2] = d_what[d_idx] - s_wghatfs[s_idx]

        vij[0] = d_u[d_idx] - s_ugfs[s_idx]
        vij[1] = d_v[d_idx] - s_vgfs[s_idx]
        vij[2] = d_w[d_idx] - s_wgfs[s_idx]

        cs2 = d_c0_ref[0] * d_c0_ref[0]

        rhoj1 = 1.0 / s_rho_fsi[s_idx]
        Vj = s_m_fsi[s_idx] * rhoj1
        rhoi = d_rho[d_idx]
        pi = d_p[d_idx]
        # rhoj = s_rho[s_idx]
        pj = s_p_fsi[s_idx]

        vij_dot_dwij = -(vij[0] * DWIJ[0] + vij[1] * DWIJ[1] +
                         vij[2] * DWIJ[2])

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
        tmp0 = pj * (s_ughatfs[s_idx] - s_ugfs[s_idx]) - pi * (d_uhat[d_idx] -
                                                               d_u[d_idx])

        tmp1 = pj * (s_vghatfs[s_idx] - s_vgfs[s_idx]) - pi * (d_vhat[d_idx] -
                                                               d_v[d_idx])

        tmp2 = pj * (s_wghatfs[s_idx] - s_wgfs[s_idx]) - pi * (d_what[d_idx] -
                                                               d_w[d_idx])

        tmpdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)
        d_ap[d_idx] += -Vj * tmpdotdwij

        #######################################################
        # fourth term on the rhs of Eq 19 of the current paper #
        #######################################################
        rhoij = d_rho[d_idx] + s_rho_fsi[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p_fsi[s_idx])


class FluidSolidWallPressureBCFSI(Equation):
    r"""Solid wall pressure boundary condition from Adami and Hu (transport
    velocity formulation).

    """
    def __init__(self, dest, sources, rho_0, p_0, gx=0.0, gy=0.0, gz=0.0):
        self.rho_0 = rho_0
        self.p_0 = p_0
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(FluidSolidWallPressureBCFSI, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p_fsi):
        d_p_fsi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p_fsi, s_p, s_rho, d_au, d_av, d_aw, WIJ,
             XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        max_val = max(0., gdotxij)
        d_p_fsi[d_idx] += s_p[s_idx] * WIJ + s_rho[s_idx] * max_val * WIJ

    def post_loop(self, d_idx, d_wij, d_p_fsi, d_rho_fsi):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p_fsi[d_idx] /= d_wij[d_idx]

        d_rho_fsi[d_idx] = self.rho_0 * (d_p_fsi[d_idx] / self.p_0 + 1.)


class ClampWallPressureWall(Equation):
    r"""Clamp the wall pressure to non-negative values.
    """
    def __init__(self, dest, sources, rho_0, p_0):
        self.rho_0 = rho_0
        self.p_0 = p_0

        super(ClampWallPressureWall, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_p, d_rho):
        if d_p[d_idx] < 0.0:
            d_p[d_idx] = 0.0

        d_rho[d_idx] = self.rho_0 * (d_p[d_idx] / self.p_0 + 1.)


class ClampWallPressureFSI(Equation):
    r"""Clamp the wall pressure to non-negative values.
    """
    def __init__(self, dest, sources, rho_0, p_0):
        self.rho_0 = rho_0
        self.p_0 = p_0

        super(ClampWallPressureFSI, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_p_fsi, d_rho_fsi):
        if d_p_fsi[d_idx] < 0.0:
            d_p_fsi[d_idx] = 0.0

        d_rho_fsi[d_idx] = self.rho_0 * (d_p_fsi[d_idx] / self.p_0 + 1.)


class FluidComputeAuHatGTVF(Equation):
    def __init__(self, dest, sources):
        super(FluidComputeAuHatGTVF, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat, d_p0, d_p, d_p_ref):
        d_p0[d_idx] = min(10. * abs(d_p[d_idx]), d_p_ref[0])

        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_p0, s_rho, s_m, s_m_frac, d_auhat,
             d_avhat, d_awhat, WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ, HIJ):
        dwijhat = declare('matrix(3)')

        rhoa = d_rho[d_idx]
        mb = s_m[s_idx] * s_m_frac[s_idx]

        rhoa21 = 1. / (rhoa * rhoa)

        # add the background pressure acceleration
        tmp = -d_p0[d_idx] * mb * rhoa21

        SPH_KERNEL.gradient(XIJ, RIJ, 0.5 * HIJ, dwijhat)

        d_auhat[d_idx] += tmp * dwijhat[0]
        d_avhat[d_idx] += tmp * dwijhat[1]
        d_awhat[d_idx] += tmp * dwijhat[2]


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


class AccelerationOnFluidDueToStructure(Equation):
    def loop(self, d_rho, s_rho_fsi, d_idx, s_idx, d_p, s_p_fsi, s_m, s_m_fsi,
             d_au, d_av, d_aw, DWIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho_fsi[s_idx] * s_rho_fsi[s_idx]

        pij = d_p[d_idx] / rhoi2 + s_p_fsi[s_idx] / rhoj2

        tmp = -s_m_fsi[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class AccelerationOnFluidDueToStructureRogersConnor(Equation):
    def loop(self, d_rho, s_rho_fsi, d_idx, s_idx, d_p, s_p_fsi, s_m, s_m_fsi,
             d_au, d_av, d_aw, DWIJ):
        rhoi = d_rho[d_idx]
        rhoj = s_rho_fsi[s_idx]

        pij = (d_p[d_idx] + s_p_fsi[s_idx]) / (rhoi * rhoj)

        tmp = -s_m_fsi[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class AccelerationOnStructureDueToFluid(Equation):
    def initialize(self, d_idx, d_au_fluid, d_av_fluid, d_aw_fluid):
        d_au_fluid[d_idx] = 0.
        d_av_fluid[d_idx] = 0.
        d_aw_fluid[d_idx] = 0.

    def loop(self, d_rho_fsi, s_rho, d_idx, d_m, d_m_fsi, s_idx, d_p_fsi, s_p,
             s_m, d_au, d_av, d_aw, d_au_fluid, d_av_fluid, d_aw_fluid, DWIJ):
        rhoi2 = d_rho_fsi[d_idx] * d_rho_fsi[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p_fsi[d_idx] / rhoi2 + s_p[s_idx] / rhoj2

        tmp = -s_m[s_idx] * pij * d_m_fsi[d_idx] / d_m[d_idx]

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]

        d_au_fluid[d_idx] += tmp * DWIJ[0]
        d_av_fluid[d_idx] += tmp * DWIJ[1]
        d_aw_fluid[d_idx] += tmp * DWIJ[2]


class AccelerationOnStructureDueToFluidRogersConnor(Equation):
    def initialize(self, d_idx, d_au_fluid, d_av_fluid, d_aw_fluid):
        d_au_fluid[d_idx] = 0.
        d_av_fluid[d_idx] = 0.
        d_aw_fluid[d_idx] = 0.

    def loop(self, d_rho_fsi, s_rho, d_idx, d_m, d_m_fsi, s_idx, d_p_fsi, s_p,
             s_m, d_au, d_av, d_aw, d_au_fluid, d_av_fluid, d_aw_fluid, DWIJ):
        rhoi = d_rho_fsi[d_idx]
        rhoj = s_rho[s_idx]

        pij = (d_p_fsi[d_idx] + s_p[s_idx]) / (rhoi * rhoj)

        tmp = -s_m[s_idx] * pij * d_m_fsi[d_idx] / d_m[d_idx]

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]

        d_au_fluid[d_idx] += tmp * DWIJ[0]
        d_av_fluid[d_idx] += tmp * DWIJ[1]
        d_aw_fluid[d_idx] += tmp * DWIJ[2]


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


class EDACGTVFStep(IntegratorStep):
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
               d_arho, d_m, d_p, d_ap, dt):
        d_rho[d_idx] += dt * d_arho[d_idx]
        d_p[d_idx] += dt * d_ap[d_idx]

        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt, d_uhat,
               d_vhat, d_what, d_auhat, d_avhat, d_awhat):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]


class FluidStage1(Equation):
    def __init__(self, dest, sources, dt):
        self.dt = dt
        super(FluidStage1, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_uhat,
                   d_vhat, d_what, d_auhat, d_avhat, d_awhat):
        dtb2 = 0.5 * self.dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]


class FluidStage2(Equation):
    def __init__(self, dest, sources, dt):
        self.dt = dt

        super(FluidStage2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_uhat, d_vhat, d_what, d_x, d_y, d_z, d_rho,
                   d_arho, d_m, d_p, d_ap):
        dt = self.dt
        d_rho[d_idx] += dt * d_arho[d_idx]
        d_p[d_idx] += dt * d_ap[d_idx]

        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]


class FluidStage3(Equation):
    def __init__(self, dest, sources, dt):
        self.dt = dt

        super(FluidStage3, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_uhat,
                   d_vhat, d_what, d_auhat, d_avhat, d_awhat):
        dtb2 = 0.5 * self.dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]


class SolidsStage1(Equation):
    def __init__(self, dest, sources, dt):
        self.dt = dt
        super(SolidsStage1, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_uhat, d_vhat,
                   d_what, d_auhat, d_avhat, d_awhat):
        dtb2 = 0.5 * self.dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]


class SolidsStage2(Equation):
    def __init__(self, dest, sources, dt):
        self.dt = dt

        super(SolidsStage2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_uhat, d_vhat, d_what, d_x, d_y, d_z,
                   d_rho, d_arho, d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                   d_as00, d_as01, d_as02, d_as11, d_as12, d_as22, d_sigma00,
                   d_sigma01, d_sigma02, d_sigma11, d_sigma12, d_sigma22, d_p,
                   d_ap):
        dt = self.dt
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


class SolidsStage3(Equation):
    def __init__(self, dest, sources, dt):
        self.dt = dt

        super(SolidsStage3, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_uhat,
                   d_vhat, d_what, d_auhat, d_avhat, d_awhat):
        dtb2 = 0.5 * self.dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]


class SubSteppingIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.compute_accelerations()


class FSIETVFScheme(Scheme):
    def __init__(self, fluids, structures, solids, structure_solids, dim,
                 h_fluid, c0_fluid, nu_fluid, rho0_fluid, mach_no_fluid,
                 mach_no_structure, dt_fluid=1., dt_solid=1., pb_fluid=0.0,
                 gx=0.0, gy=0.0, gz=0.0, alpha_solid=1.0,
                 beta_solid=0.0, alpha_fluid=0.0, edac_alpha=0.5,
                 pst="sun2019", edac=False):
        """Parameters
        ----------

        fluids: list
            List of names of fluid particle arrays.
        solids: list
            List of names of solid particle arrays (or boundaries).
        stuctures: list
            List of names of solid particle arrays (or boundaries).
        structure_solids: list
            List of names of solid particle arrays (or boundaries).
        dim: int
            Dimensionality of the problem.
        h_fluid: float
            Reference smoothing length of fluid medium.
        c0_fluid: float
            Reference speed of sound of fluid medium.
        nu_fluid: float
            Real viscosity of the fluid, defaults to no viscosity.
        rho_fluid: float
            Reference density of fluid medium.
        gx, gy, gz: float
            Body force acceleration components.
        alpha_solid: float
            Coefficient for artificial viscosity for solid.
        beta_solid: float
            Coefficient for artificial viscosity for solid.
        edac: bool
            Use edac equation for fluid
        damping: bool
            Use damping for the elastic structure part
        damping_coeff: float
            The damping coefficient for the elastic structure
        """
        self.fluids = fluids
        self.solids = solids
        self.structures = structures
        if structure_solids is None:
            self.structure_solids = []
        else:
            self.structure_solids = structure_solids
        self.dim = dim
        self.h_fluid = h_fluid
        self.c0_fluid = c0_fluid
        self.nu_fluid = nu_fluid
        self.rho0_fluid = rho0_fluid
        self.mach_no_fluid = mach_no_fluid
        self.mach_no_structure = mach_no_structure
        self.artificial_stress_eps = 0.3
        self.dt_fluid = dt_fluid
        self.dt_solid = dt_solid
        self.pb_fluid = pb_fluid
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.alpha_solid = alpha_solid
        self.beta_solid = beta_solid
        self.alpha_fluid = alpha_fluid
        self.edac_alpha = edac_alpha
        self.pst = pst
        self.edac = edac
        self.wall_pst = True
        self.damping = False
        self.damping_coeff = 0.002

        # common properties
        self.solver = None
        self.kernel_factor = 2

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--alpha-fluid", action="store", type=float,
                           dest="alpha_fluid",
                           default=0.1,
                           help="Alpha for the artificial viscosity in fluid.")

        group.add_argument("--alpha-solid", action="store", type=float,
                           dest="alpha_solid",
                           default=1,
                           help="Alpha for the artificial viscosity in solid.")

        group.add_argument("--beta-solid", action="store", type=float,
                           dest="beta_solid",
                           default=0.0,
                           help="Beta for the artificial viscosity in solid.")

        group.add_argument("--edac-alpha", action="store", type=float,
                           dest="edac_alpha", default=None,
                           help="Alpha for the EDAC scheme viscosity.")

        add_bool_argument(group, 'wall-pst', dest='wall_pst',
                          default=True, help='Add wall as PST source')

        add_bool_argument(group, 'damping',
                          dest='damping',
                          default=False,
                          help='Use damping')

        group.add_argument("--damping-coeff", action="store",
                           dest="damping_coeff", default=0.000, type=float,
                           help="Damping coefficient for Bui")

        choices = ['sun2019', 'gtvf']
        group.add_argument(
            "--pst", action="store", dest='pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use (one of %s)." % choices)

    def consume_user_options(self, options):
        vars = ['alpha_fluid', 'alpha_solid', 'beta_solid',
                'edac_alpha', 'pst', 'wall_pst', 'damping', 'damping_coeff']
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def attributes_changed(self):
        if self.pb_fluid is not None:
            self.use_tvf = abs(self.pb_fluid) > 1e-14
        if self.h_fluid is not None and self.c0_fluid is not None:
            self.art_nu = self.edac_alpha * self.h_fluid * self.c0_fluid / 8

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.wc.gtvf import GTVFIntegrator
        from solid_mech import (GTVFSolidMechStepEDAC, SolidMechStep)
        kernel = QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        # fluid stepper
        step_cls = EDACGTVFStep
        cls = (integrator_cls
               if integrator_cls is not None else GTVFIntegrator)

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()
        integrator = cls(**steppers)

        # structure stepper
        if self.edac is True:
            step_cls = GTVFSolidMechStepEDAC
        else:
            step_cls = SolidMechStep

        for name in self.structures:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        # from pysph.sph.wc.gtvf import (MomentumEquationArtificialStress)
        from fluids import (FluidSetWallVelocityUFreeSlipAndNoSlip,
                            FluidSetWallVelocityUhatFreeSlipAndNoSlip,
                            FluidContinuityEquationGTVF,
                            FluidContinuityEquationETVFCorrection,
                            FluidContinuityEquationGTVFSolid,
                            FluidContinuityEquationETVFCorrectionSolid,
                            FluidEDACEquation,
                            FluidEDACEquationSolid,
                            FluidMomentumEquationPressureGradient,
                            FluidMomentumEquationPressureGradientRogersConnor,
                            FluidMomentumEquationTVFDivergence)

        from solid_mech import (AdamiBoundaryConditionExtrapolateNoSlip,
                                SetHIJForInsideParticles,
                                ElasticSolidMomentumEquation,
                                ElasticSolidComputeAuHatETVFSun2019,
                                AddGravityToStructure,
                                BuiFukagawaDampingGraularSPH)

        from solid_mech import (ElasticSolidContinuityEquationUhat,
                                ElasticSolidContinuityEquationETVFCorrection,
                                ElasticSolidMomentumEquation,
                                ElasticSolidComputeAuHatETVFSun2019)

        from pysph.sph.wc.gtvf import (ContinuityEquationGTVF,
                                       MomentumEquationArtificialStress)

        from pysph.sph.solid_mech.basic import (
            IsothermalEOS, MomentumEquationWithStress,
            HookesDeviatoricStressRate, MonaghanArtificialStress)

        nu_edac = self._get_edac_nu()
        all = self.fluids + self.solids
        stage1 = []

        # =========================#
        # fluid equations
        # =========================#
        if len(self.solids) > 0:
            eqs_u = []
            eqs_uhat = []

            for solid in self.solids:
                eqs_u.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=solid,
                                                           sources=self.fluids), )

                eqs_uhat.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(
                        dest=solid, sources=self.fluids))

            stage1.append(Group(equations=eqs_u, real=False))
            stage1.append(Group(equations=eqs_uhat, real=False))

        if len(self.structure_solids) > 0:
            eqs_u = []
            eqs_uhat = []

            for solid in self.structure_solids:
                eqs_u.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=solid,
                                                           sources=self.fluids), )

                eqs_uhat.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(
                        dest=solid, sources=self.fluids))

            stage1.append(Group(equations=eqs_u, real=False))
            stage1.append(Group(equations=eqs_uhat, real=False))

        eqs = []
        for fluid in self.fluids:
            eqs.append(FluidContinuityEquationGTVF(
                dest=fluid,
                sources=self.fluids))
            eqs.append(FluidContinuityEquationETVFCorrection(
                dest=fluid,
                sources=self.fluids))

            eqs.append(FluidEDACEquation(
                dest=fluid,
                sources=self.fluids,
                nu=nu_edac
            ))

            if len(self.solids) > 0:
                eqs.append(FluidContinuityEquationGTVFSolid(
                    dest=fluid,
                    sources=self.solids))
                eqs.append(FluidContinuityEquationETVFCorrectionSolid(
                    dest=fluid,
                    sources=self.solids))

                eqs.append(FluidEDACEquationSolid(
                    dest=fluid,
                    sources=self.solids,
                    nu=nu_edac
                ))

            if len(self.structures) > 0:
                eqs.append(FluidContinuityEquationGTVFFSI(
                    dest=fluid,
                    sources=self.structures))

                eqs.append(FluidContinuityEquationETVFCorrectionFSI(
                    dest=fluid,
                    sources=self.structures))

                eqs.append(FluidEDACEquationFSI(
                    dest=fluid,
                    sources=self.structures,
                    nu=nu_edac
                ))

            if len(self.structure_solids) > 0:
                eqs.append(FluidContinuityEquationGTVFFSISolid(
                    dest=fluid,
                    sources=self.structure_solids))

                eqs.append(FluidContinuityEquationETVFCorrectionFSISolid(
                    dest=fluid,
                    sources=self.structure_solids))

                eqs.append(FluidEDACEquationFSISolid(
                    dest=fluid,
                    sources=self.structure_solids,
                    nu=nu_edac
                ))

        stage1.append(Group(equations=eqs, real=False))
        # =========================#
        # fluid equations ends
        # =========================#

        # =========================#
        # structure equations
        # =========================#
        all = self.structures + self.structure_solids
        g1 = []

        if len(self.structures) > 0.:
            for structure in self.structures:
                g1.append(ElasticSolidContinuityEquationUhat(dest=structure,
                                                             sources=all))
                g1.append(
                    ElasticSolidContinuityEquationETVFCorrection(
                        dest=structure, sources=all))
                g1.append(VelocityGradient2D(dest=structure, sources=all))

            stage1.append(Group(equations=g1))

            g2 = []
            for structure in self.structures:
                g2.append(HookesDeviatoricStressRate(dest=structure,
                                                     sources=None))

            stage1.append(Group(equations=g2))
        # =========================#
        # structure equations ends
        # =========================#

        # =========================#
        # stage 2 equations start
        # =========================#
        stage2 = []
        if len(self.solids) > 0:
            eqs = []
            for solid in self.solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=solid,
                                                           sources=self.fluids))
                eqs.append(
                    SolidWallPressureBC(dest=solid, sources=self.fluids,
                                        gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    ClampWallPressureWall(dest=solid, sources=None,
                                          p_0=self.pb_fluid,
                                          rho_0=self.rho0_fluid))

            stage2.append(Group(equations=eqs, real=False))

        # FSI coupling equations, set the pressure
        if len(self.structure_solids) > 0:
            eqs = []
            for structure in self.structure_solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=structure,
                                                           sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCFSI(dest=structure,
                                                sources=self.fluids,
                                                p_0=self.pb_fluid,
                                                rho_0=self.rho0_fluid,
                                                gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    ClampWallPressureFSI(dest=structure, sources=None,
                                         p_0=self.pb_fluid,
                                         rho_0=self.rho0_fluid))

            stage2.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        # FSI coupling equations, set the pressure
        if len(self.structures) > 0:
            eqs = []
            for structure in self.structures:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=structure,
                                                           sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCFSI(dest=structure,
                                                sources=self.fluids,
                                                p_0=self.pb_fluid,
                                                rho_0=self.rho0_fluid,
                                                gx=self.gx, gy=self.gy,
                                                gz=self.gz))

            stage2.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        # fluid momentum equations
        eqs = []
        for fluid in self.fluids:
            eqs.append(
                SetHIJForInsideParticles(dest=fluid, sources=[fluid],
                                         kernel_factor=self.kernel_factor))
        stage2.append(Group(eqs))

        eqs = []
        for fluid in self.fluids:
            # if self.alpha_fluid > 0.:
            #     if self.visc_to_solids is True:
            #         eqs.append(
            #             MomentumEquationArtificialViscosity(
            #                 dest=fluid,
            #                 sources=self.fluids+self.solids+self.structures
            #                 + self.structure_solids,
            #                 c0=self.c0_fluid,
            #                 alpha=self.alpha_fluid
            #             )
            #         )
            #     else:
            #         eqs.append(
            #             MomentumEquationArtificialViscosity(
            #                 dest=fluid,
            #                 sources=self.fluids,
            #                 c0=self.c0_fluid,
            #                 alpha=self.alpha_fluid
            #             )
            #         )

            # if self.nu_fluid > 0.:
            #     eqs.append(
            #         MomentumEquationViscosity(
            #             dest=fluid, sources=self.fluids, nu=self.nu_fluid
            #         )

            #     )

            eqs.append(
                FluidMomentumEquationPressureGradient(
                    dest=fluid, sources=self.fluids + self.solids,
                    gx=self.gx, gy=self.gy, gz=self.gz), )

            eqs.append(
                MomentumEquationArtificialStress(dest=fluid,
                                                 sources=self.fluids,
                                                 dim=self.dim))
            eqs.append(
              FluidMomentumEquationTVFDivergence(dest=fluid, sources=self.fluids))

            eqs.append(
                FluidComputeAuHatGTVF(dest=fluid,
                                      sources=self.fluids + self.solids
                                      + self.structures + self.structure_solids))

            if len(self.structure_solids + self.structures) > 0.:
                eqs.append(
                    AccelerationOnFluidDueToStructure(
                        dest=fluid,
                        sources=self.structures + self.structure_solids),
                )

        stage2.append(Group(equations=eqs, real=True))
        # ============================================
        # fluid momentum equations ends
        # ============================================

        # ============================================
        # structures momentum equations
        # ============================================
        g1 = []
        g2 = []
        g3 = []
        g4 = []

        if len(self.structures) > 0.:
            for structure in self.structures:
                g1.append(
                    SetHIJForInsideParticles(dest=structure, sources=[structure],
                                             kernel_factor=self.kernel_factor))
            stage2.append(Group(g1))

        if len(self.structures) > 0.:
            for structure in self.structures:
                g2.append(IsothermalEOS(structure, sources=None))
            stage2.append(Group(g2))

        # -------------------
        # boundary conditions
        # -------------------
        if len(self.structure_solids) > 0:
            for boundary in self.structure_solids:
                g3.append(
                    AdamiBoundaryConditionExtrapolateNoSlip(
                        dest=boundary, sources=self.structures, gx=self.gx,
                        gy=self.gy, gz=self.gz))
            stage2.append(Group(g3))

        # -------------------
        # solve momentum equation for solid
        # -------------------
        if len(self.structures) > 0.:
            g4 = []
            for structure in self.structures:
                # add only if there is some positive value
                if self.alpha_solid > 0. or self.beta_solid > 0.:
                    g4.append(
                        MonaghanArtificialViscosity(
                            dest=structure,
                            sources=self.structures + self.structure_solids,
                            alpha=self.alpha_solid,
                            beta=self.beta_solid))

                g4.append(
                    ElasticSolidMomentumEquation(
                        dest=structure,
                        sources=self.structures + self.structure_solids))

                g4.append(
                    ElasticSolidComputeAuHatETVFSun2019(
                        dest=structure,
                        sources=[structure] + self.structure_solids,
                        mach_no=self.mach_no_structure))

                g4.append(
                    AccelerationOnStructureDueToFluid(dest=structure,
                                                        sources=self.fluids), )

            stage2.append(Group(g4))

            g5 = []
            for structure in self.structures:
                g5.append(
                    AddGravityToStructure(dest=structure, sources=None,
                                          gx=self.gx, gy=self.gy,
                                          gz=self.gz))

                if self.damping == True:
                    g5.append(
                        BuiFukagawaDampingGraularSPH(
                            dest=structure, sources=None,
                            damping_coeff=self.damping_coeff))

                stage2.append(Group(g5))

        return MultiStageEquations([stage1, stage2])

    def setup_properties(self, particles, clean=True):
        pas = dict([(p.name, p) for p in particles])
        for fluid in self.fluids:
            pa = pas[fluid]

            add_properties(pa, 'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'rho0',
                           'arho', 'ap', 'arho', 'p0', 'uhat', 'vhat', 'what',
                           'auhat', 'avhat', 'awhat', 'h_b', 'V', 'div_r', 'cs')

            pa.h_b[:] = pa.h[:]

            if 'c0_ref' not in pa.constants:
                pa.add_constant('c0_ref', self.c0_fluid)
                pa.add_constant('p_ref', self.rho0_fluid * self.c0_fluid**2.)
            pa.cs[:] = self.c0_fluid
            pa.add_output_arrays(['p'])

            if 'wdeltap' not in pa.constants:
                kernel = QuinticSpline(dim=self.dim)
                wdeltap = kernel.kernel(rij=pa.h[0], h=pa.h[0])
                pa.add_constant('wdeltap', wdeltap)

            if 'n' not in pa.constants:
                pa.add_constant('n', 4.)

            add_boundary_identification_properties(pa)

            pa.h_b[:] = pa.h

        for solid in self.solids:
            pa = pas[solid]

            add_properties(pa, 'rho', 'V', 'wij2', 'wij', 'uhat', 'vhat',
                           'what', 'ug', 'vf', 'vg', 'wg', 'uf', 'wf', 'ugfs',
                           'vgfs', 'wgfs', 'ughatns', 'vghatns', 'wghatns',
                           'ughatfs', 'vghatfs', 'wghatfs',
                           'ugns', 'vgns', 'wgns')

            # for normals
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

        # Add fsi props
        for structure in self.structures:
            pa = pas[structure]

            add_properties(pa, 'm_fsi', 'p_fsi', 'rho_fsi', 'V', 'wij2', 'wij',
                           'uhat', 'vhat', 'what', 'ap')
            add_properties(pa, 'div_r')
            # add_properties(pa, 'p_fsi', 'wij', 'm_fsi', 'rho_fsi')

            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf')
            add_properties(pa, 'ugfs', 'vgfs', 'wgfs')

            add_properties(pa, 'ughatns', 'vghatns', 'wghatns')
            add_properties(pa, 'ughatfs', 'vghatfs', 'wghatfs')

            # No slip boundary conditions for viscosity force
            add_properties(pa, 'ugns', 'vgns', 'wgns')

            # pa.h_b[:] = pa.h[:]
            # force on structure due to fluid
            add_properties(pa, 'au_fluid', 'av_fluid', 'aw_fluid')

            # for normals
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

        for solid in self.structure_solids:
            pa = pas[solid]

            add_properties(pa, 'm_fsi', 'p_fsi', 'rho_fsi', 'rho', 'V', 'wij2',
                           'wij', 'uhat', 'vhat', 'what')
            # add_properties(pa, 'p_fsi', 'wij', 'm_fsi', 'rho_fsi')

            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf')
            add_properties(pa, 'ugfs', 'vgfs', 'wgfs')

            add_properties(pa, 'ughatns', 'vghatns', 'wghatns')
            add_properties(pa, 'ughatfs', 'vghatfs', 'wghatfs')

            # No slip boundary conditions for viscosity force
            add_properties(pa, 'ugns', 'vgns', 'wgns')

            # force on structure due to fluid
            add_properties(pa, 'au_fluid', 'av_fluid', 'aw_fluid')
            # pa.h_b[:] = pa.h[:]

            # for normals
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

        # add the elastic dynamics properties
        for structure in self.structures:
            # we expect the solid to have Young's modulus, Poisson ration as
            # given
            pa = pas[structure]

            # add the properties that are used by all the schemes
            add_properties(pa, 'cs', 'v00', 'v01', 'v02', 'v10', 'v11', 'v12',
                           'v20', 'v21', 'v22', 's00', 's01', 's02', 's11',
                           's12', 's22', 'as00', 'as01', 'as02', 'as11',
                           'as12', 'as22', 'arho', 'au', 'av', 'aw')

            # this will change
            kernel = QuinticSpline(dim=self.dim)
            wdeltap = kernel.kernel(rij=pa.h[0], h=pa.h[0])
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

            # for boundary identification and for sun2019 pst
            pa.add_property('normal', stride=3)
            pa.add_property('normal_tmp', stride=3)
            pa.add_property('normal_norm')

            # check for boundary particle
            pa.add_property('is_boundary', type='int')

            # used to set the particles near the boundary
            pa.add_property('h_b')

            # for edac
            if self.edac is True:
                add_properties(pa, 'ap')

            # update the h if using wendlandquinticc4
            pa.add_output_arrays(['p'])

        for boundary in self.structure_solids:
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

            pa.add_property('ughat')
            pa.add_property('vghat')
            pa.add_property('wghat')

        # Properties for new equations, setting mass and other fractions
        for name in self.fluids+self.structures+self.solids+self.structure_solids:
            pa = pas[name]
            add_properties(pa, 'p_frac', 'rho_frac', 'm_frac')
            pa.p_frac[:] = 1.
            pa.rho_frac[:] = 1.
            pa.m_frac[:] = 1.

    def get_solver(self):
        return self.solver

    def _get_edac_nu(self):
        if self.art_nu > 0:
            nu = self.art_nu
            print(self.art_nu)
            print("Using artificial viscosity for EDAC with nu = %s" % nu)
        else:
            nu = self.nu
            print("Using real viscosity for EDAC with nu = %s" % self.nu)
        return nu


class FSIETVFSubSteppingScheme(FSIETVFScheme):
    def attributes_changed(self):
        super().attributes_changed()

        self.dt_factor = int(self.dt_fluid / self.dt_solid) + 1
        self.dt_fluid_simulated = self.dt_factor * self.dt_solid

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.wc.gtvf import GTVFIntegrator
        from solid_mech import (GTVFSolidMechStepEDAC, SolidMechStep)
        kernel = QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        # fluid stepper
        step_cls = EDACGTVFStep
        cls = (integrator_cls
               if integrator_cls is not None else SubSteppingIntegrator)

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()
        integrator = cls(**steppers)

        # structure stepper
        if self.edac is True:
            step_cls = GTVFSolidMechStepEDAC
        else:
            step_cls = SolidMechStep

        for name in self.structures:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        from fluids import (FluidSetWallVelocityUFreeSlipAndNoSlip,
                            FluidSetWallVelocityUhatFreeSlipAndNoSlip,
                            FluidContinuityEquationGTVF,
                            FluidContinuityEquationETVFCorrection,
                            FluidContinuityEquationGTVFSolid,
                            FluidContinuityEquationETVFCorrectionSolid,
                            FluidEDACEquation,
                            FluidEDACEquationSolid,
                            FluidMomentumEquationPressureGradient,
                            FluidMomentumEquationPressureGradientRogersConnor,
                            FluidMomentumEquationTVFDivergence)

        from solid_mech import (AdamiBoundaryConditionExtrapolateNoSlip,
                                SetHIJForInsideParticles,
                                ElasticSolidMomentumEquation,
                                ElasticSolidComputeAuHatETVFSun2019,
                                AddGravityToStructure
                                )

        from solid_mech import (ElasticSolidContinuityEquationUhat,
                                ElasticSolidContinuityEquationETVFCorrection,
                                ElasticSolidMomentumEquation,
                                ElasticSolidComputeAuHatETVFSun2019)

        from pysph.sph.wc.gtvf import (ContinuityEquationGTVF,
                                       MomentumEquationArtificialStress)

        nu_edac = self._get_edac_nu()
        all = self.fluids + self.solids
        stage1 = []

        eqs = []
        for fluid in self.fluids:
            eqs.append(FluidStage1(dest=fluid,
                                   sources=None, dt=self.dt_fluid_simulated), )

        stage1.append(Group(equations=eqs, real=False))

        # =========================#
        # fluid equations
        # =========================#
        if len(self.solids) > 0:
            eqs_u = []
            eqs_uhat = []

            for solid in self.solids:
                eqs_u.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=solid,
                                                           sources=self.fluids), )

                eqs_uhat.append(
                    FluidSetWallVelocityUhatFreeSlipAndNoSlip(
                        dest=solid, sources=self.fluids))

            stage1.append(Group(equations=eqs_u, real=False))
            stage1.append(Group(equations=eqs_uhat, real=False))

        if len(self.structure_solids) > 0:
            eqs_u = []
            eqs_uhat = []

            for solid in self.structure_solids:
                eqs_u.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=solid,
                                                           sources=self.fluids), )

                eqs_uhat.append(
                    FluidSetWallVelocityUhatFreeSlipAndNoSlip(
                        dest=solid, sources=self.fluids))

            stage1.append(Group(equations=eqs_u, real=False))
            stage1.append(Group(equations=eqs_uhat, real=False))

        eqs = []
        for fluid in self.fluids:
            eqs.append(FluidContinuityEquationGTVF(
                dest=fluid,
                sources=self.fluids))
            eqs.append(FluidContinuityEquationETVFCorrection(
                dest=fluid,
                sources=self.fluids))

            eqs.append(FluidEDACEquation(
                dest=fluid,
                sources=self.fluids,
                nu=nu_edac
            ))

            if len(self.solids) > 0:
                eqs.append(FluidContinuityEquationGTVFSolid(
                    dest=fluid,
                    sources=self.solids))
                eqs.append(FluidContinuityEquationETVFCorrectionSolid(
                    dest=fluid,
                    sources=self.solids))

                eqs.append(FluidEDACEquationSolid(
                    dest=fluid,
                    sources=self.solids,
                    nu=nu_edac
                ))

            if len(self.structures) > 0:
                eqs.append(FluidContinuityEquationGTVFFSI(
                    dest=fluid,
                    sources=self.structures))

                eqs.append(FluidContinuityEquationETVFCorrectionFSI(
                    dest=fluid,
                    sources=self.structures))

                eqs.append(FluidEDACEquationFSI(
                    dest=fluid,
                    sources=self.structures,
                    nu=nu_edac
                ))

            if len(self.structure_solids) > 0:
                eqs.append(FluidContinuityEquationGTVFFSISolid(
                    dest=fluid,
                    sources=self.structure_solids))

                eqs.append(FluidContinuityEquationETVFCorrectionFSISolid(
                    dest=fluid,
                    sources=self.structure_solids))

                eqs.append(FluidEDACEquationFSISolid(
                    dest=fluid,
                    sources=self.structure_solids,
                    nu=nu_edac
                ))

        stage1.append(Group(equations=eqs, real=False))
        # =========================#
        # fluid equations ends
        # =========================#

        eqs = []
        for fluid in self.fluids:
            eqs.append(FluidStage2(dest=fluid, sources=None,
                                   dt=self.dt_fluid_simulated))

        stage1.append(Group(equations=eqs, real=False))

        if len(self.solids) > 0:
            eqs = []
            for solid in self.solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=solid,
                                                           sources=self.fluids))
                eqs.append(
                    SolidWallPressureBC(dest=solid, sources=self.fluids,
                                        gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    ClampWallPressureWall(dest=solid, sources=None,
                                          p_0=self.pb_fluid,
                                          rho_0=self.rho0_fluid))

            stage1.append(Group(equations=eqs, real=False))

        # FSI coupling equations, set the pressure
        if len(self.structure_solids) > 0:
            eqs = []
            for structure in self.structure_solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=structure,
                                                           sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCFSI(dest=structure,
                                                sources=self.fluids,
                                                p_0=self.pb_fluid,
                                                rho_0=self.rho0_fluid,
                                                gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    ClampWallPressureFSI(dest=structure, sources=None,
                                         p_0=self.pb_fluid,
                                         rho_0=self.rho0_fluid))

            stage1.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        # FSI coupling equations, set the pressure
        if len(self.structures) > 0:
            eqs = []
            for structure in self.structures:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=structure,
                                                           sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCFSI(dest=structure,
                                                sources=self.fluids,
                                                p_0=self.pb_fluid,
                                                rho_0=self.rho0_fluid,
                                                gx=self.gx, gy=self.gy,
                                                gz=self.gz))

            stage1.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        # fluid momentum equations
        eqs = []
        for fluid in self.fluids:
            eqs.append(
                SetHIJForInsideParticles(dest=fluid, sources=[fluid],
                                         kernel_factor=self.kernel_factor))
        stage1.append(Group(eqs))

        eqs = []
        for fluid in self.fluids:
            # if self.alpha_fluid > 0.:
            #     if self.visc_to_solids is True:
            #         eqs.append(
            #             MomentumEquationArtificialViscosity(
            #                 dest=fluid,
            #                 sources=self.fluids+self.solids+self.structures
            #                 + self.structure_solids,
            #                 c0=self.c0_fluid,
            #                 alpha=self.alpha_fluid
            #             )
            #         )
            #     else:
            #         eqs.append(
            #             MomentumEquationArtificialViscosity(
            #                 dest=fluid,
            #                 sources=self.fluids,
            #                 c0=self.c0_fluid,
            #                 alpha=self.alpha_fluid
            #             )
            #         )

            # if self.nu_fluid > 0.:
            #     eqs.append(
            #         MomentumEquationViscosity(
            #             dest=fluid, sources=self.fluids, nu=self.nu_fluid
            #         )
            #     )

            eqs.append(
                FluidMomentumEquationPressureGradient(
                    dest=fluid, sources=self.fluids + self.solids,
                    gx=self.gx, gy=self.gy, gz=self.gz), )

            eqs.append(
                MomentumEquationArtificialStress(dest=fluid,
                                                 sources=self.fluids,
                                                 dim=self.dim))
            eqs.append(
                FluidMomentumEquationTVFDivergence(dest=fluid, sources=self.fluids))

            eqs.append(
                FluidComputeAuHatGTVF(dest=fluid,
                                      sources=self.fluids + self.solids
                                      + self.structures + self.structure_solids))

            if len(self.structure_solids + self.structures) > 0.:
                eqs.append(
                    AccelerationOnFluidDueToStructure(
                        dest=fluid,
                        sources=self.structures + self.structure_solids),
                )

                # if self.pressure_integration_fsi is True:
                #     eqs.append(
                #         AccelerationOnFluidDueToStructurePressureIntegration(
                #             dest=fluid,
                #             sources=self.structures + self.structure_solids))

        stage1.append(Group(equations=eqs, real=True))
        # ============================================
        # fluid momentum equations ends
        # ============================================

        eqs = []
        for fluid in self.fluids:
            eqs.append(FluidStage3(dest=fluid, sources=None,
                                   dt=self.dt_fluid_simulated))

        stage1.append(Group(equations=eqs, real=False))

        # =========================#
        # fluid equations ends
        # =========================#

        # =========================#
        # structure equations
        # =========================#
        stage1_stucture_eqs = []
        all = self.structures + self.structure_solids
        g1 = []

        if len(self.structures) > 0.:
            for structure in self.structures:
                g1.append(SolidsStage1(dest=structure, sources=None,
                                       dt=self.dt_solid))
            stage1_stucture_eqs.append(Group(equations=g1))

        g1 = []
        if len(self.structures) > 0.:
            for structure in self.structures:
                g1.append(ElasticSolidContinuityEquationUhat(dest=structure,
                                                                sources=all))
                g1.append(
                    ElasticSolidContinuityEquationETVFCorrection(
                        dest=structure, sources=all))
                g1.append(VelocityGradient2D(dest=structure, sources=all))

            stage1_stucture_eqs.append(Group(equations=g1))

            g2 = []
            for structure in self.structures:
                g2.append(HookesDeviatoricStressRate(dest=structure,
                                                     sources=None))

            stage1_stucture_eqs.append(Group(equations=g2))

        g1 = []
        if len(self.structures) > 0.:
            for structure in self.structures:
                g1.append(SolidsStage2(dest=structure, sources=None,
                                       dt=self.dt_solid))
            stage1_stucture_eqs.append(Group(equations=g1))
        # =========================#
        # structure equations ends
        # =========================#

        # ============================================
        # structures momentum equations
        # ============================================
        g1 = []
        g2 = []
        g3 = []
        g4 = []

        if len(self.structures) > 0.:
            for structure in self.structures:
                g1.append(
                    SetHIJForInsideParticles(dest=structure, sources=[structure],
                                             kernel_factor=self.kernel_factor))
            stage1_stucture_eqs.append(Group(g1))

        # if self.edac is False:
        if len(self.structures) > 0.:
            for structure in self.structures:
                g2.append(IsothermalEOS(structure, sources=None))
            stage1_stucture_eqs.append(Group(g2))

        # -------------------
        # boundary conditions
        # -------------------
        if len(self.structure_solids) > 0:
            for boundary in self.structure_solids:
                g3.append(
                    AdamiBoundaryConditionExtrapolateNoSlip(
                        dest=boundary, sources=self.structures, gx=self.gx,
                        gy=self.gy, gz=self.gz))
            stage1_stucture_eqs.append(Group(g3))

        # -------------------
        # solve momentum equation for solid
        # -------------------
        if len(self.structures) > 0.:
            g4 = []
            for structure in self.structures:
                # add only if there is some positive value
                if self.alpha_solid > 0. or self.beta_solid > 0.:
                    g4.append(
                        MonaghanArtificialViscosity(
                            dest=structure,
                            sources=self.structures + self.structure_solids,
                            alpha=self.alpha_solid,
                            beta=self.beta_solid))

                g4.append(
                    ElasticSolidMomentumEquation(
                        dest=structure,
                        sources=self.structures + self.structure_solids))

                if self.pst == "sun2019":
                    if self.wall_pst is True:
                        g4.append(
                            ElasticSolidComputeAuHatETVFSun2019(
                                dest=structure,
                                sources=[structure] + self.structure_solids,
                                mach_no=self.mach_no_structure))
                    else:
                        g4.append(
                            ElasticSolidComputeAuHatETVFSun2019(
                                dest=structure,
                                sources=[structure],
                                mach_no=self.mach_no_structure))

                g4.append(
                    AccelerationOnStructureDueToFluid(dest=structure,
                                                      sources=self.fluids), )

            stage1_stucture_eqs.append(Group(g4))

            # Add gravity
            g5 = []
            for structure in self.structures:
                g5.append(
                    AddGravityToStructure(dest=structure, sources=None,
                                          gx=self.gx, gy=self.gy,
                                          gz=self.gz))

            stage1_stucture_eqs.append(Group(g5))

        g1 = []
        if len(self.structures) > 0.:
            for structure in self.structures:
                g1.append(SolidsStage3(dest=structure, sources=None,
                                       dt=self.dt_solid))
            stage1_stucture_eqs.append(Group(equations=g1))

        stage1.append(Group(equations=stage1_stucture_eqs,
                            iterate=True, max_iterations=self.dt_factor,
                            min_iterations=self.dt_factor))

        return stage1
