"""
Run it by

python lid_driven_cavity.py --openmp --scheme etvf --integrator pec --internal-flow --pst sun2019 --re 100 --tf 25 --nx 50 --no-edac -d lid_driven_cavity_scheme_etvf_integrator_pec_pst_sun2019_re_100_nx_50_no_edac_output --detailed-output --pfreq 100

"""
import numpy
import numpy as np

from pysph.sph.integrator import Integrator
from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)

from boundary_particles import (add_boundary_identification_properties)

from boundary_particles import (ComputeNormals, SmoothNormals,
                                IdentifyBoundaryParticleCosAngleEDAC)

from pysph.examples.solid_mech.impact import add_properties


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


class FluidContinuityEquationWCSPHOnStructure(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m_fsi, d_rho, s_rho_fsi, d_uhat, d_vhat,
             d_what, s_uhat, s_vhat, s_what, d_arho, DWIJ, VIJ):
        udotdij = DWIJ[0] * VIJ[0] + DWIJ[1] * VIJ[1] + DWIJ[2] * VIJ[2]
        fac = d_rho[d_idx] * s_m_fsi[s_idx] / s_rho_fsi[s_idx]
        d_arho[d_idx] += fac * udotdij


class FluidContinuityEquationWCSPHOnStructureSolid(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m_fsi, d_rho, s_rho_fsi, d_u, d_v, d_w,
             s_ugfs, s_vgfs, s_wgfs, d_arho, DWIJ):
        uhatij = d_u[d_idx] - s_ugfs[s_idx]
        vhatij = d_v[d_idx] - s_vgfs[s_idx]
        whatij = d_w[d_idx] - s_wgfs[s_idx]

        udotdij = DWIJ[0] * uhatij + DWIJ[1] * vhatij + DWIJ[2] * whatij
        fac = d_rho[d_idx] * s_m_fsi[s_idx] / s_rho_fsi[s_idx]
        d_arho[d_idx] += fac * udotdij


class FluidEDACEquationWCSPHOnStructure(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidEDACEquationWCSPHOnStructure, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_cs, d_u, d_v, d_w, s_p_fsi,
             s_m_fsi, s_rho_fsi, d_ap, DWIJ, XIJ, s_u, s_v, s_w, R2IJ, VIJ,
             EPS):
        cs2 = d_cs[d_idx] * d_cs[d_idx]

        rhoj1 = 1.0 / s_rho_fsi[s_idx]
        Vj = s_m_fsi[s_idx] * rhoj1
        rhoi = d_rho[d_idx]

        vij_dot_dwij = (VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] +
                        VIJ[2] * DWIJ[2])

        #######################################################
        # first term on the rhs of Eq 23 of the current paper #
        #######################################################
        d_ap[d_idx] += rhoi * cs2 * Vj * vij_dot_dwij

        #######################################################
        # fourth term on the rhs of Eq 19 of the current paper #
        #######################################################
        rhoij = d_rho[d_idx] + s_rho_fsi[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p_fsi[s_idx])


class FluidEDACEquationWCSPHOnStructureSolid(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidEDACEquationWCSPHOnStructureSolid, self).__init__(
            dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_cs, d_u, d_v, d_w, s_p_fsi,
             s_m_fsi, s_rho_fsi, s_ugfs, s_vgfs, s_wgfs, d_ap, DWIJ, XIJ,
             R2IJ, VIJ, EPS):
        cs2 = d_cs[d_idx] * d_cs[d_idx]

        rhoj1 = 1.0 / s_rho_fsi[s_idx]
        Vj = s_m_fsi[s_idx] * rhoj1
        rhoi = d_rho[d_idx]

        uij = d_u[d_idx] - s_ugfs[s_idx]
        vij = d_v[d_idx] - s_vgfs[s_idx]
        wij = d_w[d_idx] - s_wgfs[s_idx]

        vij_dot_dwij = (uij * DWIJ[0] + vij * DWIJ[1] +
                        wij * DWIJ[2])

        #######################################################
        # first term on the rhs of Eq 23 of the current paper #
        #######################################################
        d_ap[d_idx] += rhoi * cs2 * Vj * vij_dot_dwij

        #######################################################
        # fourth term on the rhs of Eq 19 of the current paper #
        #######################################################
        rhoij = d_rho[d_idx] + s_rho_fsi[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p_fsi[s_idx])


class FluidSolidWallPressureBCStructureSolid(Equation):
    r"""Solid wall pressure boundary condition from Adami and Hu (transport
    velocity formulation).

    """
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(FluidSolidWallPressureBCStructureSolid, self).__init__(
            dest, sources)

    def initialize(self, d_idx, d_p_fsi):
        d_p_fsi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p_fsi, s_p, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p_fsi[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

    def post_loop(self, d_idx, d_wij, d_p_fsi):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p_fsi[d_idx] /= d_wij[d_idx]


class FluidClampWallPressureStructureSolid(Equation):
    def post_loop(self, d_idx, d_p_fsi):
        if d_p_fsi[d_idx] < 0.0:
            d_p_fsi[d_idx] = 0.0


class FluidSolidWallPressureBCStructure(Equation):
    r"""Solid wall pressure boundary condition from Adami and Hu (transport
    velocity formulation).

    """
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(FluidSolidWallPressureBCStructure, self).__init__(
            dest, sources)

    def initialize(self, d_idx, d_p_fsi):
        d_p_fsi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p_fsi, s_p, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p_fsi[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

    def post_loop(self, d_idx, d_wij, d_p_fsi):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p_fsi[d_idx] /= d_wij[d_idx]


class FluidClampWallPressureStructure(Equation):
    def post_loop(self, d_idx, d_p_fsi):
        if d_p_fsi[d_idx] < 0.0:
            d_p_fsi[d_idx] = 0.0


class FluidLaminarViscosityFluid(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidLaminarViscosityFluid, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_au, d_av, d_aw, d_u, d_v,
             d_w, R2IJ, EPS, DWIJ, XIJ, HIJ, VIJ):
        xdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        mb = s_m[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        fac = mb * 4 * self.nu * xdotdwij / ((rhoa + rhob) * (R2IJ + EPS))

        d_au[d_idx] += fac * VIJ[0]
        d_av[d_idx] += fac * VIJ[1]
        d_aw[d_idx] += fac * VIJ[2]


class FluidLaminarViscosityFluidSolid(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidLaminarViscosityFluidSolid, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_au, d_av, d_aw, d_u, d_v,
             d_w, s_ugns, s_vgns, s_wgns, R2IJ, EPS, DWIJ, XIJ, HIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ugns[s_idx]
        vij[1] = d_v[d_idx] - s_vgns[s_idx]
        vij[2] = d_w[d_idx] - s_wgns[s_idx]

        xdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        mb = s_m[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        fac = mb * 4 * self.nu * xdotdwij / ((rhoa + rhob) * (R2IJ + EPS))

        d_au[d_idx] += fac * vij[0]
        d_av[d_idx] += fac * vij[1]
        d_aw[d_idx] += fac * vij[2]


class FluidLaminarViscosityStructure(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidLaminarViscosityStructure, self).__init__(
            dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho_fsi, s_m_fsi,
             d_u, d_v, d_w, s_u, s_v, s_w,
             d_au, d_av, d_aw, R2IJ, EPS, DWIJ, XIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_u[s_idx]
        vij[1] = d_v[d_idx] - s_v[s_idx]
        vij[2] = d_w[d_idx] - s_w[s_idx]

        xdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        mb = s_m_fsi[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho_fsi[s_idx]

        fac = mb * 4 * self.nu * xdotdwij / ((rhoa + rhob) * (R2IJ + EPS))

        d_au[d_idx] += fac * vij[0]
        d_av[d_idx] += fac * vij[1]
        d_aw[d_idx] += fac * vij[2]


class FluidLaminarViscosityStructureSolid(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidLaminarViscosityStructureSolid, self).__init__(
            dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho_fsi, s_m_fsi,
             d_u, d_v, d_w, s_ugns, s_vgns, s_wgns,
             d_au, d_av, d_aw, R2IJ, EPS, DWIJ, XIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ugns[s_idx]
        vij[1] = d_v[d_idx] - s_vgns[s_idx]
        vij[2] = d_w[d_idx] - s_wgns[s_idx]

        xdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        mb = s_m_fsi[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho_fsi[s_idx]

        fac = mb * 4 * self.nu * xdotdwij / ((rhoa + rhob) * (R2IJ + EPS))

        d_au[d_idx] += fac * vij[0]
        d_av[d_idx] += fac * vij[1]
        d_aw[d_idx] += fac * vij[2]


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


class AccelerationOnStructureDueToFluid(Equation):
    def initialize(self, d_idx, d_au_fluid, d_av_fluid, d_aw_fluid):
        d_au_fluid[d_idx] = 0.
        d_av_fluid[d_idx] = 0.
        d_aw_fluid[d_idx] = 0.

    def loop(self, d_rho_fsi, s_rho, d_idx, d_m, d_m_fsi, s_idx, d_p_fsi, s_p,
             s_m, d_au, d_av, d_aw, DWIJ):
        rhoi2 = d_rho_fsi[d_idx] * d_rho_fsi[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p_fsi[d_idx] / rhoi2 + s_p[s_idx] / rhoj2

        tmp = -s_m[s_idx] * pij * d_m_fsi[d_idx] / d_m[d_idx]

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class AccelerationOnStructureDueToFluidViscosity(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(AccelerationOnStructureDueToFluidViscosity, self).__init__(
            dest, sources)

    def loop(self, d_idx, s_idx, d_m, d_m_fsi, d_rho_fsi, s_rho, s_m, d_u, d_v,
             d_w, s_u, s_v, s_w, d_au, d_av, d_aw, R2IJ, EPS, DWIJ, XIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_u[s_idx]
        vij[1] = d_v[d_idx] - s_v[s_idx]
        vij[2] = d_w[d_idx] - s_w[s_idx]

        xdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        mb = s_m[s_idx]

        m_frac = d_m_fsi[d_idx] / d_m[d_idx]

        tmp = mb * m_frac

        rhoa = d_rho_fsi[d_idx]
        rhob = s_rho[s_idx]

        fac = tmp * 4 * self.nu * xdotdwij / ((rhoa + rhob) * (R2IJ + EPS))

        d_au[d_idx] += fac * vij[0]
        d_av[d_idx] += fac * vij[1]
        d_aw[d_idx] += fac * vij[2]


class FSIWCSPHFluidsScheme(Scheme):
    def __init__(self, fluids, structures, solids, structure_solids, dim,
                 h_fluid, c0_fluid, nu_fluid, rho0_fluid, mach_no_fluid,
                 mach_no_structure, structure_u_max=10.,
                 dt_fluid=1., dt_solid=1., pb_fluid=0.0,
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
        self.edac = edac
        self.wall_pst = True
        self.damping = False
        self.damping_coeff = 0.002
        self.solid_velocity_bc = True
        self.fsi_type = None
        self.solid_pst = "sun2019"
        # attributes for IPST technique
        self.ipst_max_iterations = 10
        self.ipst_min_iterations = 5
        self.ipst_tolerance = 0.2
        self.ipst_interval = 1
        self.structure_u_max = structure_u_max

        self.debug = False

        # common properties
        self.solver = None
        self.kernel_factor = 3

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

        add_bool_argument(group, 'solid-velocity-bc', dest='solid_velocity_bc',
                          default=True,
                          help='Apply velocity bc to solids in Elastic dynamics')

        add_bool_argument(group, 'wall-pst', dest='wall_pst',
                          default=True, help='Add wall as PST source')

        add_bool_argument(group, 'damping',
                          dest='damping',
                          default=False,
                          help='Use damping')

        group.add_argument("--damping-coeff", action="store",
                           dest="damping_coeff", default=0.000, type=float,
                           help="Damping coefficient for Bui")

        choices = ['sun2019', 'ipst']
        group.add_argument(
            "--solid-pst", action="store", dest='solid_pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use for elastic solid (one of %s)." % choices)

    def consume_user_options(self, options):
        vars = ['alpha_fluid', 'alpha_solid', 'beta_solid',
                'edac_alpha', 'wall_pst', 'damping', 'damping_coeff',
                'solid_velocity_bc', 'solid_pst']
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

    def check_ipst_time(self, t, dt):
        if int(t / dt) % self.ipst_interval == 0:
            return True
        else:
            return False

    def get_equations(self):
        # from pysph.sph.wc.gtvf import (MomentumEquationArtificialStress)
        from fluids_wcsph import (
            FluidSetWallVelocityUFreeSlipAndNoSlip,

            FluidContinuityEquationWCSPHOnFluid,
            FluidContinuityEquationWCSPHOnFluidSolid,

            FluidEDACEquationWCSPHOnFluid,
            FluidEDACEquationWCSPHOnFluidSolid,

            StateEquation,
            FluidSolidWallPressureBCFluidSolid,
            FluidClampWallPressureFluidSolid,
            FluidMomentumEquationPressureGradient,

            MomentumEquationViscosity as FluidMomentumEquationViscosity,
            MomentumEquationArtificialViscosity as
            FluidMomentumEquationArtificialViscosity)

        from solid_mech import (
            ElasticSolidSetWallVelocityNoSlipU,
            ElasticSolidSetWallVelocityNoSlipUhat,

            ElasticSolidContinuityEquationUhat,
            ElasticSolidContinuityEquationETVFCorrection,
            VelocityGradient2D,
            VelocityGradient2DUhat,

            ElasticSolidContinuityEquationUhatSolid,
            ElasticSolidContinuityEquationETVFCorrectionSolid,
            VelocityGradient2DSolid,
            VelocityGradient2DSolidUhat,
            HookesDeviatoricStressRate,

            SetHIJForInsideParticles,

            IsothermalEOS,

            AdamiBoundaryConditionExtrapolateNoSlip,

            ElasticSolidMonaghanArtificialViscosity,
            ElasticSolidMomentumEquation,
            ElasticSolidComputeAuHatETVFSun2019,
            AddGravityToStructure,
            BuiFukagawaDampingGraularSPH)

        from ipst import (MakeAuhatZero, SavePositionsIPSTBeforeMoving,
                          AdjustPositionIPST,
                          CheckUniformityIPST,
                          ComputeAuhatETVFIPSTSolids,
                          ResetParticlePositionsIPST)

        nu_edac = self._get_edac_nu()
        all = self.fluids + self.solids
        stage1 = []

        # =========================#
        # fluid equations
        # =========================#
        # ============================
        # Continuity and EDAC equation
        # ============================
        eqs = []
        for fluid in self.fluids:
            eqs.append(FluidContinuityEquationWCSPHOnFluid(
                dest=fluid,
                sources=self.fluids))

            if len(self.solids) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnFluidSolid(
                    dest=fluid,
                    sources=self.solids))

            if len(self.structures) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnStructure(
                    dest=fluid,
                    sources=self.structures))

            if len(self.structure_solids) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnStructureSolid(
                    dest=fluid,
                    sources=self.structure_solids))

        stage1.append(Group(equations=eqs, real=False))
        # ============================
        # Continuity and EDAC equation
        # ============================
        # =========================#
        # fluid equations ends
        # =========================#

        # =========================#
        # structure equations
        # =========================#

        all = self.structures + self.structure_solids

        if self.solid_velocity_bc is True:
            tmp = []
            if len(self.structure_solids) > 0:
                for boundary in self.structure_solids:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipU(
                            dest=boundary, sources=self.structures))
                stage1.append(Group(equations=tmp))

            tmp = []
            if len(self.structure_solids) > 0:
                for boundary in self.structure_solids:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipUhat(
                            dest=boundary, sources=self.structures))
                stage1.append(Group(equations=tmp))

        # ===================================================== #
        # Continuity, Velocity gradient and Hooke's Stress rate
        # ===================================================== #
        g1 = []
        if len(self.structures) > 0:
            for structure in self.structures:
                g1.append(ElasticSolidContinuityEquationUhat(
                    dest=structure, sources=self.structures))
                g1.append(
                    ElasticSolidContinuityEquationETVFCorrection(
                        dest=structure, sources=self.structures))

                g1.append(
                    VelocityGradient2D(
                        dest=structure, sources=self.structures))

                if len(self.structure_solids) > 0:
                    g1.append(
                        ElasticSolidContinuityEquationUhatSolid(dest=structure,
                                                                sources=self.structure_solids))

                    g1.append(
                        ElasticSolidContinuityEquationETVFCorrectionSolid(
                            dest=structure, sources=self.structure_solids))

                    g1.append(
                        VelocityGradient2DSolid(dest=structure,
                                                sources=self.structure_solids))

            stage1.append(Group(equations=g1))

            g2 = []
            for structure in self.structures:
                g2.append(
                    HookesDeviatoricStressRate(dest=structure, sources=None))

            stage1.append(Group(equations=g2))
        # =========================#
        # structure equations ends
        # =========================#

        # =========================#
        # stage 2 equations start
        # =========================#
        stage2 = []

        tmp = []
        for fluid in self.fluids:
            tmp.append(
                StateEquation(dest=fluid, sources=None, p0=self.pb_fluid,
                              rho0=self.rho0_fluid))

        stage2.append(Group(equations=tmp, real=False))

        if len(self.solids) > 0:
            eqs = []
            for solid in self.solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=solid,
                                                           sources=self.fluids))
                eqs.append(
                    FluidSolidWallPressureBCFluidSolid(dest=solid, sources=self.fluids,
                                                       gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    FluidClampWallPressureFluidSolid(dest=solid, sources=None,))

            stage2.append(Group(equations=eqs, real=False))

        # FSI coupling equations, set the pressure
        if len(self.structure_solids) > 0:
            eqs = []
            for structure in self.structure_solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=structure,
                                                           sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCStructureSolid(
                        dest=structure, sources=self.fluids,
                        gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    FluidClampWallPressureStructureSolid(
                        dest=structure, sources=None))

            stage2.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        # FSI coupling equations, set the pressure
        if len(self.structures) > 0:
            eqs = []
            for structure in self.structures:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(
                        dest=structure, sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCStructure(dest=structure,
                                                      sources=self.fluids,
                                                      gx=self.gx, gy=self.gy,
                                                      gz=self.gz))
                # eqs.append(
                #     FluidClampWallPressureStructure(dest=structure, sources=None))

            stage2.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        eqs = []
        for fluid in self.fluids:
            if self.alpha_fluid > 0.:
                eqs.append(
                    FluidMomentumEquationArtificialViscosity(
                        dest=fluid,
                        sources=self.fluids+self.solids+self.structures
                        + self.structure_solids,
                        c0=self.c0_fluid,
                        alpha=self.alpha_fluid
                    )
                )

            if self.nu_fluid > 0.:
                eqs.append(
                    FluidLaminarViscosityFluid(
                        dest=fluid, sources=self.fluids, nu=self.nu_fluid
                    ))

                eqs.append(
                    FluidLaminarViscosityFluidSolid(
                        dest=fluid, sources=self.solids, nu=self.nu_fluid
                    ))

                if len(self.structures) > 0:
                    eqs.append(
                        FluidLaminarViscosityStructure(
                            dest=fluid, sources=self.structures,
                            nu=self.nu_fluid))

                if len(self.structure_solids) > 0:
                    eqs.append(
                        FluidLaminarViscosityStructureSolid(
                            dest=fluid, sources=self.structure_solids,
                            nu=self.nu_fluid))

            eqs.append(
                FluidMomentumEquationPressureGradient(
                    dest=fluid, sources=self.fluids + self.solids,
                    gx=self.gx, gy=self.gy, gz=self.gz), )

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
                        ElasticSolidMonaghanArtificialViscosity(
                            dest=structure,
                            sources=self.structures + self.structure_solids,
                            alpha=self.alpha_solid,
                            beta=self.beta_solid))

                g4.append(
                    ElasticSolidMomentumEquation(
                        dest=structure,
                        sources=self.structures + self.structure_solids))

                if self.solid_pst == "sun2019":
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

                if self.nu_fluid > 0.:
                    g4.append(
                        AccelerationOnStructureDueToFluidViscosity(
                            dest=structure, sources=self.fluids, nu=self.nu_fluid))

            stage2.append(Group(g4))

            # this PST is handled separately
            if self.solid_pst == "ipst" and self.wall_pst is True:
                g5 = []
                g6 = []
                g7 = []
                g8 = []

                # make auhat zero before computation of ipst force
                eqns = []
                for structure in self.structures:
                    eqns.append(MakeAuhatZero(dest=structure, sources=None))

                stage2.append(Group(eqns))

                for structure in self.structures:
                    g5.append(
                        SavePositionsIPSTBeforeMoving(dest=structure, sources=None))

                    # these two has to be in the iterative group and the nnps has to
                    # be updated
                    # ---------------------------------------
                    g6.append(
                        AdjustPositionIPST(dest=structure,
                                           sources=[structure] + self.structure_solids,
                                           u_max=self.structure_u_max))

                    g7.append(
                        CheckUniformityIPST(dest=structure,
                                            sources=[structure] + self.structure_solids,
                                            debug=self.debug))
                    # ---------------------------------------

                    g8.append(ComputeAuhatETVFIPSTSolids(dest=structure,
                                                         sources=None))
                    g8.append(ResetParticlePositionsIPST(dest=structure,
                                                         sources=None))

                stage2.append(Group(g5, condition=self.check_ipst_time))

                # this is the iterative group
                stage2.append(
                    Group(equations=[Group(equations=g6),
                                     Group(equations=g7)], iterate=True,
                          max_iterations=self.ipst_max_iterations,
                          min_iterations=self.ipst_min_iterations,
                          condition=self.check_ipst_time))

                stage2.append(Group(g8, condition=self.check_ipst_time))

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
        from solid_mech import (get_shear_modulus, get_speed_of_sound)
        from ipst import (setup_ipst, QuinticSpline)

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

            # for isothermal eqn
            pa.rho_ref[:] = pa.rho[:]

            # this will change
            kernel = QuinticSpline(dim=self.dim)
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
            if self.edac is True:
                add_properties(pa, 'ap')

            if self.solid_pst == "ipst":
                setup_ipst(pa, QuinticSpline)

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


class FSIWCSPHFluidsSubSteppingScheme(FSIWCSPHFluidsScheme):
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
        from fluids_wcsph import (
            FluidSetWallVelocityUFreeSlipAndNoSlip,

            FluidContinuityEquationWCSPHOnFluid,
            FluidContinuityEquationWCSPHOnFluidSolid,

            FluidEDACEquationWCSPHOnFluid,
            FluidEDACEquationWCSPHOnFluidSolid,

            FluidSolidWallPressureBCFluidSolid,
            FluidClampWallPressureFluidSolid,
            FluidMomentumEquationPressureGradient,

            MomentumEquationViscosity as FluidMomentumEquationViscosity,
            MomentumEquationArtificialViscosity as
            FluidMomentumEquationArtificialViscosity)

        from solid_mech import (
            ElasticSolidSetWallVelocityNoSlipU,
            ElasticSolidSetWallVelocityNoSlipUhat,

            ElasticSolidContinuityEquationUhat,
            ElasticSolidContinuityEquationETVFCorrection,
            VelocityGradient2D,

            ElasticSolidContinuityEquationUhatSolid,
            ElasticSolidContinuityEquationETVFCorrectionSolid,
            VelocityGradient2DSolid,

            HookesDeviatoricStressRate,

            SetHIJForInsideParticles,

            IsothermalEOS,

            AdamiBoundaryConditionExtrapolateNoSlip,

            ElasticSolidMonaghanArtificialViscosity,
            ElasticSolidMomentumEquation,
            ElasticSolidComputeAuHatETVFSun2019,
            AddGravityToStructure,
            BuiFukagawaDampingGraularSPH)

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

            for solid in self.solids:
                eqs_u.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(
                        dest=solid, sources=self.fluids))

            stage1.append(Group(equations=eqs_u, real=False))

        if len(self.structure_solids) > 0:
            eqs_u = []

            for solid in self.structure_solids:
                eqs_u.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(
                        dest=solid, sources=self.fluids))

            stage1.append(Group(equations=eqs_u, real=False))

        # ============================
        # Continuity and EDAC equation
        # ============================
        eqs = []
        for fluid in self.fluids:
            eqs.append(FluidContinuityEquationWCSPHOnFluid(
                dest=fluid,
                sources=self.fluids))

            eqs.append(FluidEDACEquationWCSPHOnFluid(
                dest=fluid,
                sources=self.fluids,
                nu=nu_edac
            ))

            if len(self.solids) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnFluidSolid(
                    dest=fluid,
                    sources=self.solids))

                eqs.append(FluidEDACEquationWCSPHOnFluidSolid(
                    dest=fluid,
                    sources=self.solids,
                    nu=nu_edac
                ))

            if len(self.structures) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnStructure(
                    dest=fluid,
                    sources=self.structures))

                eqs.append(FluidEDACEquationWCSPHOnStructure(
                    dest=fluid,
                    sources=self.structures,
                    nu=nu_edac
                ))

            if len(self.structure_solids) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnStructureSolid(
                    dest=fluid,
                    sources=self.structure_solids))

                eqs.append(FluidEDACEquationWCSPHOnStructureSolid(
                    dest=fluid,
                    sources=self.structure_solids,
                    nu=nu_edac
                ))

        stage1.append(Group(equations=eqs, real=False))
        # ============================
        # Continuity and EDAC equation
        # ============================

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
                    FluidSolidWallPressureBCFluidSolid(dest=solid, sources=self.fluids,
                                                       gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    FluidClampWallPressureFluidSolid(dest=solid, sources=None))

            stage1.append(Group(equations=eqs, real=False))

        # FSI coupling equations, set the pressure
        if len(self.structure_solids) > 0:
            eqs = []
            for structure in self.structure_solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=structure,
                                                           sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCStructureSolid(
                        dest=structure, sources=self.fluids,
                        gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    FluidClampWallPressureStructureSolid(
                        dest=structure, sources=None))

            stage1.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        # FSI coupling equations, set the pressure
        if len(self.structures) > 0:
            eqs = []
            for structure in self.structures:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(
                        dest=structure, sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCStructure(dest=structure,
                                                      sources=self.fluids,
                                                      gx=self.gx, gy=self.gy,
                                                      gz=self.gz))
                # eqs.append(
                #     FluidClampWallPressureStructure(dest=structure, sources=None))

            stage1.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        eqs = []
        for fluid in self.fluids:
            if self.alpha_fluid > 0.:
                eqs.append(
                    FluidMomentumEquationArtificialViscosity(
                        dest=fluid,
                        sources=self.fluids+self.solids+self.structures
                        + self.structure_solids,
                        c0=self.c0_fluid,
                        alpha=self.alpha_fluid
                    )
                )

            if self.nu_fluid > 0.:
                eqs.append(
                    FluidLaminarViscosityFluid(
                        dest=fluid, sources=self.fluids, nu=self.nu_fluid
                    ))

                eqs.append(
                    FluidLaminarViscosityFluidSolid(
                        dest=fluid, sources=self.solids, nu=self.nu_fluid
                    ))

                if len(self.structures) > 0:
                    eqs.append(
                        FluidLaminarViscosityStructure(
                            dest=fluid, sources=self.structures,
                            nu=self.nu_fluid))

                if len(self.structure_solids) > 0:
                    eqs.append(
                        FluidLaminarViscosityStructureSolid(
                            dest=fluid, sources=self.structure_solids,
                            nu=self.nu_fluid))

            eqs.append(
                FluidMomentumEquationPressureGradient(
                    dest=fluid, sources=self.fluids + self.solids,
                    gx=self.gx, gy=self.gy, gz=self.gz), )

            if len(self.structure_solids + self.structures) > 0.:
                eqs.append(
                    AccelerationOnFluidDueToStructure(
                        dest=fluid,
                        sources=self.structures + self.structure_solids),
                )

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

        if self.solid_velocity_bc is True:
            tmp = []
            if len(self.structure_solids) > 0:
                for boundary in self.structure_solids:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipU(
                            dest=boundary, sources=self.structures))
                stage1_stucture_eqs.append(Group(equations=tmp))

            tmp = []
            if len(self.structure_solids) > 0:
                for boundary in self.structure_solids:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipUhat(
                            dest=boundary, sources=self.structures))
                stage1_stucture_eqs.append(Group(equations=tmp))

        g1 = []
        if len(self.structures) > 0.:
            for structure in self.structures:
                g1.append(ElasticSolidContinuityEquationUhat(
                    dest=structure, sources=self.structures))
                g1.append(
                    ElasticSolidContinuityEquationETVFCorrection(
                        dest=structure, sources=self.structures))
                g1.append(
                    VelocityGradient2D(
                        dest=structure, sources=self.structures))

                if len(self.structure_solids) > 0:
                    g1.append(
                        ElasticSolidContinuityEquationUhatSolid(dest=structure,
                                                                sources=self.structure_solids))

                    g1.append(
                        ElasticSolidContinuityEquationETVFCorrectionSolid(
                            dest=structure, sources=self.structure_solids))

                    g1.append(
                        VelocityGradient2DSolid(dest=structure, sources=self.structure_solids))

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
                        ElasticSolidMonaghanArtificialViscosity(
                            dest=structure,
                            sources=self.structures + self.structure_solids,
                            alpha=self.alpha_solid,
                            beta=self.beta_solid))

                g4.append(
                    ElasticSolidMomentumEquation(
                        dest=structure,
                        sources=self.structures + self.structure_solids))

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

                if self.nu_fluid > 0.:
                    g4.append(
                        AccelerationOnStructureDueToFluidViscosity(
                            dest=structure, sources=self.fluids,
                            nu=self.nu_fluid))

            stage1_stucture_eqs.append(Group(g4))

            # Add gravity
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


class FSIWCSPHScheme(FSIWCSPHFluidsScheme):
    def get_equations(self):
        # from pysph.sph.wc.gtvf import (MomentumEquationArtificialStress)
        from fluids_wcsph import (
            FluidSetWallVelocityUFreeSlipAndNoSlip,

            FluidContinuityEquationWCSPHOnFluid,
            FluidContinuityEquationWCSPHOnFluidSolid,

            FluidEDACEquationWCSPHOnFluid,
            FluidEDACEquationWCSPHOnFluidSolid,

            StateEquation,
            FluidSolidWallPressureBCFluidSolid,
            FluidClampWallPressureFluidSolid,
            FluidMomentumEquationPressureGradient,

            MomentumEquationViscosity as FluidMomentumEquationViscosity,
            MomentumEquationArtificialViscosity as
            FluidMomentumEquationArtificialViscosity)

        from solid_mech import (
            ElasticSolidSetWallVelocityNoSlipU,
            ElasticSolidSetWallVelocityNoSlipUhat,

            ElasticSolidContinuityEquationU,
            VelocityGradient2D,

            ElasticSolidContinuityEquationUSolid,
            VelocityGradient2DSolid,

            HookesDeviatoricStressRate,

            IsothermalEOS,

            AdamiBoundaryConditionExtrapolateNoSlip,

            ElasticSolidMonaghanArtificialViscosity,
            AddGravityToStructure,
            BuiFukagawaDampingGraularSPH)

        from pysph.sph.solid_mech.basic import (
            MonaghanArtificialStress,

            MomentumEquationWithStress as ElasticSolidMomentumEquationWithStress,
        )


        nu_edac = self._get_edac_nu()
        all = self.fluids + self.solids
        stage1 = []

        # =========================#
        # fluid equations
        # =========================#
        # ============================
        # Continuity and EDAC equation
        # ============================
        eqs = []
        for fluid in self.fluids:
            eqs.append(FluidContinuityEquationWCSPHOnFluid(
                dest=fluid,
                sources=self.fluids))

            if len(self.solids) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnFluidSolid(
                    dest=fluid,
                    sources=self.solids))

            if len(self.structures) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnStructure(
                    dest=fluid,
                    sources=self.structures))

            if len(self.structure_solids) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnStructureSolid(
                    dest=fluid,
                    sources=self.structure_solids))

        stage1.append(Group(equations=eqs, real=False))
        # ============================
        # Continuity and EDAC equation
        # ============================
        # =========================#
        # fluid equations ends
        # =========================#

        # =========================#
        # structure equations
        # =========================#

        all = self.structures + self.structure_solids

        if self.solid_velocity_bc is True:
            tmp = []
            if len(self.structure_solids) > 0:
                for boundary in self.structure_solids:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipU(
                            dest=boundary, sources=self.structures))
                stage1.append(Group(equations=tmp))

            tmp = []
            if len(self.structure_solids) > 0:
                for boundary in self.structure_solids:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipUhat(
                            dest=boundary, sources=self.structures))
                stage1.append(Group(equations=tmp))

        # ===================================================== #
        # Continuity, Velocity gradient and Hooke's Stress rate
        # ===================================================== #
        g1 = []
        if len(self.structures) > 0:
            for structure in self.structures:
                g1.append(ElasticSolidContinuityEquationU(
                    dest=structure, sources=self.structures))
                g1.append(
                    VelocityGradient2D(
                        dest=structure, sources=self.structures))

                if len(self.structure_solids) > 0:
                    g1.append(
                        ElasticSolidContinuityEquationUSolid(dest=structure,
                                                             sources=self.structure_solids))

                    g1.append(
                        VelocityGradient2DSolid(dest=structure, sources=self.structure_solids))

            stage1.append(Group(equations=g1))

            g2 = []
            for structure in self.structures:
                g2.append(
                    MonaghanArtificialStress(dest=structure, sources=None,
                                             eps=self.artificial_stress_eps))

                g2.append(
                    HookesDeviatoricStressRate(dest=structure, sources=None))

            stage1.append(Group(equations=g2))
        # =========================#
        # structure equations ends
        # =========================#

        # =========================#
        # stage 2 equations start
        # =========================#
        stage2 = []

        tmp = []
        for fluid in self.fluids:
            tmp.append(
                StateEquation(dest=fluid, sources=None, p0=self.pb_fluid,
                              rho0=self.rho0_fluid))

        stage2.append(Group(equations=tmp, real=False))

        if len(self.solids) > 0:
            eqs = []
            for solid in self.solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=solid,
                                                           sources=self.fluids))
                eqs.append(
                    FluidSolidWallPressureBCFluidSolid(dest=solid, sources=self.fluids,
                                                       gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    FluidClampWallPressureFluidSolid(dest=solid, sources=None,))

            stage2.append(Group(equations=eqs, real=False))

        # FSI coupling equations, set the pressure
        if len(self.structure_solids) > 0:
            eqs = []
            for structure in self.structure_solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=structure,
                                                           sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCStructureSolid(
                        dest=structure, sources=self.fluids,
                        gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    FluidClampWallPressureStructureSolid(
                        dest=structure, sources=None))

            stage2.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        # FSI coupling equations, set the pressure
        if len(self.structures) > 0:
            eqs = []
            for structure in self.structures:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(
                        dest=structure, sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCStructure(dest=structure,
                                                      sources=self.fluids,
                                                      gx=self.gx, gy=self.gy,
                                                      gz=self.gz))
                # eqs.append(
                #     FluidClampWallPressureStructure(dest=structure, sources=None))

            stage2.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        eqs = []
        for fluid in self.fluids:
            if self.alpha_fluid > 0.:
                eqs.append(
                    FluidMomentumEquationArtificialViscosity(
                        dest=fluid,
                        sources=self.fluids+self.solids+self.structures
                        + self.structure_solids,
                        c0=self.c0_fluid,
                        alpha=self.alpha_fluid
                    )
                )

            if self.nu_fluid > 0.:
                eqs.append(
                    FluidLaminarViscosityFluid(
                        dest=fluid, sources=self.fluids, nu=self.nu_fluid
                    ))

                eqs.append(
                    FluidLaminarViscosityFluidSolid(
                        dest=fluid, sources=self.solids, nu=self.nu_fluid
                    ))

                if len(self.structures) > 0:
                    eqs.append(
                        FluidLaminarViscosityStructure(
                            dest=fluid, sources=self.structures,
                            nu=self.nu_fluid))

                if len(self.structure_solids) > 0:
                    eqs.append(
                        FluidLaminarViscosityStructureSolid(
                            dest=fluid, sources=self.structure_solids,
                            nu=self.nu_fluid))

            eqs.append(
                FluidMomentumEquationPressureGradient(
                    dest=fluid, sources=self.fluids + self.solids,
                    gx=self.gx, gy=self.gy, gz=self.gz), )

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
                        ElasticSolidMonaghanArtificialViscosity(
                            dest=structure,
                            sources=self.structures + self.structure_solids,
                            alpha=self.alpha_solid,
                            beta=self.beta_solid))

                g4.append(
                    ElasticSolidMomentumEquationWithStress(
                        dest=structure,
                        sources=self.structures + self.structure_solids))

                g4.append(
                    AccelerationOnStructureDueToFluid(dest=structure,
                                                      sources=self.fluids), )

                if self.nu_fluid > 0.:
                    g4.append(
                        AccelerationOnStructureDueToFluidViscosity(
                            dest=structure, sources=self.fluids, nu=self.nu_fluid))

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
        from solid_mech import (get_shear_modulus, get_speed_of_sound)

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
                           'uhat', 'vhat', 'what', 'ap',
                           'r00', 'r01', 'r02', 'r11', 'r12', 'r22')
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
                           'wij', 'uhat', 'vhat', 'what',
                           'r00', 'r01', 'r02', 'r11', 'r12', 'r22')
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

            # for isothermal eqn
            pa.rho_ref[:] = pa.rho[:]

            # this will change
            kernel = QuinticSpline(dim=self.dim)
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


class FSIWCSPHSubSteppingScheme(FSIWCSPHFluidsScheme):
    def attributes_changed(self):
        super().attributes_changed()

        self.dt_factor = int(self.dt_fluid / self.dt_solid) + 1
        self.dt_fluid_simulated = self.dt_factor * self.dt_solid

    def get_equations(self):
        from fluids_wcsph import (
            FluidSetWallVelocityUFreeSlipAndNoSlip,

            FluidContinuityEquationWCSPHOnFluid,
            FluidContinuityEquationWCSPHOnFluidSolid,

            FluidEDACEquationWCSPHOnFluid,
            FluidEDACEquationWCSPHOnFluidSolid,

            StateEquation,
            FluidSolidWallPressureBCFluidSolid,
            FluidClampWallPressureFluidSolid,
            FluidMomentumEquationPressureGradient,

            MomentumEquationViscosity as FluidMomentumEquationViscosity,
            MomentumEquationArtificialViscosity as
            FluidMomentumEquationArtificialViscosity)

        from solid_mech import (
            ElasticSolidSetWallVelocityNoSlipU,
            ElasticSolidSetWallVelocityNoSlipUhat,

            ElasticSolidContinuityEquationU,
            VelocityGradient2D,

            ElasticSolidContinuityEquationUSolid,
            VelocityGradient2DSolid,

            HookesDeviatoricStressRate,

            IsothermalEOS,

            AdamiBoundaryConditionExtrapolateNoSlip,

            ElasticSolidMonaghanArtificialViscosity,
            AddGravityToStructure,
            BuiFukagawaDampingGraularSPH)

        from pysph.sph.solid_mech.basic import (
            MonaghanArtificialStress,

            MomentumEquationWithStress as ElasticSolidMomentumEquationWithStress,
        )

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

            for solid in self.solids:
                eqs_u.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(
                        dest=solid, sources=self.fluids))

            stage1.append(Group(equations=eqs_u, real=False))

        if len(self.structure_solids) > 0:
            eqs_u = []

            for solid in self.structure_solids:
                eqs_u.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(
                        dest=solid, sources=self.fluids))

            stage1.append(Group(equations=eqs_u, real=False))

        # ============================
        # Continuity and EDAC equation
        # ============================
        eqs = []
        for fluid in self.fluids:
            eqs.append(FluidContinuityEquationWCSPHOnFluid(
                dest=fluid,
                sources=self.fluids))

            eqs.append(FluidEDACEquationWCSPHOnFluid(
                dest=fluid,
                sources=self.fluids,
                nu=nu_edac
            ))

            if len(self.solids) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnFluidSolid(
                    dest=fluid,
                    sources=self.solids))

                eqs.append(FluidEDACEquationWCSPHOnFluidSolid(
                    dest=fluid,
                    sources=self.solids,
                    nu=nu_edac
                ))

            if len(self.structures) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnStructure(
                    dest=fluid,
                    sources=self.structures))

                eqs.append(FluidEDACEquationWCSPHOnStructure(
                    dest=fluid,
                    sources=self.structures,
                    nu=nu_edac
                ))

            if len(self.structure_solids) > 0:
                eqs.append(FluidContinuityEquationWCSPHOnStructureSolid(
                    dest=fluid,
                    sources=self.structure_solids))

                eqs.append(FluidEDACEquationWCSPHOnStructureSolid(
                    dest=fluid,
                    sources=self.structure_solids,
                    nu=nu_edac
                ))

        stage1.append(Group(equations=eqs, real=False))
        # ============================
        # Continuity and EDAC equation
        # ============================

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
                    FluidSolidWallPressureBCFluidSolid(dest=solid, sources=self.fluids,
                                                       gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    FluidClampWallPressureFluidSolid(dest=solid, sources=None))

            stage1.append(Group(equations=eqs, real=False))

        # FSI coupling equations, set the pressure
        if len(self.structure_solids) > 0:
            eqs = []
            for structure in self.structure_solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=structure,
                                                           sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCStructureSolid(
                        dest=structure, sources=self.fluids,
                        gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    FluidClampWallPressureStructureSolid(
                        dest=structure, sources=None))

            stage1.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        # FSI coupling equations, set the pressure
        if len(self.structures) > 0:
            eqs = []
            for structure in self.structures:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(
                        dest=structure, sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCStructure(dest=structure,
                                                      sources=self.fluids,
                                                      gx=self.gx, gy=self.gy,
                                                      gz=self.gz))
                # eqs.append(
                #     FluidClampWallPressureStructure(dest=structure, sources=None))

            stage1.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        eqs = []
        for fluid in self.fluids:
            if self.alpha_fluid > 0.:
                eqs.append(
                    FluidMomentumEquationArtificialViscosity(
                        dest=fluid,
                        sources=self.fluids+self.solids+self.structures
                        + self.structure_solids,
                        c0=self.c0_fluid,
                        alpha=self.alpha_fluid
                    )
                )

            if self.nu_fluid > 0.:
                eqs.append(
                    FluidLaminarViscosityFluid(
                        dest=fluid, sources=self.fluids, nu=self.nu_fluid
                    ))

                eqs.append(
                    FluidLaminarViscosityFluidSolid(
                        dest=fluid, sources=self.solids, nu=self.nu_fluid
                    ))

                if len(self.structures) > 0:
                    eqs.append(
                        FluidLaminarViscosityStructure(
                            dest=fluid, sources=self.structures,
                            nu=self.nu_fluid))

                if len(self.structure_solids) > 0:
                    eqs.append(
                        FluidLaminarViscosityStructureSolid(
                            dest=fluid, sources=self.structure_solids,
                            nu=self.nu_fluid))

            eqs.append(
                FluidMomentumEquationPressureGradient(
                    dest=fluid, sources=self.fluids + self.solids,
                    gx=self.gx, gy=self.gy, gz=self.gz), )

            if len(self.structure_solids + self.structures) > 0.:
                eqs.append(
                    AccelerationOnFluidDueToStructure(
                        dest=fluid,
                        sources=self.structures + self.structure_solids),
                )

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

        if self.solid_velocity_bc is True:
            tmp = []
            if len(self.structure_solids) > 0:
                for boundary in self.structure_solids:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipU(
                            dest=boundary, sources=self.structures))
                stage1_stucture_eqs.append(Group(equations=tmp))

            tmp = []
            if len(self.structure_solids) > 0:
                for boundary in self.structure_solids:
                    tmp.append(
                        ElasticSolidSetWallVelocityNoSlipUhat(
                            dest=boundary, sources=self.structures))
                stage1_stucture_eqs.append(Group(equations=tmp))

        g1 = []
        if len(self.structures) > 0.:
            for structure in self.structures:
                g1.append(ElasticSolidContinuityEquationU(
                    dest=structure, sources=self.structures))
                g1.append(
                    VelocityGradient2D(
                        dest=structure, sources=self.structures))

                if len(self.structure_solids) > 0:
                    g1.append(
                        ElasticSolidContinuityEquationUSolid(dest=structure,
                                                             sources=self.structure_solids))

                    g1.append(
                        VelocityGradient2DSolid(dest=structure, sources=self.structure_solids))

            stage1_stucture_eqs.append(Group(equations=g1))

            g2 = []
            for structure in self.structures:
                g2.append(
                    MonaghanArtificialStress(dest=structure, sources=None,
                                             eps=self.artificial_stress_eps))

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
                        ElasticSolidMonaghanArtificialViscosity(
                            dest=structure,
                            sources=self.structures + self.structure_solids,
                            alpha=self.alpha_solid,
                            beta=self.beta_solid))

                g4.append(
                    ElasticSolidMomentumEquationWithStress(
                        dest=structure,
                        sources=self.structures + self.structure_solids))

                g4.append(
                    AccelerationOnStructureDueToFluid(dest=structure,
                                                      sources=self.fluids), )

                if self.nu_fluid > 0.:
                    g4.append(
                        AccelerationOnStructureDueToFluidViscosity(
                            dest=structure, sources=self.fluids,
                            nu=self.nu_fluid))

            stage1_stucture_eqs.append(Group(g4))

            # Add gravity
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
