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

# from solid_mech import (ComputeAuHatETVFSun2019,
#                         SavePositionsIPSTBeforeMoving, AdjustPositionIPST,
#                         CheckUniformityIPST, ComputeAuhatETVFIPST,
#                         ResetParticlePositionsIPST, EDACEquation, setup_ipst,
#                         SetHIJForInsideParticles)

from pysph.sph.integrator import PECIntegrator
from boundary_particles import (add_boundary_identification_properties)

from boundary_particles import (ComputeNormals, SmoothNormals,
                                IdentifyBoundaryParticleCosAngleEDAC)

from pysph.sph.wc.gtvf import MomentumEquationViscosity
from pysph.sph.wc.transport_velocity import (
    MomentumEquationArtificialViscosity)

from common import EDACIntegrator
from pysph.examples.solid_mech.impact import add_properties
from pysph.sph.wc.linalg import mat_vec_mult


class FluidSetWallVelocityUFreeSlipAndNoSlip(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf, s_u, s_v, s_w, d_wij, WIJ):
        # normalisation factor is different from 'V' as the particles
        # near the boundary do not have full kernel support
        d_wij[d_idx] += WIJ

        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_u[s_idx] * WIJ
        d_vf[d_idx] += s_v[s_idx] * WIJ
        d_wf[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ugfs, d_vgfs, d_wgfs,
                  d_ugns, d_vgns, d_wgns, d_u, d_v, d_w, d_normal):
        idx3 = declare('int', 1)
        idx3 = 3 * d_idx
        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        tmp1 = d_u[d_idx] - d_uf[d_idx]
        tmp2 = d_v[d_idx] - d_vf[d_idx]
        tmp3 = d_w[d_idx] - d_wf[d_idx]

        projection = (tmp1 * d_normal[idx3] + tmp2 * d_normal[idx3 + 1] +
                      tmp3 * d_normal[idx3 + 2])

        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ugfs[d_idx] = 2 * projection * d_normal[idx3] + d_uf[d_idx]
        d_vgfs[d_idx] = 2 * projection * d_normal[idx3 + 1] + d_vf[d_idx]
        d_wgfs[d_idx] = 2 * projection * d_normal[idx3 + 2] + d_wf[d_idx]

        # vn = (d_ugfs[d_idx]*d_normal[idx3] + d_vgfs[d_idx]*d_normal[idx3+1]
        #       + d_wgfs[d_idx]*d_normal[idx3+2])
        # if vn < 0:
        #     d_ugfs[d_idx] -= vn*d_normal[idx3]
        #     d_vgfs[d_idx] -= vn*d_normal[idx3+1]
        #     d_wgfs[d_idx] -= vn*d_normal[idx3+2]

        # For No slip boundary conditions
        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ugns[d_idx] = 2 * d_u[d_idx] - d_uf[d_idx]
        d_vgns[d_idx] = 2 * d_v[d_idx] - d_vf[d_idx]
        d_wgns[d_idx] = 2 * d_w[d_idx] - d_wf[d_idx]

        vn = (d_ugns[d_idx]*d_normal[idx3] + d_vgns[d_idx]*d_normal[idx3+1]
              + d_wgns[d_idx]*d_normal[idx3+2])
        if vn < 0:
            d_ugns[d_idx] -= vn*d_normal[idx3]
            d_vgns[d_idx] -= vn*d_normal[idx3+1]
            d_wgns[d_idx] -= vn*d_normal[idx3+2]


class FluidContinuityEquationWCSPHOnFluid(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_uhat, d_vhat, d_what,
             s_uhat, s_vhat, s_what, d_arho, DWIJ, VIJ):
        udotdij = DWIJ[0] * VIJ[0] + DWIJ[1] * VIJ[1] + DWIJ[2] * VIJ[2]
        fac = d_rho[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_arho[d_idx] += fac * udotdij


class FluidContinuityEquationWCSPHOnFluidSolid(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_u, d_v, d_w,
             s_ugfs, s_vgfs, s_wgfs, d_arho, DWIJ):
        uhatij = d_u[d_idx] - s_ugfs[s_idx] * 0.
        vhatij = d_v[d_idx] - s_vgfs[s_idx] * 0.
        whatij = d_w[d_idx] - s_wgfs[s_idx] * 0.

        udotdij = DWIJ[0] * uhatij + DWIJ[1] * vhatij + DWIJ[2] * whatij
        fac = d_rho[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_arho[d_idx] += fac * udotdij


class FluidEDACEquationWCSPHOnFluid(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidEDACEquationWCSPHOnFluid, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_cs, d_u, d_v, d_w, s_p, s_m,
             s_rho, d_ap, DWIJ, XIJ, s_u, s_v, s_w, R2IJ, VIJ, EPS):
        cs2 = d_cs[d_idx] * d_cs[d_idx]

        rhoj1 = 1.0 / s_rho[s_idx]
        Vj = s_m[s_idx] * rhoj1
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
        rhoij = d_rho[d_idx] + s_rho[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p[s_idx])


class FluidEDACEquationWCSPHOnFluidSolid(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidEDACEquationWCSPHOnFluidSolid, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_cs, d_u, d_v, d_w, s_p, s_m,
             s_rho, s_ugfs, s_vgfs, s_wgfs, d_ap, DWIJ, XIJ, s_u, s_v, s_w,
             R2IJ, VIJ, EPS):
        cs2 = d_cs[d_idx] * d_cs[d_idx]

        rhoj1 = 1.0 / s_rho[s_idx]
        Vj = s_m[s_idx] * rhoj1
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
        rhoij = d_rho[d_idx] + s_rho[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p[s_idx])


class FluidClampWallPressureFluidSolid(Equation):
    def post_loop(self, d_idx, d_p):
        if d_p[d_idx] < 0.0:
            d_p[d_idx] = 0.0


class FluidSolidWallPressureBCFluidSolid(Equation):
    r"""Solid wall pressure boundary condition from Adami and Hu (transport
    velocity formulation).

    """
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(FluidSolidWallPressureBCFluidSolid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

    def post_loop(self, d_idx, d_wij, d_p):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]


class MomentumEquationViscosityNoSlip(Equation):
    def __init__(self, dest, sources, nu):
        r"""
        Parameters
        ----------
        nu : float
            viscosity of the fluid.
        """

        self.nu = nu
        super(MomentumEquationViscosityNoSlip, self).__init__(dest, sources)

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


class StateEquation(Equation):
    def __init__(self, dest, sources, p0, rho0, b=1.0):
        self.b = b
        self.p0 = p0
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho):
        d_p[d_idx] = self.p0 * (d_rho[d_idx] / self.rho0 - self.b)


class FluidMomentumEquationViscosityOnFluid(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidMomentumEquationViscosityOnFluid, self).__init__(
            dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_au,
             d_av, d_aw, VIJ, R2IJ, EPS, DWIJ, XIJ):
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 4 * (etai * etaj)/(etai + etaj)

        xdotdij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]

        tmp = s_m[s_idx]/(d_rho[d_idx] * s_rho[s_idx])
        fac = tmp * etaij * xdotdij/(R2IJ + EPS)

        d_au[d_idx] += fac * VIJ[0]
        d_av[d_idx] += fac * VIJ[1]
        d_aw[d_idx] += fac * VIJ[2]


class FluidMomentumEquationViscosityOnFluidSolid(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidMomentumEquationViscosityOnFluidSolid, self).__init__(
            dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m,
             d_u, d_v, d_w, s_ugns, s_vgns, s_wgns,
             d_au, d_av, d_aw, R2IJ, EPS, DWIJ, XIJ):
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


class FluidMomentumEquationPressureGradient(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_au, d_av, d_aw,
             DWIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx] / rhoi2 + s_p[s_idx] / rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class FluidMomentumEquationPressureGradientRogers(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_au, d_av, d_aw,
             DWIJ):
        pij = (d_p[d_idx] + s_p[s_idx]) / (d_rho[d_idx] * s_rho[s_idx])

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class SetDensityFromPressure(Equation):
    def __init__(self, dest, sources, rho0, p0, b=1.):
        self.rho0 = rho0
        self.p0 = p0
        self.b = b
        super(SetDensityFromPressure, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_rho, d_p, d_m):
        # update the density from the pressure Eq. (28)
        d_rho[d_idx] = self.rho0 * (d_p[d_idx] / self.p0 + self.b)
        # d_vol[d_idx] = d_m[d_idx] / d_rho[d_idx]


class FluidsWCSPHStep(IntegratorStep):
    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

    def stage2(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_rho,
               d_arho, d_m, d_p, d_ap, dt):
        d_rho[d_idx] += dt * d_arho[d_idx]
        d_p[d_idx] += dt * d_ap[d_idx]

        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt, d_uhat,
               d_vhat, d_what, d_auhat, d_avhat, d_awhat):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]


class FluidsWCSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, c0, nu, rho0, u_max, mach_no,
                 pb=0.0, gx=0.0, gy=0.0, gz=0.0, tdamp=0.0, eps=0.0, h=0.0,
                 kernel_factor=3, edac_alpha=0.5, alpha=0.0, pst="sun2019",
                 edac=False, kernel_choice="1"):
        self.c0 = c0
        self.nu = nu
        self.rho0 = rho0
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tdamp = tdamp
        self.dim = dim
        self.eps = eps
        self.fluids = fluids
        self.solids = solids
        self.pb = pb
        self.solver = None
        self.h = h
        self.kernel_factor = kernel_factor
        self.edac_alpha = edac_alpha
        self.alpha = alpha
        self.pst = pst
        self.cont_vc_bc = False
        self.clamp_p = True

        # attributes for P Sun 2019 PST technique
        self.u_max = u_max
        self.mach_no = mach_no

        # attributes for IPST technique
        self.edac = edac

        # TODO: kernel_fac will change with kernel. This should change
        self.kernel_choice = kernel_choice
        self.kernel = QuinticSpline
        self.kernel_factor = 2

        # edac
        self.surf_p_zero = True

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--alpha", action="store", type=float, dest="alpha",
                           default=None,
                           help="Alpha for the artificial viscosity.")

        group.add_argument("--edac-alpha", action="store", type=float,
                           dest="edac_alpha", default=None,
                           help="Alpha for the EDAC scheme viscosity.")

        add_bool_argument(group, 'edac', dest='edac', default=True,
                          help='Use pressure evolution by EDAC')

        add_bool_argument(group, 'cont-vc-bc', dest='cont_vc_bc',
                          default=False,
                          help='Use velocity BC in continuity equation')

        add_bool_argument(group, 'clamp-p', dest='clamp_p', default=True,
                          help='Clamp pressure')

        choices = ['sun2019', 'ipst', 'tvf', 'None']
        group.add_argument(
            "--pst", action="store", dest='pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use (one of %s)." % choices)

    def consume_user_options(self, options):
        vars = [
            'alpha', 'edac_alpha', 'pst', 'edac', 'cont_vc_bc', 'clamp_p'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def attributes_changed(self):
        if self.h is not None and self.c0 is not None:
            self.art_nu = self.edac_alpha * self.h * self.c0 / 8

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.wc.gtvf import GTVFIntegrator
        kernel = self.kernel(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = FluidsWCSPHStep
        cls = (integrator_cls
               if integrator_cls is not None else GTVFIntegrator)

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()
        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        nu_edac = self._get_edac_nu()
        all = self.fluids + self.solids
        stage1 = []

        eqs = []
        if self.cont_vc_bc == True:
            if len(self.solids) > 0:
                for solid in self.solids:
                    eqs.append(
                        FluidSetWallVelocityUFreeSlipAndNoSlip(
                            dest=solid, sources=self.fluids))

                stage1.append(Group(equations=eqs, real=False))

        eqs = []
        for fluid in self.fluids:
            eqs.append(FluidContinuityEquationWCSPHOnFluid(dest=fluid,
                                                           sources=self.fluids), )

            if self.edac is True:
                eqs.append(
                    FluidEDACEquationWCSPHOnFluid(dest=fluid, sources=self.fluids,
                                                  nu=nu_edac), )

        if len(self.solids) > 0:
            for fluid in self.fluids:
                eqs.append(
                    FluidContinuityEquationWCSPHOnFluidSolid(dest=fluid,
                                                             sources=self.solids), )

            if self.edac is True:
                eqs.append(
                    FluidEDACEquationWCSPHOnFluidSolid(dest=fluid,
                                                       sources=self.solids,
                                                       nu=nu_edac), )

        stage1.append(Group(equations=eqs, real=False))

        # =========================#
        # stage 2 equations start
        # =========================#

        stage2 = []

        if self.edac is False:
            tmp = []
            for fluid in self.fluids:
                tmp.append(
                    StateEquation(dest=fluid, sources=None, p0=self.pb,
                                  rho0=self.rho0))

            stage2.append(Group(equations=tmp, real=False))

        if len(self.solids) > 0:
            eqs = []
            for solid in self.solids:
                eqs.append(
                    FluidSetWallVelocityUFreeSlipAndNoSlip(dest=solid,
                                                           sources=self.fluids))

                eqs.append(
                    FluidSolidWallPressureBCFluidSolid(
                        dest=solid, sources=self.fluids,
                        gx=self.gx, gy=self.gy, gz=self.gz))

                if self.clamp_p == True:
                    eqs.append(
                        FluidClampWallPressureFluidSolid(dest=solid, sources=None))

            stage2.append(Group(equations=eqs, real=False))

        eqs = []
        for fluid in self.fluids:
            if self.alpha > 0.:
                eqs.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=all, c0=self.c0,
                        alpha=self.alpha
                    )
                )

            if self.nu > 0:
                eqs.append(
                    MomentumEquationViscosity(dest=fluid, sources=self.fluids,
                                              nu=self.nu))

                if len(self.solids) > 0:
                    eqs.append(
                        MomentumEquationViscosityNoSlip(
                            dest=fluid, sources=self.solids, nu=self.nu))

            eqs.append(
                FluidMomentumEquationPressureGradientRogers(dest=fluid, sources=all,
                                                            gx=self.gx, gy=self.gy,
                                                            gz=self.gz), )

        stage2.append(Group(equations=eqs, real=True))

        return MultiStageEquations([stage1, stage2])

    def setup_properties(self, particles, clean=True):
        pas = dict([(p.name, p) for p in particles])
        for fluid in self.fluids:
            pa = pas[fluid]
            props = 'u0 v0 w0 x0 y0 z0 rho0 arho ap arho p0 uhat vhat what auhat avhat awhat h_b V cs'.split(
            )
            for prop in props:
                pa.add_property(prop)

            pa.h_b[:] = pa.h[:]

            if 'c0_ref' not in pa.constants:
                pa.add_constant('c0_ref', self.c0)
            pa.add_output_arrays(['p'])
            pa.cs[:] = self.c0

            if 'wdeltap' not in pa.constants:
                pa.add_constant('wdeltap', -1.)

            if 'n' not in pa.constants:
                pa.add_constant('n', 0.)

            add_boundary_identification_properties(pa)

            pa.h_b[:] = pa.h

        for solid in self.solids:
            pa = pas[solid]

            add_properties(pa, 'rho', 'V', 'wij2', 'wij', 'uhat', 'vhat',
                           'what')

            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf')
            add_properties(pa, 'ugfs', 'vgfs', 'wgfs')

            add_properties(pa, 'ughatns', 'vghatns', 'wghatns')
            add_properties(pa, 'ughatfs', 'vghatfs', 'wghatfs')

            # No slip boundary conditions for viscosity force
            add_properties(pa, 'ugns', 'vgns', 'wgns')

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
