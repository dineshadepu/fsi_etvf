import numpy
import numpy as np

from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)

from solid_mech import (ComputeAuHatETVF, ComputeAuHatETVFSun2019,
                        SavePositionsIPSTBeforeMoving, AdjustPositionIPST,
                        CheckUniformityIPST, ComputeAuhatETVFIPST,
                        ResetParticlePositionsIPST, EDACEquation, setup_ipst,
                        SetHIJForInsideParticles, ContinuityEquationUhat,
                        ContinuityEquationETVFCorrection)

from pysph.sph.wc.edac import (SolidWallPressureBC)

from pysph.sph.integrator import PECIntegrator
from boundary_particles import (add_boundary_identification_properties)

from boundary_particles import (ComputeNormals, SmoothNormals,
                                IdentifyBoundaryParticleCosAngleEDAC)

from common import EDACIntegrator
from pysph.examples.solid_mech.impact import add_properties
from pysph.sph.wc.linalg import mat_vec_mult
from pysph.sph.basic_equations import (ContinuityEquation,
                                       MonaghanArtificialViscosity,
                                       VelocityGradient3D, VelocityGradient2D)
from pysph.sph.solid_mech.basic import (IsothermalEOS,
                                        HookesDeviatoricStressRate)
from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from solid_mech_common import AddGravityToStructure
from pysph.sph.wc.transport_velocity import (MomentumEquationArtificialViscosity)
from pysph.sph.wc.basic import (TaitEOSHGCorrection)


class SolidWallPressureBCFSI(Equation):
    r"""Solid wall pressure boundary condition from Adami and Hu (transport
    velocity formulation).

    """
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(SolidWallPressureBCFSI, self).__init__(dest, sources)

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

        d_p_fsi[d_idx] += s_p[s_idx] * WIJ + s_rho[s_idx] * gdotxij * WIJ

    def post_loop(self, d_idx, d_wij, d_p_fsi):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p_fsi[d_idx] /= d_wij[d_idx]


class AccelerationOnFluidDueToStructure(Equation):
    def loop(self, d_rho, s_rho_fsi, d_idx, s_idx, d_p, s_p_fsi, s_m_fsi, d_au,
             d_av, d_aw, DWIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho_fsi[s_idx] * s_rho_fsi[s_idx]

        pij = d_p[d_idx] / rhoi2 + s_p_fsi[s_idx] / rhoj2

        tmp = -s_m_fsi[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class AccelerationOnStructureDueToFluid(Equation):
    def loop(self, d_rho_fsi, s_rho, d_idx, s_idx, d_p_fsi, s_p, s_m, d_au,
             d_av, d_aw, DWIJ):
        rhoi2 = d_rho_fsi[d_idx] * d_rho_fsi[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p_fsi[d_idx] / rhoi2 + s_p[s_idx] / rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class AccelerationOnFluidDueToStructureHan(Equation):
    """
    SPH modeling of fluid-structure interaction
    https://doi.org/10.1007/s42241-018-0006-9
    """
    def loop(self, s_rho, d_idx, d_m, d_rho, s_idx, d_p, s_m, d_au, d_av, d_aw,
             DWIJ):
        V_d = d_m[d_idx] / d_rho[d_idx]
        V_s = s_m[s_idx] / s_rho[s_idx]

        V2_d = V_d * V_d
        V2_s = V_s * V_s

        tmp = - (V2_d + V2_s) * d_p[d_idx] / d_m[d_idx]

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class AccelerationOnStructureDueToFluidHan(Equation):
    """
    SPH modeling of fluid-structure interaction
    https://doi.org/10.1007/s42241-018-0006-9
    """
    def loop(self, s_rho, d_idx, d_m, d_rho, s_idx, s_p, s_m, d_au, d_av, d_aw,
             DWIJ):
        V_d = d_m[d_idx] / d_rho[d_idx]
        V_s = s_m[s_idx] / s_rho[s_idx]

        V2_d = V_d * V_d
        V2_s = V_s * V_s

        tmp = - (V2_d + V2_s) * s_p[s_idx] / d_m[d_idx]

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class ContinuitySolidEquationGTVFFSI(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m_fsi, d_rho, s_rho_fsi, d_uhat, d_vhat,
             d_what, s_ughatfs, s_vghatfs, s_wghatfs, s_ughatns, s_vghatns,
             s_wghatns, d_arho, DWIJ):
        uhatij = d_uhat[d_idx] - s_ughatfs[s_idx]
        vhatij = d_vhat[d_idx] - s_vghatfs[s_idx]
        whatij = d_what[d_idx] - s_wghatfs[s_idx]

        # uhatij = d_uhat[d_idx] - s_ughatns[s_idx]
        # vhatij = d_vhat[d_idx] - s_vghatns[s_idx]
        # whatij = d_what[d_idx] - s_wghatns[s_idx]

        udotdij = DWIJ[0] * uhatij + DWIJ[1] * vhatij + DWIJ[2] * whatij
        fac = d_rho[d_idx] * s_m_fsi[s_idx] / s_rho_fsi[s_idx]
        d_arho[d_idx] += fac * udotdij


class ContinuitySolidEquationETVFCorrectionFSI(Equation):
    """
    This is the additional term arriving in the new ETVF continuity equation
    """
    def loop(self, d_idx, d_arho, d_rho, d_u, d_v, d_w, d_uhat, d_vhat, d_what,
             s_idx, s_rho_fsi, s_m_fsi, s_ugfs, s_vgfs, s_wgfs, s_ughatfs,
             s_vghatfs, s_wghatfs, s_ugns, s_vgns, s_wgns, s_ughatns,
             s_vghatns, s_wghatns, DWIJ):
        # by using free ship boundary conditions
        tmp0 = s_rho_fsi[s_idx] * (
            s_ughatfs[s_idx] - s_ugfs[s_idx]) - d_rho[d_idx] * (d_uhat[d_idx] -
                                                                d_u[d_idx])

        tmp1 = s_rho_fsi[s_idx] * (
            s_vghatfs[s_idx] - s_vgfs[s_idx]) - d_rho[d_idx] * (d_vhat[d_idx] -
                                                                d_v[d_idx])

        tmp2 = s_rho_fsi[s_idx] * (
            s_wghatfs[s_idx] - s_wgfs[s_idx]) - d_rho[d_idx] * (d_what[d_idx] -
                                                                d_w[d_idx])

        # # by using no ship boundary conditions
        # tmp0 = s_rho[s_idx] * (s_ughatns[s_idx] - s_ugns[s_idx]
        #                        ) - d_rho[d_idx] * (d_uhat[d_idx] - d_u[d_idx])

        # tmp1 = s_rho[s_idx] * (s_vghatns[s_idx] - s_vgns[s_idx]
        #                        ) - d_rho[d_idx] * (d_vhat[d_idx] - d_v[d_idx])

        # tmp2 = s_rho[s_idx] * (s_wghatns[s_idx] - s_wgns[s_idx]
        #                        ) - d_rho[d_idx] * (d_what[d_idx] - d_w[d_idx])

        vijdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)

        d_arho[d_idx] += s_m_fsi[s_idx] / s_rho_fsi[s_idx] * vijdotdwij


class SummationDensity(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, s_idx, d_m, WIJ):
        d_rho[d_idx] += d_m[d_idx]*WIJ


class FSIScheme(Scheme):
    def __init__(self, fluids, structures, solids, structure_solids, dim,
                 h_fluid, c0_fluid, nu_fluid, rho0_fluid, mach_no_fluid,
                 mach_no_structure, pb_fluid=0.0, gx=0.0, gy=0.0, gz=0.0,
                 artificial_vis_alpha=1.0, artificial_vis_beta=0.0, tdamp=0.0,
                 eps=0.0, kernel_factor=3, edac_alpha=0.5, alpha=0.0,
                 pst="sun2019", debug=False, edac=False, summation=False,
                 ipst_max_iterations=10, ipst_tolerance=0.2, ipst_interval=5,
                 internal_flow=False, kernel_choice="1", integrator='gtvf'):
        self.c0_fluid = c0_fluid
        self.rho0_fluid = rho0_fluid
        self.nu_fluid = nu_fluid
        self.pb_fluid = pb_fluid
        self.h_fluid = h_fluid

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tdamp = tdamp
        self.dim = dim
        self.eps = eps
        self.fluids = fluids
        self.solids = solids

        self.structures = structures
        self.structure_solids = structure_solids

        self.kernel_factor = kernel_factor
        self.edac_alpha = edac_alpha
        self.alpha = alpha
        self.pst = pst

        # attributes for P Sun 2019 PST technique
        self.mach_no_fluid = mach_no_fluid
        self.mach_no_structure = mach_no_structure

        # structure properties
        self.artificial_vis_alpha = artificial_vis_alpha
        self.artificial_vis_beta = artificial_vis_beta

        # attributes for IPST technique
        self.ipst_max_iterations = ipst_max_iterations
        self.ipst_tolerance = ipst_tolerance
        self.ipst_interval = ipst_interval
        self.debug = debug
        self.internal_flow = internal_flow
        self.edac = edac
        self.summation = summation

        # TODO: kernel_fac will change with kernel. This should change
        self.kernel_choice = kernel_choice
        self.kernel = QuinticSpline
        self.kernel_factor = 2

        self.integrator = integrator

        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--alpha", action="store", type=float, dest="alpha",
                           default=None,
                           help="Alpha for the artificial viscosity.")

        group.add_argument("--edac-alpha", action="store", type=float,
                           dest="edac_alpha", default=None,
                           help="Alpha for the EDAC scheme viscosity.")

        group.add_argument("--ipst-max-iterations", action="store",
                           dest="ipst_max_iterations", default=10, type=int,
                           help="Max iterations of IPST")

        group.add_argument("--ipst-interval", action="store",
                           dest="ipst_interval", default=5, type=int,
                           help="Frequency at which IPST is to be done")

        add_bool_argument(group, 'debug', dest='debug', default=False,
                          help='Check if the IPST converged')

        add_bool_argument(group, 'edac', dest='edac', default=False,
                          help='Use pressure evolution by EDAC')

        add_bool_argument(group, 'summation', dest='summation', default=False,
                          help='Use summation density to compute the density')

        choices = ['sun2019', 'ipst', 'tvf', 'None']
        group.add_argument(
            "--pst", action="store", dest='pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use (one of %s)." % choices)

        choices = ['pec', 'gtvf']
        group.add_argument(
            "--integrator", action="store", dest="integrator", default='gtvf',
            choices=choices,
            help="Specify integrator to use (one of %s)." % choices)

        add_bool_argument(group, 'internal-flow', dest='internal_flow',
                          default=False,
                          help='Check if it is an internal flow')

        group.add_argument("--ipst-tolerance", action="store", type=float,
                           dest="ipst_tolerance", default=None,
                           help="Tolerance limit of IPST")

        add_bool_argument(
            group, 'surf-p-zero', dest='surf_p_zero', default=False,
            help='Make the surface pressure and acceleration to be zero')

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

    def consume_user_options(self, options):
        vars = [
            'alpha', 'edac_alpha', 'pst', 'debug', 'ipst_max_iterations',
            'integrator', 'internal_flow', 'ipst_tolerance', 'ipst_interval',
            'edac', 'summation', 'kernel_choice'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def attributes_changed(self):
        if self.pb_fluid is not None:
            self.use_tvf = abs(self.pb_fluid) > 1e-14
        if self.h_fluid is not None and self.c0_fluid is not None:
            self.art_nu = self.edac_alpha * self.h_fluid * self.c0_fluid / 8

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

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.wc.gtvf import GTVFIntegrator
        from solid_mech import (SolidMechStepEDAC, SolidMechStep)
        from fluids import (EDACGTVFStep)
        kernel = self.kernel(dim=self.dim)
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
            step_cls = SolidMechStepEDAC
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
        return self._get_gtvf_equations()

    def _get_gtvf_equations(self):
        from pysph.sph.wc.gtvf import (ContinuityEquationGTVF,
                                       MomentumEquationArtificialStress,
                                       MomentumEquationViscosity)
        # from pysph.sph.iisph import (SummationDensity)
        from fluids import (
            SetWallVelocityFreeSlipAndNoSlip,
            SetWallVelocityUhatFreeSlipAndNoSlip, ContinuitySolidEquationGTVF,
            ContinuitySolidEquationETVFCorrection, EDACSolidEquation,
            MakeSurfaceParticlesPressureApZeroEDACFluids, StateEquation,
            MomentumEquationPressureGradient, MomentumEquationTVFDivergence,
            MomentumEquationViscosityNoSlip, MakeAuhatZero,
            CheckUniformityIPSTFluidInternalFlow, setup_ipst_fluids)

        from solid_mech import (AdamiBoundaryConditionExtrapolateNoSlip,
                                MomentumEquationSolids,
                                ComputeAuHatETVFSun2019Solid)
        nu_edac = self._get_edac_nu()
        all = self.fluids + self.solids
        stage1 = []

        # =========================#
        # fluid equations
        # =========================#
        if self.summation is not True:
            eqs = []
            if len(self.solids) > 0:
                for solid in self.solids:
                    eqs.append(
                        SetWallVelocityFreeSlipAndNoSlip(dest=solid,
                                                         sources=self.fluids), )

                stage1.append(Group(equations=eqs, real=False))

            if len(self.solids) > 0:
                eqs = []
                for solid in self.solids:
                    eqs.append(
                        SetWallVelocityUhatFreeSlipAndNoSlip(
                            dest=solid, sources=self.fluids))

                stage1.append(Group(equations=eqs, real=False))

            # for the elastic structure
            eqs = []
            if len(self.structures) > 0:
                for structure in self.structures:
                    eqs.append(
                        SetWallVelocityFreeSlipAndNoSlip(dest=structure,
                                                         sources=self.fluids), )

                stage1.append(Group(equations=eqs, real=False))

            if len(self.structures) > 0:
                eqs = []
                for structure in self.structures:
                    eqs.append(
                        SetWallVelocityUhatFreeSlipAndNoSlip(
                            dest=structure, sources=self.fluids))

                stage1.append(Group(equations=eqs, real=False))
            # for the elastic structure ends

            # for the elastic structure solid support
            eqs = []
            if len(self.structure_solids) > 0:
                for solid in self.structure_solids:
                    eqs.append(
                        SetWallVelocityFreeSlipAndNoSlip(dest=solid,
                                                         sources=self.fluids), )

                stage1.append(Group(equations=eqs, real=False))

            if len(self.structure_solids) > 0:
                eqs = []
                for solid in self.structure_solids:
                    eqs.append(
                        SetWallVelocityUhatFreeSlipAndNoSlip(
                            dest=solid, sources=self.fluids))

                stage1.append(Group(equations=eqs, real=False))
            # for the elastic structure solid support ends

            eqs = []
            for fluid in self.fluids:
                eqs.append(ContinuityEquationGTVF(dest=fluid,
                                                sources=self.fluids), )
                eqs.append(
                    ContinuityEquationETVFCorrection(dest=fluid,
                                                    sources=self.fluids), )
                # if self.edac is True:
                #     eqs.append(
                #         EDACEquation(dest=fluid, sources=self.fluids,
                #                      nu=nu_edac), )

            if len(self.solids) > 0:
                for fluid in self.fluids:
                    eqs.append(
                        ContinuitySolidEquationGTVF(dest=fluid,
                                                    sources=self.solids), )
                    eqs.append(
                        ContinuitySolidEquationETVFCorrection(
                            dest=fluid, sources=self.solids), )

                # if self.edac is True:
                #     eqs.append(
                #         EDACSolidEquation(dest=fluid, sources=self.solids,
                #                           nu=nu_edac), )

            # TODO: Should we use direct density or the density of the fluid
            if len(self.structures) > 0:
                for fluid in self.fluids:
                    eqs.append(
                        ContinuitySolidEquationGTVFFSI(dest=fluid,
                                                    sources=self.structures), )
                    eqs.append(
                        ContinuitySolidEquationETVFCorrectionFSI(
                            dest=fluid, sources=self.structures), )

            if len(self.structure_solids) > 0:
                for fluid in self.fluids:
                    eqs.append(
                        ContinuitySolidEquationGTVFFSI(dest=fluid,
                                                       sources=self.structure_solids))
                    eqs.append(
                        ContinuitySolidEquationETVFCorrectionFSI(
                            dest=fluid, sources=self.structure_solids))

            stage1.append(Group(equations=eqs, real=False))

        else:
            eqs = []
            for fluid in self.fluids:
                eqs.append(SummationDensity(dest=fluid,
                                            sources=self.fluids
                                            + self.solids
                                            + self.structures
                                            + self.structure_solids),)
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
                g1.append(ContinuityEquationUhat(dest=structure, sources=all))
                g1.append(
                    ContinuityEquationETVFCorrection(dest=structure, sources=all))

                if self.dim == 2:
                    g1.append(VelocityGradient2D(dest=structure, sources=all))
                elif self.dim == 3:
                    g1.append(VelocityGradient3D(dest=structure, sources=all))

            stage1.append(Group(equations=g1))

            g2 = []
            for structure in self.structures:
                g2.append(HookesDeviatoricStressRate(dest=structure, sources=None))

            stage1.append(Group(equations=g2))

            # # edac pressure evolution equation
            # if self.edac is True:
            #     gtmp = []
            #     for solid in self.solids:
            #         gtmp.append(
            #             EDACEquation(dest=solid, sources=all, nu=self.edac_nu))

            #     stage1.append(Group(gtmp))

        # =========================#
        # structure equations ends
        # =========================#

        # =========================#
        # stage 2 equations start
        # =========================#

        stage2 = []

        # if self.edac is False:
        tmp = []
        for fluid in self.fluids:
            if self.summation is False:
                tmp.append(
                    # TODO: THESE PRESSURE VALUES WILL BE DIFFERENT FOR DIFFERENT PHASES
                    StateEquation(dest=fluid, sources=None, p0=self.pb_fluid,
                                  rho0=self.rho0_fluid))
            else:
                tmp.append(
                    TaitEOSHGCorrection(dest=fluid, sources=None,
                                        rho0=self.rho0_fluid,
                                        c0=self.c0_fluid, gamma=7.0),)

        stage2.append(Group(equations=tmp, real=False))

        if len(self.solids) > 0:
            eqs = []
            for solid in self.solids:
                eqs.append(
                    SetWallVelocityFreeSlipAndNoSlip(dest=solid,
                                                     sources=self.fluids))

                eqs.append(
                    SolidWallPressureBC(dest=solid, sources=self.fluids,
                                        gx=self.gx, gy=self.gy, gz=self.gz))
            stage2.append(Group(equations=eqs, real=False))

        # # FSI coupling equations, set the pressure
        # if len(self.structures) > 0:
        #     eqs = []
        #     for structure in self.structures:
        #         eqs.append(
        #             SetWallVelocityFreeSlipAndNoSlip(dest=structure,
        #                                              sources=self.fluids))

        #         eqs.append(
        #             SolidWallPressureBCFSI(dest=structure, sources=self.fluids,
        #                                    gx=self.gx, gy=self.gy, gz=self.gz))

        #     stage2.append(Group(equations=eqs, real=False))

        # if len(self.structure_solids) > 0:
        #     eqs = []
        #     for structure_solid in self.structure_solids:
        #         eqs.append(
        #             SetWallVelocityFreeSlipAndNoSlip(dest=structure_solid,
        #                                              sources=self.fluids))

        #         eqs.append(
        #             SolidWallPressureBCFSI(dest=structure_solid,
        #                                    sources=self.fluids, gx=self.gx,
        #                                    gy=self.gy, gz=self.gz))
        #     stage2.append(Group(equations=eqs, real=False))
        # FSI coupling equations, set the pressure

        # fluid momentum equations
        eqs = []
        if self.internal_flow is not True:
            for fluid in self.fluids:
                eqs.append(
                    SetHIJForInsideParticles(dest=fluid, sources=[fluid],
                                             kernel_factor=self.kernel_factor))
            stage2.append(Group(eqs))

        eqs = []
        for fluid in self.fluids:
            # FIXME: Change alpha to variable
            if self.alpha > 0.:
                eqs.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid,
                        sources=self.fluids+self.solids,
                        c0=self.c0_fluid,
                        alpha=self.alpha
                    )
                )

            eqs.append(
                MomentumEquationPressureGradient(
                    dest=fluid, sources=self.fluids + self.solids, gx=self.gx,
                    gy=self.gy, gz=self.gz), )

            eqs.append(
                MomentumEquationArtificialStress(dest=fluid,
                                                 sources=self.fluids,
                                                 dim=self.dim))
            eqs.append(
                MomentumEquationTVFDivergence(dest=fluid, sources=self.fluids))

            eqs.append(
                ComputeAuHatETVFSun2019(dest=fluid,
                                        sources=self.fluids + self.solids,
                                        mach_no=self.mach_no_fluid))
            # if self.nu_fluid > 0:
            #     eqs.append(
            #         MomentumEquationViscosity(dest=fluid, sources=self.fluids,
            #                                   nu=self.nu_fluid))

            #     if len(self.solids) > 0:
            #         eqs.append(
            #             MomentumEquationViscosityNoSlip(
            #                 dest=fluid, sources=self.solids, nu=self.nu_fluid))
            if len(self.structure_solids + self.structures) > 0.:
                eqs.append(
                    AccelerationOnFluidDueToStructureHan(
                        dest=fluid,
                        sources=self.structures + self.structure_solids), )

        stage2.append(Group(equations=eqs, real=True))
        # fluid momentum equations ends

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

        # if self.edac is False:
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
                if self.artificial_vis_alpha > 0. or self.artificial_vis_beta > 0.:
                    g4.append(
                        MonaghanArtificialViscosity(
                            dest=structure,
                            sources=self.structures + self.structure_solids,
                            alpha=self.artificial_vis_alpha,
                            beta=self.artificial_vis_beta))

                g4.append(
                    MomentumEquationSolids(
                        dest=structure,
                        sources=self.structures + self.structure_solids))

                g4.append(
                    ComputeAuHatETVFSun2019Solid(
                        dest=structure,
                        sources=[structure] + self.structure_solids,
                        mach_no=self.mach_no_structure))

                g4.append(
                    AccelerationOnStructureDueToFluidHan(
                        dest=structure,
                        sources=self.fluids), )

            stage2.append(Group(g4))

            # Add gravity
            g5 = []
            for structure in self.structures:
                g5.append(
                    AddGravityToStructure(dest=structure, sources=None, gx=self.gx,
                                          gy=self.gy, gz=self.gz))

            stage2.append(Group(g5))

        return MultiStageEquations([stage1, stage2])

    def setup_properties(self, particles, clean=True):
        pas = dict([(p.name, p) for p in particles])
        for fluid in self.fluids:
            pa = pas[fluid]
            props = 'u0 v0 w0 x0 y0 z0 rho0 arho ap arho p0 uhat vhat what auhat avhat awhat h_b V'.split(
            )
            for prop in props:
                pa.add_property(prop)

            add_properties(pa, 'div_r')

            pa.h_b[:] = pa.h[:]

            if 'c0_ref' not in pa.constants:
                pa.add_constant('c0_ref', self.c0_fluid)
            pa.add_output_arrays(['p'])

            if 'wdeltap' not in pa.constants:
                # pa.add_constant('wdeltap', -1.)

                # this will change
                kernel = self.kernel(dim=self.dim)
                wdeltap = kernel.kernel(rij=pa.h[0], h=pa.h[0])
                pa.add_constant('wdeltap', wdeltap)

            if 'n' not in pa.constants:
                pa.add_constant('n', 4.)

            add_boundary_identification_properties(pa)

            pa.h_b[:] = pa.h

            if self.summation is True:
                pa.add_property('cs')

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

        # Add fsi props
        for structure in self.structures:
            pa = pas[structure]

            add_properties(pa, 'm_fsi', 'p_fsi', 'rho_fsi', 'V', 'wij2', 'wij',
                           'uhat', 'vhat', 'what')
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
            kernel = self.kernel(dim=self.dim)
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
            if self.kernel_choice == "4":
                pa.h[:] = pa.h[:] / self.hdx * 2.

            pa.add_output_arrays(['p'])

        for boundary in self.structure_solids:
            pa = pas[boundary]
            pa.add_property('wij')

            # for adami boundary condition

            add_properties(pa, 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 's00',
                           's01', 's02', 's11', 's12', 's22', 'cs', 'uhat',
                           'vhat', 'what')

            cs = np.ones_like(pa.x) * get_speed_of_sound(
                pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.cs[:] = cs[:]

            pa.add_property('ughat')
            pa.add_property('vghat')
            pa.add_property('wghat')

            if self.kernel_choice == "4":
                pa.h[:] = pa.h[:] / self.hdx * 2.

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
