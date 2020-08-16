import numpy as np

from pysph.sph.scheme import Scheme
from pysph.sph.equation import Equation
from pysph.sph.scheme import add_bool_argument
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from pysph.base.kernels import QuinticSpline


class AdamiBoundaryConditionExtrapolate(Equation):
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
        d_s01[d_idx] += s_s01[s_idx] * WIJ
        d_s02[d_idx] += s_s02[s_idx] * WIJ
        d_s11[d_idx] += s_s11[s_idx] * WIJ
        d_s12[d_idx] += s_s12[s_idx] * WIJ
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


class ApplyBodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(ApplyBodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] += self.gx
        d_av[d_idx] += self.gy
        d_aw[d_idx] += self.gz


class AkinciFSICoupling(Equation):
    def __init__(self, dest, sources, fluid_rho=1000):
        super(AkinciFSICoupling, self).__init__(dest, sources)
        self.fluid_rho = fluid_rho

    def loop(self, d_idx, d_m, d_rho, d_au, d_av, d_aw, s_idx, d_vol, DWIJ,
             s_m, s_p, s_rho):

        psi = d_vol[s_idx] * self.fluid_rho

        _t1 = 2 * s_p[d_idx] / (s_rho[d_idx]**2)

        d_au[d_idx] += -psi * _t1 * DWIJ[0]
        d_av[d_idx] += -psi * _t1 * DWIJ[1]
        d_aw[d_idx] += -psi * _t1 * DWIJ[2]


class CDSBTRepulsiveForceFSI(Equation):
    """
    Equation 27

    Numerical simulation of hydro-elastic problems with smoothed particle hydro-
    dynamics method.
    """
    def __init__(self, dest, sources, c0):
        super(CDSBTRepulsiveForceFSI, self).__init__(dest, sources)
        self.c0 = c0

    def loop(self, d_idx, d_m, d_rho, d_au, d_av, d_aw, s_idx, d_vol,
             d_spacing0, DWIJ, s_m, s_p, s_rho, RIJ, HIJ, R2IJ, XIJ):
        # equation 28
        eta = RIJ / (0.75 * HIJ)

        # equation 29
        khi = 1. - (RIJ / d_spacing0[0])

        # equation 30
        tmp = 2. / 3.
        if eta > 0. and eta < tmp:
            f_eta = tmp
        elif eta > tmp and eta < 1.:
            f_eta = 2. * eta - 1.5 * eta * eta
        elif eta > 1. and eta < 2.:
            f_eta = 0.5 * (2. - eta) * (2. - eta)
        else:
            f_eta = 0.

        tmp = 0.01 * self.c0 * self.c0 * khi * f_eta / (R2IJ * d_m[d_idx])
        d_au[d_idx] -= tmp * XIJ[0]
        d_av[d_idx] -= tmp * XIJ[1]
        d_aw[d_idx] -= tmp * XIJ[2]


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


class SolidMechStep(IntegratorStep):
    """Predictor corrector Integrator for solid mechanics problems"""
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
                   d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_s00, d_s01, d_s02,
                   d_s11, d_s12, d_s22, d_s000, d_s010, d_s020, d_s110, d_s120,
                   d_s220):
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

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw, d_arho, d_s00,
               d_s01, d_s02, d_s11, d_s12, d_s22, d_s000, d_s010, d_s020,
               d_s110, d_s120, d_s220, d_as00, d_as01, d_as02, d_as11, d_as12,
               d_as22, dt):
        dtb2 = 0.5 * dt

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

        # update deviatoric stress components
        d_s00[d_idx] = d_s000[d_idx] + dtb2 * d_as00[d_idx]
        d_s01[d_idx] = d_s010[d_idx] + dtb2 * d_as01[d_idx]
        d_s02[d_idx] = d_s020[d_idx] + dtb2 * d_as02[d_idx]
        d_s11[d_idx] = d_s110[d_idx] + dtb2 * d_as11[d_idx]
        d_s12[d_idx] = d_s120[d_idx] + dtb2 * d_as12[d_idx]
        d_s22[d_idx] = d_s220[d_idx] + dtb2 * d_as22[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw, d_arho, d_s00,
               d_s01, d_s02, d_s11, d_s12, d_s22, d_s000, d_s010, d_s020,
               d_s110, d_s120, d_s220, d_as00, d_as01, d_as02, d_as11, d_as12,
               d_as22, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]

        # update deviatoric stress components
        d_s00[d_idx] = d_s000[d_idx] + dt * d_as00[d_idx]
        d_s01[d_idx] = d_s010[d_idx] + dt * d_as01[d_idx]
        d_s02[d_idx] = d_s020[d_idx] + dt * d_as02[d_idx]
        d_s11[d_idx] = d_s110[d_idx] + dt * d_as11[d_idx]
        d_s12[d_idx] = d_s120[d_idx] + dt * d_as12[d_idx]
        d_s22[d_idx] = d_s220[d_idx] + dt * d_as22[d_idx]


class WCSPHStep(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
                   d_w0, d_u, d_v, d_w, d_rho0, d_rho):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw, d_arho, dt):
        dtb2 = 0.5 * dt

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0,
               d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw, d_arho, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]


class FSIScheme(Scheme):
    def __init__(self,
                 fluids,
                 solids,
                 solid_supports,
                 boundaries,
                 dim,
                 rho0,
                 c0,
                 p0,
                 h0,
                 hdx,
                 gamma=7.0,
                 gx=0.0,
                 gy=0.0,
                 gz=0.0,
                 alpha=0.1,
                 beta=0.0,
                 nu=0.0,
                 tensile_correction=False,
                 hg_correction=False,
                 artificial_stress_eps=0.3):
        self.fluids = fluids
        self.boundaries = boundaries
        self.solids = solids
        self.solid_supports = solid_supports
        self.solver = None
        self.rho0 = rho0
        self.c0 = c0
        self.p0 = p0
        self.gamma = gamma
        self.dim = dim
        self.h0 = h0
        self.hdx = hdx
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.tensile_correction = tensile_correction
        self.hg_correction = hg_correction
        self.artificial_stress_eps = artificial_stress_eps

    def add_user_options(self, group):
        group.add_argument("--alpha",
                           action="store",
                           type=float,
                           dest="alpha",
                           default=None,
                           help="Alpha for the artificial viscosity.")
        group.add_argument("--beta",
                           action="store",
                           type=float,
                           dest="beta",
                           default=None,
                           help="Beta for the artificial viscosity.")
        group.add_argument(
            "--artificial-stress-eps",
            action="store",
            type=float,
            dest="artificial_stress_eps",
            default=0.3,
            help=
            "Used in Monaghan Artificial stress to reduce tensile instability")
        group.add_argument("--gamma",
                           action="store",
                           type=float,
                           dest="gamma",
                           default=None,
                           help="Gamma for the state equation.")
        add_bool_argument(group,
                          'tensile-correction',
                          dest='tensile_correction',
                          help="Use tensile instability correction.",
                          default=None)
        add_bool_argument(group,
                          "hg-correction",
                          dest="hg_correction",
                          help="Use the Hughes Graham correction.",
                          default=None)

    def consume_user_options(self, options):
        vars = [
            'gamma', 'tensile_correction', 'hg_correction', 'alpha', 'beta',
            'artificial_stress_eps'
        ]

        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import EPECIntegrator

        cls = integrator_cls if integrator_cls is not None else EPECIntegrator
        step_cls = WCSPHStep
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = step_cls()

        step_cls = SolidMechStep
        for name in self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim,
                             integrator=integrator,
                             kernel=kernel,
                             **kw)

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.wc.basic import (MomentumEquation, TaitEOS)
        from pysph.sph.basic_equations import (ContinuityEquation)
        from pysph.sph.wc.viscosity import (LaminarViscosity)
        from pysph.sph.wc.transport_velocity import (SetWallVelocity,
                                                     ContinuitySolid)
        from pysph.sph.wc.edac import (SolidWallPressureBC)

        # elastic solid equations
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import (ContinuityEquation,
                                               MonaghanArtificialViscosity,
                                               XSPHCorrection,
                                               VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (IsothermalEOS,
                                                MomentumEquationWithStress,
                                                HookesDeviatoricStressRate,
                                                MonaghanArtificialStress)

        equations = []
        g1 = []
        all = self.fluids + self.boundaries + self.solids + self.solid_supports
        boundaries = self.boundaries + self.solids + self.solid_supports

        if len(self.fluids) > 0:
            for name in self.fluids:
                g1.append(
                    TaitEOS(dest=name,
                            sources=None,
                            rho0=self.rho0,
                            c0=self.c0,
                            gamma=self.gamma))

            for name in boundaries:
                g1.append(SetWallVelocity(dest=name, sources=self.fluids))

            if len(g1) > 0:
                equations.append(Group(equations=g1, real=False))

            tmp = []
            for solid in boundaries:
                tmp.append(
                    SolidWallPressureBC(dest=solid,
                                        sources=self.fluids,
                                        gx=self.gx,
                                        gy=self.gy,
                                        gz=self.gz))

            if len(tmp) > 0:
                equations.append(Group(equations=tmp, real=False))

            g2 = []
            for name in self.fluids:
                g2.append(ContinuityEquation(dest=name, sources=self.fluids))

                if len(boundaries) > 0:
                    g2.append(ContinuitySolid(dest=name, sources=boundaries))

                # This is required since MomentumEquation (ME) adds artificial
                # viscosity (AV), so make alpha 0.0 for ME and enable delta sph AV.
                alpha = self.alpha
                g2.append(
                    MomentumEquation(
                        dest=name,
                        sources=all,
                        c0=self.c0,
                        alpha=alpha,
                        beta=self.beta,
                        gx=self.gx,
                        gy=self.gy,
                        gz=self.gz,
                        tensile_correction=self.tensile_correction))

                if abs(self.nu) > 1e-14:
                    eq = LaminarViscosity(dest=name, sources=all, nu=self.nu)
                    g2.insert(-1, eq)
            equations.append(Group(equations=g2))

        ###########################
        # Elastic solid equations #
        ###########################
        # FIXME: These equations can be combined with the fluid equations and
        # made in one simple group. But there is a clash with
        # SolidWallPressureBC as it is overriding the pressure. So I have to
        # move the equations else where. Some thing has to be done to fix it.
        g1 = []
        all = self.solids + self.solid_supports
        for solid in self.solids:
            g1.append(
                # p
                IsothermalEOS(solid, sources=None))
            g1.append(
                # vi,j : requires properties v00, v01, v10, v11
                VelocityGradient2D(dest=solid, sources=all))
            g1.append(
                # rij : requires properties r00, r01, r02, r11, r12, r22,
                #                           s00, s01, s02, s11, s12, s22
                MonaghanArtificialStress(dest=solid,
                                         sources=None,
                                         eps=self.artificial_stress_eps))

        equations.append(Group(equations=g1))

        # -------------------
        # boundary conditions
        # -------------------
        tmp = []
        for boundary in self.solid_supports:
            tmp.append(
                AdamiBoundaryConditionExtrapolate(dest=boundary,
                                                  sources=self.solids))
        if len(tmp) > 0:
            equations.append(Group(tmp))

        g2 = []
        for solid in self.solids:
            g2.append(ContinuityEquation(dest=solid,
                                         sources=all + self.fluids))
            g2.append(
                # au, av
                MomentumEquationSolids(dest=solid, sources=all))

            g2.append(
                MonaghanArtificialStressCorrection(dest=solid, sources=all))
            g2.append(
                # au, av
                MonaghanArtificialViscosity(dest=solid,
                                            sources=all,
                                            alpha=1.0,
                                            beta=self.beta))
            g2.append(
                # a_s00, a_s01, a_s11
                HookesDeviatoricStressRate(dest=solid, sources=None))

            # g2.append(
            #     # au, av
            #     AkinciFSICoupling(dest=solid,
            #                       sources=self.fluids,
            #                       fluid_rho=self.rho0))
            g2.append(
                # au, av, aw
                CDSBTRepulsiveForceFSI(dest=solid,
                                       sources=self.fluids,
                                       c0=self.c0))

            # g2.append(
            #     # au, av
            #     ApplyBodyForce(dest=solid,
            #                    sources=None,
            #                    gx=self.gx,
            #                    gy=self.gy,
            #                    gz=self.gz))

        equations.append(Group(g2))
        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties
        pas = dict([(p.name, p) for p in particles])
        for name in self.fluids:
            pa = pas[name]

            add_properties(pa, 'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'arho',
                           'rho0', 'p0', 'cs')

            add_properties(pa, 'dt_force', 'dt_cfl')

            if 'c0_ref' not in pa.constants:
                pa.add_constant('c0_ref', self.c0)

            pa.cs[:] = self.c0
            pa.add_output_arrays(['p'])

        for name in self.boundaries:
            pa = pas[name]

            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf', 'wij')

            add_properties(pa, 'cs')
            pa.cs[:] = self.c0

            # No slip boundary conditions for viscosity force
            # add_properties(pa, 'ugns', 'vgns', 'wgns')

        for name in self.solids:
            pa = pas[name]

            add_properties(pa, 'cs', 'e', 'v00', 'v01', 'v02', 'v10', 'v11',
                           'v12', 'v20', 'v21', 'v22', 'r00', 'r01', 'r02',
                           'r11', 'r12', 'r22', 's00', 's01', 's02', 's11',
                           's12', 's22', 'as00', 'as01', 'as02', 'as11',
                           'as12', 'as22', 's000', 's010', 's020', 's110',
                           's120', 's220', 'arho', 'au', 'av', 'aw', 'ax',
                           'ay', 'az', 'ae', 'rho0', 'u0', 'v0', 'w0', 'x0',
                           'y0', 'z0', 'e0')

            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf', 'wij')

            # this will change
            kernel = QuinticSpline(dim=2)
            wdeltap = kernel.kernel(rij=pa.spacing0[0], h=pa.h[0])
            pa.add_constant('wdeltap', wdeltap)
            # No slip boundary conditions for viscosity force
            # add_properties(pa, 'ugns', 'vgns', 'wgns')

            # set the shear modulus G
            G = get_shear_modulus(pa.E[0], pa.nu[0])
            pa.add_constant('G', G)

            # set the speed of sound
            cs = np.ones_like(pa.x) * get_speed_of_sound(
                pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.cs[:] = cs[:]

            c0_ref = get_speed_of_sound(pa.E[0], pa.nu[0], pa.rho_ref[0])
            pa.add_constant('c0_ref', c0_ref)
            pa.add_output_arrays(['p'])

            # this should be fixed by changing the momentum equation
            add_properties(pa, 'dt_force', 'dt_cfl')

            # fsi coupling force
            add_properties(pa, 'vol')
            pa.vol[:] = pa.spacing0[0]**self.dim

        for name in self.solid_supports:
            pa = pas[name]
            pa.add_property('wij')

            # for adami boundary condition

            add_properties(pa, 'ug', 'vg', 'wg', 'uf', 'vf', 'wf', 's00',
                           's01', 's02', 's11', 's12', 's22', 'cs')
            add_properties(pa, 'r02', 'r11', 'r22', 'r01', 'r00', 'r12')
