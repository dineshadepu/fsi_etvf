"""Elastic Vibration of a Cantilever Beam

# Taken from


1. Nonlinear transient analysis of isotropic and composite shell structures
under dynamic loading by SPH method. Jun Lin thesis

1. Updated Smoothed Particle Hydrodynamics forSimulating Bending and Compression FailureProgress of Ice
doi:10.3390/w9110882


2. A numerical study on ice failure process and ice-ship interactions by Smoothed Particle Hydrodynamics
https://doi.org/10.1016/j.ijnaoe.2019.02.008

"""
import numpy as np
from math import cos, sin, sinh, cosh

# SPH equations
from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations, BasicCodeBlock
from pysph.sph.scheme import SchemeChooser
from pysph.sph.equation import Equation

from pysph.base.utils import get_particle_array

from solid_mech import SetHIJForInsideParticles, SolidsScheme

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.sph.scheme import add_bool_argument

from compyle.api import declare
from pysph.sph.equation import Equation, Group, BasicCodeBlock
from pysph.sph.wc.linalg import gj_solve, augmented_matrix, identity
from mako.template import Template


def shepard_corrections():
    Group.pre_comp.update(
        OLDWIJ=BasicCodeBlock(code="OLDWIJ = KERNEL(XIJ, RIJ, HIJ)", OLDWIJ=0.0),
        WIJ=BasicCodeBlock(code="WIJ = OLDWIJ*d_wij_shepard_denom_1[d_idx]", WIJ=0.0)
    )


class GradientCorrectionPreStep(Equation):
    """`L` is a symmetric matrix so we only need to
    store `dim*(dim + 1)/2` values. First we build `L` matrix,
    then compute the inverse for every particle. We use `Linv`
    property throughout.
    """
    def __init__(self, dest, sources, dim):
        # Add `1` for the zeroth moment.
        self.sim_dim = dim + 1
        self.dim = int(3 + 1)
        self.dim2 = self.dim*self.dim
        super(GradientCorrectionPreStep, self).__init__(dest, sources)

    def initialize(self, d_Linv, d_idx, d_L):
        i, j = declare('int', 2)

        for i in range(self.dim):
            for j in range(self.dim):
                d_L[d_idx*self.dim2 + i*self.dim + j] = 0.0
                d_Linv[d_idx*self.dim2 + i*self.dim + j] = 0.0
                if i == j:
                    d_Linv[d_idx*self.dim2 + i*self.dim + j] = 1.0

    def _get_helpers_(self):
        return [gj_solve, augmented_matrix, identity]

    def loop(self, d_L, d_idx, s_m, s_rho, s_idx, RIJ, XIJ, HIJ,
             SPH_KERNEL):
        V_j = s_m[s_idx] / s_rho[s_idx]

        idx, i, j = declare('int', 3)
        idx = self.dim2*d_idx

        # Append `1` to XIJ to compute zeroth moment.
        YJI = declare('matrix(4)')
        YJI[0] = 1.0
        YJI[1] = -XIJ[0]
        YJI[2] = -XIJ[1]
        YJI[3] = -XIJ[2]

        dwij = declare('matrix(3)')
        SPH_KERNEL.gradient(XIJ, RIJ, HIJ, dwij)

        wij = SPH_KERNEL.kernel(XIJ, RIJ, HIJ)
        DIJ = declare('matrix(4)')
        DIJ[0] = wij
        DIJ[1] = dwij[0]
        DIJ[2] = dwij[1]
        DIJ[3] = dwij[2]

        for i in range(self.dim):
            for j in range(self.dim):
                d_L[idx + self.dim*i + j] += V_j*DIJ[i]*YJI[j]

    def post_loop(self, d_Linv, d_idx, d_L):
        i, j, k, idx, dim, dim2 = declare('int', 6)
        dim = self.dim
        dim2 = self.dim2

        tmp = declare('matrix(16)')
        idn = declare('matrix(16)')

        identity(tmp, dim)
        identity(idn, dim)

        idx = dim2*d_idx

        for i in range(self.sim_dim):
            for j in range(self.sim_dim):
                tmp[dim*i + j] = d_L[idx + i*dim + j]

        aug_matrix = declare('matrix(32)')
        augmented_matrix(tmp, idn, dim, dim, dim, aug_matrix)

        error_code = gj_solve(aug_matrix, dim, dim, tmp)
        # If singular make d_Linv an identity matrix.
        if abs(error_code) < 0.5:
            for i in range(dim):
                for j in range(dim):
                    d_Linv[idx + dim*i + j] = tmp[dim*i + j]


class PreStepWIJShepardCorrection(Equation):
    def initialize(self, d_idx, d_wij_shepard_denom_1):
        d_wij_shepard_denom_1[d_idx] = 0.

    def loop(self, d_idx, d_wij_shepard_denom_1, s_idx, s_m, s_rho, OLDWIJ):
        d_wij_shepard_denom_1[d_idx] += s_m[s_idx] / s_rho[s_idx] * OLDWIJ

    def post_loop(self, d_idx, d_wij_shepard_denom_1):
        if d_wij_shepard_denom_1[d_idx] >= 1e-9:
            d_wij_shepard_denom_1[d_idx] = 1. / d_wij_shepard_denom_1[d_idx]
        else:
            d_wij_shepard_denom_1[d_idx] = 1


def grad_correction_precomp():
    dim = 3
    dim1 = dim + 1
    dim2 = dim1*dim1

    Linv_code = Template("""
        %for i in range(dim2):
        Linv[${i}] = d_Linv[d_idx*${dim2} + ${i}]
        %endfor
    """).render(dim2=dim2)

    OLD_WIJ_code = Template("""
        OLD_WIJ = KERNEL(XIJ, RIJ, HIJ)
    """).render()

    ODWIJ_code = Template("""
        GRADIENT(XIJ, RIJ, HIJ, ODWIJ)
    """).render()

    WIJ_code = Template("""
        WIJ = Linv[0]*OLD_WIJ + Linv[1]*ODWIJ[0] + Linv[2]*ODWIJ[1] + Linv[3]*ODWIJ[2]
    """).render()

    DWIJ_code = Template("""
        DWIJ[0] = Linv[4]*OLD_WIJ + Linv[5]*ODWIJ[0] + Linv[6]*ODWIJ[1] + Linv[7]*ODWIJ[2]
        DWIJ[1] = Linv[8]*OLD_WIJ + Linv[9]*ODWIJ[0] + Linv[10]*ODWIJ[1] + Linv[11]*ODWIJ[2]
        DWIJ[2] = Linv[12]*OLD_WIJ + Linv[13]*ODWIJ[0] + Linv[14]*ODWIJ[1] + Linv[15]*ODWIJ[2]
    """).render()

    Group.pre_comp.update(
        Linv=BasicCodeBlock(
            code=Linv_code,
            Linv=[0.0]*dim2
        ),
        OLD_WIJ=BasicCodeBlock(
            code=OLD_WIJ_code,
            OLD_WIJ=0.0
        ),
        WIJ=BasicCodeBlock(
            code=WIJ_code,
            WIJ=0.0
        ),
        DWIJ=BasicCodeBlock(
            code=DWIJ_code,
            DWIJ=[0.0, 0.0, 0.0]
        ),
        ODWIJ=BasicCodeBlock(
            code=ODWIJ_code,
            ODWIJ=[0.0, 0.0, 0.0]
        )
    )


class KGF(GradientCorrectionPreStep):
    def loop(self, d_L, d_idx, s_m, s_rho, s_idx, RIJ, XIJ, HIJ,
            SPH_KERNEL):
        V_j = s_m[s_idx] / s_rho[s_idx]

        idx, i, j = declare('int', 3)
        idx = self.dim2*d_idx

        # Append `1` to XIJ to compute zeroth moment.
        YJI = declare('matrix(4)')
        YJI[0] = 1.0
        YJI[1] = -XIJ[0]
        YJI[2] = -XIJ[1]
        YJI[3] = -XIJ[2]

        wij = SPH_KERNEL.kernel(XIJ, RIJ, HIJ)

        for i in range(self.dim):
            for j in range(self.dim):
                d_L[idx + self.dim*i + j] += V_j*wij*YJI[i]*YJI[j]


def kgf_precomp():
    dim = 3
    dim1 = dim + 1
    dim2 = dim1*dim1

    Linv_code = Template("""
        %for i in range(dim2):
        Linv[${i}] = d_Linv[d_idx*${dim2} + ${i}]
        %endfor
    """).render(dim2=dim2)

    OLD_WIJ_code = Template("""
        OLD_WIJ = KERNEL(XIJ, RIJ, HIJ)
    """).render()

    WIJ_code = Template("""
        WIJ = Linv[0]*OLD_WIJ - Linv[1]*OLD_WIJ*XIJ[0] - Linv[2]*OLD_WIJ*XIJ[1] - Linv[3]*OLD_WIJ*XIJ[2]
    """).render()

    DWIJ_code = Template("""
        DWIJ[0] = Linv[4]*OLD_WIJ - Linv[5]*OLD_WIJ*XIJ[0] - Linv[6]*OLD_WIJ*XIJ[1] - Linv[7]*OLD_WIJ*XIJ[2]
        DWIJ[1] = Linv[8]*OLD_WIJ - Linv[9]*OLD_WIJ*XIJ[0] - Linv[10]*OLD_WIJ*XIJ[1] - Linv[11]*OLD_WIJ*XIJ[2]
        DWIJ[2] = Linv[12]*OLD_WIJ - Linv[13]*OLD_WIJ*XIJ[0] - Linv[14]*OLD_WIJ*XIJ[1] - Linv[15]*OLD_WIJ*XIJ[2]
    """).render()

    Group.pre_comp.update(
        Linv=BasicCodeBlock(
            code=Linv_code,
            Linv=[0.0]*dim2
        ),
        OLD_WIJ=BasicCodeBlock(
            code=OLD_WIJ_code,
            OLD_WIJ=0.0
        ),
        WIJ=BasicCodeBlock(
            code=WIJ_code,
            WIJ=0.0
        ),
        DWIJ=BasicCodeBlock(
            code=DWIJ_code,
            DWIJ=[0.0, 0.0, 0.0]
        )
    )


def get_fixed_beam(beam_length, beam_height, boundary_layers, spacing):
    """
 |||=============
 |||=============
 |||===================================================================|
 |||===================================================================|Beam height
 |||===================================================================|
 |||=============
 |||=============
   <------------><---------------------------------------------------->
      Beam inside                   Beam length
      length
    """
    import matplotlib.pyplot as plt
    # create a block first
    xb, yb = get_2d_block(dx=spacing, length=beam_length+spacing/2.,
                          height=beam_height+spacing/2.)

    xs, ys = get_2d_block(dx=spacing, length=beam_length+spacing/2.,
                          height=4. * spacing)
    ys += max(yb) - min(ys) + spacing
    # plt.scatter(xs, ys, s=1)
    # plt.scatter(xb, yb, s=1)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig("geometry", dpi=300)
    # plt.show()

    return xb, yb, xs, ys


class ApplyForceGradual(Equation):
    def __init__(self, dest, sources, delta_fx=0, delta_fy=0, delta_fz=0):
        self.force_increment_x = delta_fx
        self.force_increment_y = delta_fy
        self.force_increment_z = delta_fz
        super(ApplyForceGradual, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_final_force_time,
                   d_total_force_applied_x, d_total_force_applied_y,
                   d_total_force_applied_z, d_force_idx, d_m, t):
        if t < d_final_force_time[0]:
            if d_idx == 0:
                d_total_force_applied_x[0] += self.force_increment_x
                d_total_force_applied_y[0] += self.force_increment_y
                d_total_force_applied_z[0] += self.force_increment_z

        if d_force_idx[d_idx] == 1:
            d_au[d_idx] += d_total_force_applied_x[0] / d_m[d_idx]
            d_av[d_idx] += d_total_force_applied_y[0] / d_m[d_idx]
            d_aw[d_idx] += d_total_force_applied_z[0] / d_m[d_idx]


class ApplyDampingForce(Equation):
    def __init__(self, dest, sources, c=0.1):
        self.c = c
        super(ApplyDampingForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_m, t):
        d_au[d_idx] -= self.c * d_u[d_idx] / d_m[d_idx]
        d_av[d_idx] -= self.c * d_v[d_idx] / d_m[d_idx]
        d_aw[d_idx] -= self.c * d_w[d_idx] / d_m[d_idx]


class CantileverBeamDeflectionWithTipload(Application):
    def add_user_options(self, group):
        group.add_argument("--rho", action="store", type=float, dest="rho",
                           default=7800.,
                           help="Density of the particle (Defaults to 7800.)")

        group.add_argument(
            "--Vf", action="store", type=float, dest="Vf", default=0.05,
            help="Velocity of the plate (Vf) (Defaults to 0.05)")

        group.add_argument("--length", action="store", type=float,
                           dest="length", default=0.01,
                           help="Length of the plate")

        group.add_argument("--height", action="store", type=float,
                           dest="height", default=0.2,
                           help="height of the plate")

        group.add_argument("--deflection", action="store", type=float,
                           dest="deflection", default=1e-4,
                           help="Deflection of the plate")

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=10,
                           help="No of particles in the height direction")

        group.add_argument("--final-force-time", action="store", type=float,
                           dest="final_force_time", default=1e-3,
                           help="Total time taken to apply the external load")

        group.add_argument("--damping-c", action="store", type=float,
                           dest="damping_c", default=0.1,
                           help="Damping constant in damping force")

        group.add_argument("--material", action="store", type=str,
                           dest="material", default="steel",
                           help="Material of the plate")

        add_bool_argument(group, 'shepard', dest='use_shepard_correction',
                          default=False, help='Use shepard correction')

        add_bool_argument(group, 'bonet', dest='use_bonet_correction',
                          default=False, help='Use Bonet and Lok correction')

        add_bool_argument(group, 'kgf', dest='use_kgf_correction',
                          default=False, help='Use KGF correction')

    def consume_user_options(self):
        self.L = self.options.length
        self.H = self.options.height
        self.N = self.options.N
        self.damping_c = self.options.damping_c

        if self.options.material == "steel":
            self.plate_rho0 = 7800
            self.plate_E = 210 * 1e9
            self.plate_nu = 0.3
        elif self.options.material == "rubber":
            self.plate_rho0 = 1000
            self.plate_E = 2 * 1e6
            self.plate_nu = 0.3975

        elif self.options.material == "aluminium":
            self.plate_rho0 = 2800
            self.plate_G = 26.5 * 1e6
            self.plate_K = 69 * 1e6
            self.plate_nu = (3. * self.plate_K - 2. * self.plate_G) / (
                6. * self.plate_K + 2. * self.plate_G)
            self.plate_E = 2. * self.plate_G * (1 + self.plate_nu)

        self.c0 = get_speed_of_sound(self.plate_E, self.plate_nu,
                                     self.plate_rho0)
        # self.c0 = 5960
        # print("speed of sound is")
        # print(self.c0)
        self.pb = self.plate_rho0 * self.c0**2

        self.edac_alpha = 0.5
        self.hdx = 1.2

        self.dx_plate = self.H / self.N
        self.h = self.hdx * self.dx_plate
        self.edac_nu = self.edac_alpha * self.c0 * self.h / 8

        # attributes for Sun PST technique
        # dummy value, will be updated in consume user options
        self.u_max = 13
        self.mach_no = self.u_max / self.c0

        # force to be applied
        # self.fx = 0
        # self.fy = 262.5 * 1e3
        # self.fz = 0

        self.delta_fx = 0.
        self.delta_fy = 0.
        self.delta_fz = 0.

        # for pre step
        self.seval = None

        # boundary equations
        # self.boundary_equations = get_boundary_identification_etvf_equations(
        # destinations=["plate"], sources=["plate"])
        self.boundary_equations = get_boundary_identification_etvf_equations(
            destinations=["plate"], sources=["plate", "wall"],
            boundaries=["wall"])

        self.wall_layers = 2

        # set the total time of the simulation
        if self.options.material == "steel":
            self.tf = 2e-3
            self.final_force_time = 1e-3

        elif self.options.material == "rubber":
            self.tf = 10e-01
            self.final_force_time = 2e-02

        elif self.options.material == "aluminium":
            self.tf = 10e-01
            self.final_force_time = 2e-02

        self.dt = 0.25 * self.h / (
            (self.plate_E / self.plate_rho0)**0.5 + self.u_max)

        print("timestep is")
        print(self.dt)
        print("total time to run is")
        print(self.tf)

        self.c0 = get_speed_of_sound(self.plate_E, self.plate_nu,
                                     self.plate_rho0)
        self.pb = self.plate_rho0 * self.c0**2
        # self.pb = 0.
        print("timestep is,", self.dt)

        self.dim = 2

        # self.alpha = 0.
        # self.beta = 0.

        self.artificial_stress_eps = 0.3

        # edac constants
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0 * self.h / 8

        # variables regarding the plate which are used to simulate the
        # deflection
        self.deflection = self.options.deflection
        I = self.H**3. / 12.
        L = self.L

        self.fx = 0.
        self.fy = self.deflection * 3. * self.plate_E * I / L**3.
        self.fz = 0.
        print("force to be applied is ")
        print(self.fy)

        # the incremental force to be applied
        self.final_force_time = self.options.final_force_time
        timesteps = self.final_force_time / self.dt
        step_force_x = self.fx / timesteps
        step_force_y = self.fy / timesteps
        step_force_z = self.fz / timesteps

        # now distribute such force over a group of particles
        self.delta_fx = step_force_x / (self.N + 1)
        self.delta_fy = step_force_y / (self.N + 1)
        self.delta_fz = step_force_z / (self.N + 1)

        # for post process
        # get the indices which are half way of the beam
        xp, yp, xw, yw = get_fixed_beam(self.L, self.H,
                                        self.wall_layers, self.dx_plate)
        xp_unique = np.unique(xp)
        size = len(xp_unique)
        half_xp = xp_unique[int(size / 2 + size / 8)]
        self.indices_half_way = np.where(xp == half_xp)[0]
        # print(self.indices_half_way)

        # get the indices in the mid way (y=0)
        # y_zero_indices = np.where(yp == 0)
        # self.indices_mid_way = np.where(xp[tmp_indices] >= 0.)
        # x_greater_than_zero_indices = np.where(xp >= 0.)

        # self.indices_mid_way = np.intersect1d(y_zero_indices,
        #                                       x_greater_than_zero_indices)
        self.indices_mid_way = np.where(yp == 0)
        # print(self.indices_mid_way)

        # for shepard correction
        self.use_shepard_correction = self.options.use_shepard_correction
        self.use_bonet_correction = self.options.use_bonet_correction
        self.use_kgf_correction = self.options.use_kgf_correction

    def create_particles(self):
        xp, yp, xw, yw = get_fixed_beam(self.L, self.H, self.wall_layers,
                                        self.dx_plate)
        # make sure that the beam intersection with wall starts at the 0.
        min_xp = np.min(xp)

        # add this to the beam and wall
        xp += abs(min_xp)
        xw += abs(min_xp)

        max_xw = np.max(xw)
        xp -= abs(max_xw)
        xw -= abs(max_xw)

        m = self.plate_rho0 * self.dx_plate**2.

        plate = get_particle_array(
            x=xp, y=yp, m=m, h=self.h, rho=self.plate_rho0, name="plate",
            constants={
                'E': self.plate_E,
                'n': 4,
                'nu': self.plate_nu,
                'spacing0': self.dx_plate,
                'rho_ref': self.plate_rho0
            })
        # ============================================ #
        # find all the indices which are at the right most
        # ============================================ #
        max_x = max(xp)
        indices = np.where(xp > max_x - self.dx_plate / 2.)
        # print(indices[0])
        force_idx = np.zeros_like(plate.x)
        plate.add_property('force_idx', type='int', data=force_idx)
        # print(beam.zero_force_idx)
        plate.force_idx[indices] = 1

        plate.add_constant('final_force_time',
                           np.array([self.final_force_time]))
        plate.add_constant('total_force_applied_x', np.array([0.]))
        plate.add_constant('total_force_applied_y', np.array([0.]))
        plate.add_constant('total_force_applied_z', np.array([0.]))

        # save the indices of which we compute the bending moment
        plate.add_property('indices_half_way')
        plate.indices_half_way[:] = 0.
        plate.indices_half_way[self.indices_half_way] = 1.

        # indices at y=0
        plate.add_property('indices_mid_way')
        plate.indices_mid_way[:] = 0.
        plate.indices_mid_way[self.indices_mid_way] = 1.

        # create the particle array
        wall = get_particle_array(
            x=xw, y=yw, m=m, h=self.h, rho=self.plate_rho0, name="wall",
            constants={
                'E': self.plate_E,
                'n': 4,
                'nu': self.plate_nu,
                'spacing0': self.dx_plate,
                'rho_ref': self.plate_rho0
            })

        self.scheme.setup_properties([wall, plate])

        # properties for shepard correction
        if self.use_shepard_correction is True:
            # 1 / wij_shepard_denom
            plate.add_property('wij_shepard_denom_1')
            plate.wij_shepard_denom_1[:] = 1.

            wall.add_property('wij_shepard_denom_1')
            wall.wij_shepard_denom_1[:] = 1.

        if self.use_bonet_correction or self.use_kgf_correction:
            dim = 3 + 1
            dim2 = dim*dim

            plate.add_property('L', stride=dim2)
            plate.add_property('Linv', stride=dim2)
            for i in range(dim):
                plate.Linv[i*(dim + 1)::dim2] = 1.0

            wall.add_property('L', stride=dim2)
            wall.add_property('Linv', stride=dim2)
            for i in range(dim):
                wall.Linv[i*(dim + 1)::dim2] = 1.0

        return [plate, wall]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        # c0 = self.c0
        self.ma = self.u_max / self.c0
        self.scheme.configure(pb=self.pb, edac_nu=self.edac_nu,
                              mach_no=self.mach_no,
                              hdx=self.hdx)

        self.scheme.configure_solver(tf=tf, dt=dt, pfreq=500)

    def create_scheme(self):
        solid = SolidsScheme(solids=['plate'], boundaries=['wall'], dim=2,
                             artificial_vis_alpha=1., pb=0., edac_nu=0.,
                             mach_no=0., hdx=0.,
                             gx=0., gy=-9.81)

        s = SchemeChooser(default='solid', solid=solid)
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # Shepard corrections
        if self.use_shepard_correction is True:
            shepard_corrections()

            shepard_eqs = []
            shepard_eqs.append(
                PreStepWIJShepardCorrection("plate", sources=["plate", "wall"]))

            eqns.groups[-1].insert(0, Group(shepard_eqs))

        if self.use_bonet_correction is True:
            bonet_eqs = []
            grad_correction_precomp()
            bonet_eqs.append(
                GradientCorrectionPreStep(
                    dest="plate", sources=["plate", "wall"], dim=self.dim
            ))

            eqns.groups[-1].insert(1, Group(bonet_eqs))

        if self.use_kgf_correction is True:
            kgf_eqs = []
            kgf_precomp()
            kgf_eqs.append(
                KGF(
                    dest="plate", sources=["plate", "wall"], dim=self.dim
            ))

            eqns.groups[-1].insert(1, Group(kgf_eqs))

        # # Apply external force
        # force_eqs = []
        # force_eqs.append(
        #     ApplyForceGradual("plate", sources=None, delta_fx=self.delta_fx,
        #                       delta_fy=self.delta_fy, delta_fz=self.delta_fz))

        # eqns.groups[-1].append(Group(force_eqs))
        # # print(eqns.groups[-1])

        # Apply damping force
        force_eqs = []
        force_eqs.append(
            ApplyDampingForce("plate", sources=None, c=self.damping_c))

        eqns.groups[-1].append(Group(force_eqs))
        # print(eqns)

        return eqns

    def _make_accel_eval(self, equations, pa_arrays):
        if self.seval is None:
            kernel = self.scheme.scheme.kernel(dim=self.dim)
            seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                                 dim=self.dim, kernel=kernel)
            self.seval = seval
            return self.seval
        else:
            self.seval.update()
            return self.seval
        return seval

    def pre_step(self, solver):
        if self.options.pst in ['sun2019', 'ipst']:
            if solver.count % 1 == 0:
                t = solver.t
                dt = solver.dt

                arrays = self.particles
                a_eval = self._make_accel_eval(self.boundary_equations, arrays)

                # When
                a_eval.evaluate(t, dt)

    def post_process(self, fname):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output, load

        files = self.output_files
        t, max_y = [], []
        for sd, array in iter_output(files, 'plate'):
            _t = sd['t']
            t.append(_t)
            max_y.append(max(array.y))

        import matplotlib
        import os
        # matplotlib.use('Agg')

        from matplotlib import pyplot as plt

        # res = os.path.join(self.output_dir, "results.npz")
        # np.savez(res, t=t, amplitude=amplitude)

        # gtvf data
        # data = np.loadtxt('./oscillating_plate.csv', delimiter=',')
        # t_gtvf, amplitude_gtvf = data[:, 0], data[:, 1]

        # plt.clf()

        # plt.plot(t_gtvf, amplitude_gtvf, "s-", label='GTVF Paper')
        label = 'max y position of plate with length ' + str(
            self.L) + ' and a deflection of ' + str(self.deflection)
        plt.plot(t, max_y, label=label)

        plt.xlabel('t')
        plt.ylabel('Max y')
        plt.legend()
        fig = os.path.join(self.output_dir, "max_y.png")
        plt.savefig(fig, dpi=300)
        plt.clf()

        files = self.output_files
        last_file = files[-1]
        data = load(last_file)
        # solver_data = data['solver_data']
        arrays = data['arrays']
        pa = arrays['plate']
        y = pa.y[self.indices_half_way]
        sigma_00 = pa.sigma00[self.indices_half_way]
        label = 'sigma00_vs_y with length ' + str(
            self.L) + ' and a deflection of ' + str(self.deflection)
        plt.plot(sigma_00, y, label=label)
        plt.xlabel('sigma xx')
        plt.ylabel('y')
        plt.legend()
        fig = os.path.join(self.output_dir, "sigma00_vs_y.png")
        plt.savefig(fig, dpi=300)
        # plt.show()


if __name__ == '__main__':
    app = CantileverBeamDeflectionWithTipload()
    app.run()
    app.post_process(app.info_filename)
    # app.create_rings_geometry()
_
