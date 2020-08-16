import numpy as np

from pysph.sph.scheme import add_bool_argument

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline)
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

# from pysph.sph.rigid_body import (BodyForce)

from pysph.base.kernels import (CubicSpline, WendlandQuintic,
                                QuinticSpline, WendlandQuinticC4,
                                Gaussian, SuperGaussian)

from rigid_body_common import (
    set_total_mass, set_center_of_mass,
    set_body_frame_position_vectors,
    set_body_frame_normal_vectors,
    set_moment_of_inertia_and_its_inverse,
    BodyForce,
    SumUpExternalForces,
    normalize_R_orientation)

# compute the boundary particles
from boundary_particles import (get_boundary_identification_etvf_equations,
                                add_boundary_identification_properties)
from numpy import sin, cos


class ContinuityEquation(Equation):
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_arho, s_m, s_vol, d_rho, VIJ, DWIJ):
        vijdotdwij = VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] + VIJ[2] * DWIJ[2]
        d_arho[d_idx] += d_rho[d_idx] * vijdotdwij * s_vol[d_idx]

class SetVolumeFromDensityBoundary(Equation):
    def post_loop(self, d_idx, d_rho, d_vol, d_m_fluid):
        d_vol[d_idx] = d_m_fluid[d_idx] / d_rho[d_idx]


class MomentumEquationPressureGradient(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_au, d_av,
             d_aw, DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class MomentumEquationPressureGradientBoundary(Equation):
    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m_fluid, d_au, d_av,
             d_aw, DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2

        tmp = -s_m_fluid[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class FluidStep(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_m, d_vol, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_arho, dt):
        dtb2 = 0.5*dt
        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]
        d_vol[d_idx] = d_m[d_idx] / d_rho[d_idx]

    def stage2(self, d_idx, d_m, d_vol, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_arho, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]
        d_vol[d_idx] = d_m[d_idx] / d_rho[d_idx]


class RK2RigidBody3DStep(IntegratorStep):
    def py_initialize(self, dst, t, dt):
        for i in range(dst.nb[0]):
            for j in range(3):
                # save the center of mass and center of mass velocity
                dst.xcm0[3*i+j] = dst.xcm[3*i+j]
                dst.vcm0[3*i+j] = dst.vcm[3*i+j]

                # save the current angular momentum
                dst.ang_mom0[j] = dst.ang_mom[j]

            # save the current orientation
            for j in range(9):
                dst.R0[9*i+j] = dst.R[9*i+j]

    def initialize(self):
        pass

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.xcm[i3+j] = dst.xcm0[i3+j] + dtb2 * dst.vcm[i3+j]
                dst.vcm[i3+j] = dst.vcm0[i3+j] + dtb2 * dst.force[i3+j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3+2], dst.omega[i3+1]],
                                  [dst.omega[i3+2], 0, -dst.omega[i3+0]],
                                  [-dst.omega[i3+1], dst.omega[i3+0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9+9] = dst.R0[i9:i9+9] + r_dot * dtb2

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9+9])

            # update the moment of inertia
            R = dst.R[i9:i9+9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(R, dst.inertia_tensor_inverse_body_frame[i9:i9+9].reshape(3, 3))
            dst.inertia_tensor_inverse_global_frame[i9:i9+9] = (np.matmul(tmp, R_t)).ravel()[:]

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3+3] = dst.ang_mom0[i3:i3+3] + (dtb2 * dst.torque[i3:i3+3])

            dst.omega[i3:i3+3] = np.matmul(dst.inertia_tensor_inverse_global_frame[i9:i9+9].reshape(3, 3), dst.ang_mom[i3:i3+3])

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_normal0, d_normal, d_is_boundary):
        # some variables to update the positions seamlessly
        bid, i9, i3, idx3 = declare('int', 4)
        bid = d_body_id[d_idx]
        idx3 = 3 * d_idx
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9+0] * d_dx0[d_idx] + d_R[i9+1] * d_dy0[d_idx] +
              d_R[i9+2] * d_dz0[d_idx])
        dy = (d_R[i9+3] * d_dx0[d_idx] + d_R[i9+4] * d_dy0[d_idx] +
              d_R[i9+5] * d_dz0[d_idx])
        dz = (d_R[i9+6] * d_dx0[d_idx] + d_R[i9+7] * d_dy0[d_idx] +
              d_R[i9+8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3+0] + dx
        d_y[d_idx] = d_xcm[i3+1] + dy
        d_z[d_idx] = d_xcm[i3+2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3+1] * dz - d_omega[i3+2] * dy
        dv = d_omega[i3+2] * dx - d_omega[i3+0] * dz
        dw = d_omega[i3+0] * dy - d_omega[i3+1] * dx

        d_u[d_idx] = d_vcm[i3+0] + du
        d_v[d_idx] = d_vcm[i3+1] + dv
        d_w[d_idx] = d_vcm[i3+2] + dw

        if d_is_boundary[d_idx] == 1:
            d_normal[idx3+0] = (d_R[i9+0] * d_normal0[idx3] + d_R[i9+1] * d_normal0[idx3+1] +
                d_R[i9+2] * d_normal0[idx3+2])
            d_normal[idx3+1] = (d_R[i9+3] * d_normal0[idx3] + d_R[i9+4] * d_normal0[idx3+1] +
                d_R[i9+5] * d_normal0[idx3+2])
            d_normal[idx3+2] = (d_R[i9+6] * d_normal0[idx3] + d_R[i9+7] * d_normal0[idx3+1] +
                d_R[i9+8] * d_normal0[idx3+2])

    def py_stage2(self, dst, t, dt):
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.xcm[i3+j] = dst.xcm0[i3+j] + dt * dst.vcm[i3+j]
                dst.vcm[i3+j] = dst.vcm0[i3+j] + dt * dst.force[i3+j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3+2], dst.omega[i3+1]],
                                  [dst.omega[i3+2], 0, -dst.omega[i3+0]],
                                  [-dst.omega[i3+1], dst.omega[i3+0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9+9] = dst.R0[i9:i9+9] + r_dot * dt

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9+9])

            # update the moment of inertia
            R = dst.R[i9:i9+9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(R, dst.inertia_tensor_inverse_body_frame[i9:i9+9].reshape(3, 3))
            dst.inertia_tensor_inverse_global_frame[i9:i9+9] = (np.matmul(tmp, R_t)).ravel()[:]

            # move angular velocity to t + dt
            dst.ang_mom[i3:i3+3] = dst.ang_mom0[i3:i3+3] + (dt * dst.torque[i3:i3+3])
            dst.omega[i3:i3+3] = np.matmul(dst.inertia_tensor_inverse_global_frame[i9:i9+9].reshape(3, 3), dst.ang_mom[i3:i3+3])

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_normal0, d_normal, d_is_boundary):
        # some variables to update the positions seamlessly
        bid, i9, i3, idx3 = declare('int', 4)
        bid = d_body_id[d_idx]
        idx3 = 3 * d_idx
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9+0] * d_dx0[d_idx] + d_R[i9+1] * d_dy0[d_idx] +
              d_R[i9+2] * d_dz0[d_idx])
        dy = (d_R[i9+3] * d_dx0[d_idx] + d_R[i9+4] * d_dy0[d_idx] +
              d_R[i9+5] * d_dz0[d_idx])
        dz = (d_R[i9+6] * d_dx0[d_idx] + d_R[i9+7] * d_dy0[d_idx] +
              d_R[i9+8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3+0] + dx
        d_y[d_idx] = d_xcm[i3+1] + dy
        d_z[d_idx] = d_xcm[i3+2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3+1] * dz - d_omega[i3+2] * dy
        dv = d_omega[i3+2] * dx - d_omega[i3+0] * dz
        dw = d_omega[i3+0] * dy - d_omega[i3+1] * dx

        d_u[d_idx] = d_vcm[i3+0] + du
        d_v[d_idx] = d_vcm[i3+1] + dv
        d_w[d_idx] = d_vcm[i3+2] + dw

        if d_is_boundary[d_idx] == 1:
            d_normal[idx3+0] = (d_R[i9+0] * d_normal0[idx3] + d_R[i9+1] * d_normal0[idx3+1] +
                d_R[i9+2] * d_normal0[idx3+2])
            d_normal[idx3+1] = (d_R[i9+3] * d_normal0[idx3] + d_R[i9+4] * d_normal0[idx3+1] +
                d_R[i9+5] * d_normal0[idx3+2])
            d_normal[idx3+2] = (d_R[i9+6] * d_normal0[idx3] + d_R[i9+7] * d_normal0[idx3+1] +
                d_R[i9+8] * d_normal0[idx3+2])


class RigidFluidCouplingScheme(Scheme):
    def __init__(self, fluids, boundaries, rigid_bodies, dim, rho0, p0, c0,
                 integrator="rk2", gx=0.0, gy=0.0, gz=0.0, kernel_choice="1",
                 kernel_factor=3):
        self.rigid_bodies = rigid_bodies

        if boundaries == None:
            self.boundaries = []
        else:
            self.boundaries = boundaries


        if fluids == None:
            self.fluids = []
        else:
            self.fluids = fluids

        if rigid_bodies == None:
            self.rigid_bodies = []
        else:
            self.rigid_bodies = rigid_bodies

        self.dim = dim

        self.kernel = CubicSpline

        self.integrator = integrator

        self.rho0 = rho0
        self.p0 = p0
        self.c0 = c0

        self.gx = gx
        self.gy = gy
        self.gz = gz

        self.solver = None

    def get_equations(self):
        return self._get_rk2_equations()

    def _get_rk2_equations(self):
        from pysph.sph.equation import Group, MultiStageEquations
        from pysph.sph.wc.transport_velocity import (SolidWallPressureBC)
        from pysph.sph.wc.basic import (TaitEOS)
        equations = []
        all = list(set(self.rigid_bodies + self.boundaries + self.fluids))
        boundaries = list(set(self.rigid_bodies + self.boundaries))

        g1 = []
        g2 = []
        if len(self.rigid_bodies) + len(self.boundaries) > 0:
            for name in self.rigid_bodies:
                g1.append(
                    BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy,
                            gz=self.gz))

            equations.append(Group(equations=g1))

            # extrapolate the pressure of the rigid body from the fluid
            if len(self.fluids) > 0:
                for name in boundaries:
                    g2.append(
                        SolidWallPressureBC(dest=name, sources=self.fluids,
                                            rho0=self.rho0, p0=self.p0,
                                            gx=self.gx, gy=self.gy, gz=self.gz))

                    g2.append(
                        SetVolumeFromDensityBoundary(dest=name, sources=None))

                equations.append(Group(equations=g2))

        if len(self.fluids) > 0:
            ######################
            # ContinuityEquation #
            ######################
            g3 = []
            for name in self.fluids:
                g3.append(
                    ContinuityEquation(dest=name, sources=all))

            equations.append(Group(equations=g3))

            #####################
            # Equation of state #
            #####################
            g4 = []
            for name in self.fluids:
                g4.append(
                    TaitEOS(dest=name, sources=None, rho0=self.rho0, c0=self.c0,
                            gamma=7.0))

            equations.append(Group(equations=g4))

        ####################
        # MomentumEquation #
        ####################
        g5 = []
        if len(self.fluids) > 0:
            for name in self.fluids:
                g5.append(
                    MomentumEquationPressureGradient(
                        dest=name, sources=self.fluids, gx=self.gx, gy=self.gy,
                        gz=self.gz))

                if len(boundaries) > 0:
                    g5.append(
                        MomentumEquationPressureGradientBoundary(
                            dest=name, sources=boundaries))

            equations.append(Group(equations=g5))

        # computation of total force and torque at cener of mass
        g6 = []
        for name in self.rigid_bodies:
            g6.append(SumUpExternalForces(dest=name, sources=None))
        equations.append(Group(equations=g6, real=False))

        return equations

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        from pysph.sph.integrator import EPECIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = FluidStep()

        for body in self.rigid_bodies:
            if body not in steppers:
                steppers[body] = RK2RigidBody3DStep()

        cls = integrator_cls if integrator_cls is not None else EPECIntegrator
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties

        pas = dict([(p.name, p) for p in particles])

        for rigid_body in self.rigid_bodies:
            pa = pas[rigid_body]

            add_properties(pa, 'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0')

            nb = int(np.max(pa.body_id) + 1)

            # dem_id = props.pop('dem_id', None)

            consts = {
                'total_mass': np.zeros(nb, dtype=float),
                'xcm': np.zeros(3*nb, dtype=float),
                'xcm0': np.zeros(3*nb, dtype=float),
                'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
                'R0': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
                # moment of inertia izz (this is only for 2d)
                'izz': np.zeros(nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_body_frame': np.zeros(9*nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_inverse_body_frame': np.zeros(9*nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_global_frame': np.zeros(9*nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_inverse_global_frame': np.zeros(9*nb, dtype=float),
                # total force at the center of mass
                'force': np.zeros(3*nb, dtype=float),
                # torque about the center of mass
                'torque': np.zeros(3*nb, dtype=float),
                # velocity, acceleration of CM.
                'vcm': np.zeros(3*nb, dtype=float),
                'vcm0': np.zeros(3*nb, dtype=float),
                # angular momentum
                'ang_mom': np.zeros(3*nb, dtype=float),
                'ang_mom0': np.zeros(3*nb, dtype=float),
                # angular velocity in global frame
                'omega': np.zeros(3*nb, dtype=float),
                'omega0': np.zeros(3*nb, dtype=float),

                'nb': nb
            }

            for key, elem in consts.items():
                pa.add_constant(key, elem)

            # compute the properties of the body
            set_total_mass(pa)
            set_center_of_mass(pa)

            # this function will compute
            # inertia_tensor_body_frame
            # inertia_tensor_inverse_body_frame
            # inertia_tensor_global_frame
            # inertia_tensor_inverse_global_frame
            # of the rigid body
            set_moment_of_inertia_and_its_inverse(pa)

            set_body_frame_position_vectors(pa)

            ####################################################
            # compute the boundary particles of the rigid body #
            ####################################################
            add_boundary_identification_properties(pa)
            # make sure your rho is not zero
            equations = get_boundary_identification_etvf_equations([pa.name], [pa.name])
            # print(equations)

            sph_eval = SPHEvaluator(arrays=[pa],
                                    equations=equations,
                                    dim=self.dim,
                                    kernel=QuinticSpline(dim=self.dim))

            sph_eval.evaluate(dt=0.1)

            # make normals of particle other than boundary particle as zero
            for i in range(len(pa.x)):
                if pa.is_boundary[i] == 0:
                    pa.normal[3*i] = 0.
                    pa.normal[3*i+1] = 0.
                    pa.normal[3*i+2] = 0.

            # normal vectors in terms of body frame
            set_body_frame_normal_vectors(pa)

            # properties for pressure extrapolation
            pa.add_property('wij')

            pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy',
                                  'normal', 'is_boundary', 'fz', 'm', 'body_id'])


        for boundary in self.boundaries:
            pa = pas[boundary]
            pa.add_property('wij')
            # pa.add_property('m_fluid')

        for fluid in self.fluids:
            pa = pas[fluid]

            add_properties(pa, 'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                           'arho', 'vol', 'cs')

            pa.vol[:] = pa.m[:] / pa.rho[:]

            pa.cs[:] = self.c0

    def _set_particle_velocities(self, pa):
        for i in range(max(pa.body_id) + 1):
            fltr = np.where(pa.body_id == i)
            bid = i
            i9 = 9 * bid
            i3 = 3 * bid

            for j in fltr:
                dx = (pa.R[i9+0] * pa.dx0[j] + pa.R[i9+1] * pa.dy0[j] +
                      pa.R[i9+2] * pa.dz0[j])
                dy = (pa.R[i9+3] * pa.dx0[j] + pa.R[i9+4] * pa.dy0[j] +
                      pa.R[i9+5] * pa.dz0[j])
                dz = (pa.R[i9+6] * pa.dx0[j] + pa.R[i9+7] * pa.dy0[j] +
                      pa.R[i9+8] * pa.dz0[j])

                du = pa.omega[i3+1] * dz - pa.omega[i3+2] * dy
                dv = pa.omega[i3+2] * dx - pa.omega[i3+0] * dz
                dw = pa.omega[i3+0] * dy - pa.omega[i3+1] * dx

                pa.u[j] = pa.vcm[i3+0] + du
                pa.v[j] = pa.vcm[i3+1] + dv
                pa.w[j] = pa.vcm[i3+2] + dw

    def set_linear_velocity(self, pa, linear_vel):
        pa.vcm[:] = linear_vel

        self._set_particle_velocities(pa)

    def set_angular_velocity(self, pa, angular_vel):
        pa.omega[:] = angular_vel[:]

        # set the angular momentum
        for i in range(max(pa.body_id) + 1):
            i9 = 9 * i
            i3 = 3 * i
            pa.ang_mom[i3:i3+3] = np.matmul(pa.inertia_tensor_global_frame[i9:i9+9].reshape(3, 3), pa.omega[i3:i3+3])[:]

        self._set_particle_velocities(pa)

    def get_solver(self):
        return self.solver
