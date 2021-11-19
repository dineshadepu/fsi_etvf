import numpy as np
from math import cos, sin, sinh, cosh

# SPH equations
from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.scheme import SchemeChooser
from pysph.sph.equation import Equation

from pysph.base.utils import get_particle_array

from solid_mech import SolidsScheme

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.sph.scheme import add_bool_argument
from force_application_utils import (ApplyForceGradual, ApplyForceSudden,
                                     setup_properties_for_gradual_force,
                                     setup_properties_for_sudden_force,
                                     tip_load_force_index_distributed,
                                     tip_load_force_index_single)


def create_circle_1(diameter=1, spacing=0.05, center=None):
    dx = spacing
    x = [0.0]
    y = [0.0]
    r = spacing
    nt = 0
    radius = diameter / 2.

    tmp_dist = radius - spacing/2.
    i = 0
    while tmp_dist > spacing/2.:
        perimeter = 2. * np.pi * tmp_dist
        no_of_points = int(perimeter / spacing) + 1
        theta = np.linspace(0., 2. * np.pi, no_of_points)
        for t in theta[:-1]:
            x.append(tmp_dist * np.cos(t))
            y.append(tmp_dist * np.sin(t))
        i = i + 1
        tmp_dist = radius - spacing/2. - i * spacing

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


def get_fixed_beam(beam_length, beam_height, cylinder_radius,
                   boundary_layers, spacing):
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
    # create a block first
    xb, yb = get_2d_block(dx=spacing,
                          length=beam_length + cylinder_radius,
                          height=beam_height)

    radius = cylinder_radius
    xs, ys = get_2d_block(dx=spacing,
                          length=2. * radius,
                          height=2. * radius)

    indices = []
    for i in range(len(xs)):
        if xs[i]**2. + ys[i]**2. > (radius - spacing/2)**2.:
            indices.append(i)

    xs = np.delete(xs, indices)
    ys = np.delete(ys, indices)

    # move the beam to the front of the wall
    xb += max(xb) - min(xs) + spacing

    # move the beam inside the cylinder
    xb -= radius

    # now remove the indices of the beam inside the cylinder
    indices = []
    for i in range(len(xs)):
        if xs[i] > min(xb) and ys[i] > min(yb) - spacing/2. and ys[i] < max(yb) + spacing/2.:
            indices.append(i)

    xs = np.delete(xs, indices)
    ys = np.delete(ys, indices)

    return xb, yb, xs, ys


def get_fixed_beam_no_clamp(beam_length, beam_height, boundary_height,
                            boundary_length, boundary_layers, spacing):
    """
 ||||||||||||||||
 ||||||||||||||||
 ||||||||||||||||======================================================|
 ||||||||||||||||======================================================|Beam
 ||||||||||||||||======================================================|height
 ||||||||||||||||
 ||||||||||||||||
   <------------><---------------------------------------------------->
      Beam inside                   Beam length
      length
    """
    # create a block first
    xb, yb = get_2d_block(dx=spacing,
                          length=beam_length,
                          height=beam_height)

    xs, ys = get_2d_block(dx=spacing,
                          length=boundary_length,
                          height=boundary_height)

    # move the beam to the front of the wall
    xb += max(xb) - min(xs) + spacing

    return xb, yb, xs, ys


def find_displacement_index(pa):
    x = pa.x
    y = pa.y
    max_x = max(x)
    max_x_indices = np.where(x == max_x)[0]
    index = max_x_indices[int(len(max_x_indices)/2)]
    pa.add_property('tip_displacemet_index', type='int',
                    data=np.zeros(len(pa.x)))
    pa.tip_displacemet_index[index] = 1


class StaticCantileverBeamUnderTipLoad(Application):
    def add_user_options(self, group):
        group.add_argument("--N",
                           action="store",
                           type=int,
                           dest="N",
                           default=10,
                           help="No of particles in the height direction")

        add_bool_argument(group, 'clamp', dest='clamp',
                          default=True, help='Clamped beam')

        group.add_argument("--clamp-factor",
                           action="store",
                           type=float,
                           dest="clamp_factor",
                           default=2.5,
                           help="Amount of beam to be clamped")

        add_bool_argument(group, 'gradual-force', dest='gradual_force',
                          default=True, help='Apply gradual force')

        group.add_argument("--gradual-force-time",
                           action="store",
                           type=float,
                           dest="gradual_force_time",
                           default=0.1,
                           help="Total time of gradual force")

        add_bool_argument(group, 'distributed-load', dest='distributed_load',
                          default=True, help='Apply load to a set of particles')

    def consume_user_options(self):
        self.N = self.options.N
        self.clamp = self.options.clamp
        self.clamp_factor = self.options.clamp_factor
        self.gradual_force = self.options.gradual_force
        self.gradual_force_time = self.options.gradual_force_time
        self.distributed_load = self.options.distributed_load

        # =============================
        # general simulation parameters
        # =============================
        self.hdx = 1.2
        self.dim = 2
        self.seval = None  # for pre step
        self.gx = 0.
        self.gy = 0.
        self.gz = 0.

        # ====================
        # structure properties
        # ====================
        self.plate_rho0 = 1000.
        self.plate_E = 1. * 1e5
        self.plate_nu = 0.3
        # attributes for Sun PST technique
        self.c0 = get_speed_of_sound(self.plate_E, self.plate_nu,
                                     self.plate_rho0)
        self.L = 0.35
        self.H = 0.02
        self.dx_plate = self.H / self.N
        self.h = self.hdx * self.dx_plate
        # CTVF Scheme specific variables for the structure
        # boundary equations
        self.u_max = 2.8513
        self.mach_no = self.u_max / self.c0
        if self.options.wall_pst is True:
            self.boundary_equations = get_boundary_identification_etvf_equations(
                destinations=["plate"],
                sources=["plate", "wall"],
                boundaries=["wall"])
        else:
            self.boundary_equations = get_boundary_identification_etvf_equations(
                destinations=["plate"],
                sources=["plate"],
                boundaries=None)

        # ===============
        # wall properties
        # ===============
        self.cylinder_r = 0.05
        self.wall_layers = 2

        # compute the timestep
        self.tf = 10.
        self.dt = 0.25 * self.h / (
            (self.plate_E / self.plate_rho0)**0.5 + 2.85)

        # ===========================
        # problem specific parameters
        # ===========================
        self.fx = 0.
        self.fy = -0.005
        self.fz = 0.

        # the incremental force to be applied
        self.gradual_force_time = self.options.gradual_force_time
        timesteps = self.gradual_force_time / self.dt
        self.step_force_x = self.fx / timesteps
        self.step_force_y = self.fy / timesteps
        self.step_force_z = self.fz / timesteps

        # now distribute such force over a group of particles
        self.delta_fx = 0.
        self.delta_fy = 0.
        self.delta_fz = 0.

        if self.distributed_load is True:
            self.delta_fx = self.step_force_x / (self.N + 1)
            self.delta_fy = self.step_force_y / (self.N + 1)
            self.delta_fz = self.step_force_z / (self.N + 1)

        if self.distributed_load is True:
            self.fx = 0.
            self.fy = -0.005 / (self.N + 1)
            self.fz = 0.

    def create_particles(self):
        if self.clamp is True:
            xp, yp, xw, yw = get_fixed_beam_no_clamp(self.L, self.H, 2. * self.H,
                                                     4. * self.dx_plate,
                                                     self.wall_layers + 1,
                                                     self.dx_plate)
        else:
            xp, yp, xw, yw = get_fixed_beam_no_clamp(self.L, self.H, self.H,
                                                     4. * self.dx_plate,
                                                     self.wall_layers + 1,
                                                     self.dx_plate)
        # make sure that the beam intersection with wall starts at the 0.
        xw -= max(xw) - min(xp) + self.dx_plate

        m = self.plate_rho0 * self.dx_plate**2.

        plate = get_particle_array(x=xp,
                                   y=yp,
                                   m=m,
                                   h=self.h,
                                   rho=self.plate_rho0,
                                   E=self.plate_E,
                                   nu=self.plate_nu,
                                   rho_ref=self.plate_rho0,
                                   name="plate",
                                   constants={
                                       'n': 4,
                                       'spacing0': self.dx_plate,
                                   })

        # create the particle array
        wall = get_particle_array(x=xw,
                                  y=yw,
                                  m=m,
                                  h=self.h,
                                  rho=self.plate_rho0,
                                  E=self.plate_E,
                                  nu=self.plate_nu,
                                  rho_ref=self.plate_rho0,
                                  name="wall",
                                  constants={
                                      'n': 4,
                                      'spacing0': self.dx_plate,
                                  })

        self.scheme.setup_properties([wall, plate])

        if self.gradual_force is True:
            setup_properties_for_gradual_force(plate)
            plate.gradual_force_time[0] = self.gradual_force_time
        else:
            setup_properties_for_sudden_force(plate)

        # find the particle indices where the force have to be applied
        if self.distributed_load is True:
            tip_load_force_index_distributed(plate)
        else:
            tip_load_force_index_single(plate)

        # add post processing variables.
        find_displacement_index(plate)

        ##################################
        # Add output arrays
        ##################################
        plate.add_output_arrays(['sigma00', 'force_idx', 'is_boundary',
                                 'tip_displacemet_index'])

        return [plate, wall]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=500)

        scheme = self.scheme
        scheme.configure(mach_no=self.mach_no, gy=self.gy)

    def create_scheme(self):
        # the scheme parameters are dummy and reinitialized in configure_scheme
        solid = SolidsScheme(solids=['plate'],
                             boundaries=['wall'],
                             dim=2,
                             mach_no=1.,
                             gy=0)

        s = SchemeChooser(default='solid', solid=solid)
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # Apply external force
        force_eqs = []
        if self.gradual_force is True:
            if self.distributed_load is True:
                force_eqs.append(
                    ApplyForceGradual(dest="plate", sources=None,
                                      delta_fx=self.delta_fx,
                                      delta_fy=self.delta_fy,
                                      delta_fz=self.delta_fz))
            else:
                force_eqs.append(
                    ApplyForceGradual(dest="plate", sources=None,
                                      delta_fx=self.step_force_x,
                                      delta_fy=self.step_force_y,
                                      delta_fz=self.step_force_z))
        else:
            force_eqs.append(
                ApplyForceSudden(dest="plate", sources=None,
                                 fx=self.fx,
                                 fy=self.fy,
                                 fz=self.fz))

        eqns.groups[-1].append(Group(force_eqs))

        return eqns

    def _make_accel_eval(self, equations, pa_arrays):
        if self.seval is None:
            kernel = self.scheme.scheme.kernel(dim=self.dim)
            seval = SPHEvaluator(arrays=pa_arrays,
                                 equations=equations,
                                 dim=self.dim,
                                 kernel=kernel)
            self.seval = seval
            return self.seval
        else:
            self.seval.update()
            return self.seval
        return seval

    def pre_step(self, solver):
        if solver.count % 10 == 0:
            t = solver.t
            dt = solver.dt

            arrays = self.particles
            a_eval = self._make_accel_eval(self.boundary_equations, arrays)

            # When
            a_eval.evaluate(t, dt)

    def post_process(self, fname):
        from pysph.solver.utils import iter_output, load
        from pysph.solver.utils import get_files

        files = self.output_files

        data = load(files[0])
        # solver_data = data['solver_data']
        arrays = data['arrays']
        pa = arrays['plate']

        files = files[0::10]
        # print(len(files))
        t, amplitude = [], []
        index = np.where(pa.tip_displacemet_index == 1)

        for sd, plate in iter_output(files, 'plate'):
            _t = sd['t']
            t.append(_t)
            amplitude.append(plate.y[index])

        import os
        from matplotlib import pyplot as plt

        plt.clf()

        plt.plot(t, amplitude, "-", label='Simulated')

        plt.xlabel('t')
        plt.ylabel('amplitude')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "amplitude_with_t.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = StaticCantileverBeamUnderTipLoad()
    app.run()
    app.post_process(app.info_filename)
