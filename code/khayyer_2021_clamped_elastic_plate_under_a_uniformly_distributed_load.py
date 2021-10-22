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

from solid_mech import SolidsScheme, SolidsSchemeGray

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)
from pysph.tools.geometry import get_2d_tank, get_2d_block
from solid_mech import AddGravityToStructure
from pysph.sph.scheme import add_bool_argument
from force_application_utils import (ApplyForceGradual, ApplyForceSudden,
                                     setup_properties_for_gradual_force,
                                     setup_properties_for_sudden_force,
                                     tip_load_force_index_distributed,
                                     tip_load_force_index_single)
from pysph.tools.geometry import (remove_overlap_particles)
from pysph.solver.utils import iter_output, load
from pysph.solver.utils import get_files


def UDL_force_index_single(plate, clamp):
    max_y = np.max(plate.y)
    indices = np.where(max_y == plate.y)[0]
    if clamp is True:
        indices = (max_y == plate.y) & (plate.x > -0.1) & (plate.x < 0.1)
    else:
        indices = np.where(max_y == plate.y)[0]

    plate.force_idx[indices] = 1


def UDL_force_index_distributed(plate, clamp):
    if clamp is True:
        indices = (plate.x > -0.1) & (plate.x < 0.1)
        plate.force_idx[indices] = 1
    else:
        plate.force_idx[:] = 1


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


def get_beam_clamp(beam_length, beam_height, boundary_length,
                   boundary_height, boundary_layers, spacing):
    # create a block first
    xb, yb = get_2d_block(dx=spacing,
                          length=beam_length + 2. * boundary_length + spacing,
                          height=beam_height)

    xs1, ys1 = get_2d_block(dx=spacing,
                            length=boundary_length,
                            height=boundary_height)
    xs1 -= min(xs1) - min(xb)

    xs2, ys2 = get_2d_block(dx=spacing,
                            length=boundary_length,
                            height=boundary_height)
    xs2 += max(xb) - max(xs2)

    xs3, ys3 = get_2d_block(dx=spacing,
                            length=5. * spacing,
                            height=boundary_height)
    xs3 -= max(xs3) - min(xb) + spacing

    xs4, ys4 = get_2d_block(dx=spacing,
                            length=5. * spacing,
                            height=boundary_height)
    xs4 += max(xb) - min(xs4) + spacing

    xs = np.concatenate([xs1, xs2, xs3, xs4])
    ys = np.concatenate([ys1, ys2, ys3, ys4])

    return xb, yb, xs, ys


def get_beam_no_clamp(beam_length, beam_height, boundary_length,
                      boundary_height, boundary_layers, spacing):
    # create a block first
    xb, yb = get_2d_block(dx=spacing,
                          length=beam_length,
                          height=beam_height)

    xs1, ys1 = get_2d_block(dx=spacing,
                            length=boundary_length,
                            height=boundary_height)

    xs2, ys2 = get_2d_block(dx=spacing,
                            length=boundary_length,
                            height=boundary_height)

    # move the beam to the front of the wall
    # xb += max(xb) - min(xs1) + spacing
    xs1 -= max(xs1) - min(xb) + spacing

    xs2 += max(xb) - min(xs2) + spacing

    xs = np.concatenate([xs1, xs2])
    ys = np.concatenate([ys1, ys2])

    return xb, yb, xs, ys


def find_displacement_index(pa):
    x = pa.x
    y = pa.y
    min_y = min(y)
    min_y_indices = np.where(y == min_y)[0]
    index = min_y_indices[int(len(min_y_indices)/2)]
    pa.add_property('tip_displacemet_index', type='int',
                    data=np.zeros(len(pa.x)))
    pa.tip_displacemet_index[index] = 1


class ClampedElasticPlateUDL(Application):
    def initialize(self):
        # dummy value to make the scheme work
        self.plate_rho0 = 1100.
        self.plate_E = 2.4 * 1e7
        self.plate_nu = 0.0
        self.c0 = get_speed_of_sound(self.plate_E, self.plate_nu,
                                     self.plate_rho0)
        self.pb = self.plate_rho0 * self.c0**2

        self.edac_alpha = 0.5
        self.hdx = 1.2

        # this is dummpy value
        self.h = 0.001
        self.edac_nu = self.edac_alpha * self.c0 * self.h / 8
        print(self.edac_nu)

        # attributes for Sun PST technique
        # dummy value, will be updated in consume user options
        self.u_max = 0.1
        self.mach_no = self.u_max / self.c0

        # self.mach_no = 100 / self.c0

        self.cylinder_r = 0.05

        # for pre step
        self.seval = None

    def add_user_options(self, group):
        group.add_argument("--N",
                           action="store",
                           type=int,
                           dest="N",
                           default=4,
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

        add_bool_argument(group, 'sandwich', dest='sandwich',
                          default=False, help='Assume sandwich plate')

        add_bool_argument(group, 'two-layer', dest='two_layer',
                          default=False, help='Assume two layer composite')

    def consume_user_options(self):
        self.N = self.options.N
        self.clamp = self.options.clamp
        self.clamp_factor = self.options.clamp_factor
        self.gradual_force = self.options.gradual_force
        self.gradual_force_time = self.options.gradual_force_time
        self.distributed_load = self.options.distributed_load
        self.sandwich = self.options.sandwich
        self.two_layer = self.options.two_layer

        self.rho = 1100.
        self.L = 0.2
        self.H = 0.012

        self.dx_plate = self.H / self.N
        print("Spacing is", self.dx_plate)
        self.h = self.hdx * self.dx_plate
        self.plate_rho0 = self.rho

        self.wall_layers = 2

        # compute the timestep
        self.tf = 0.3
        self.dt = 0.25 * self.h / (
            (self.plate_E / self.plate_rho0)**0.5 + 2.85)

        self.c0 = get_speed_of_sound(self.plate_E, self.plate_nu,
                                     self.plate_rho0)
        self.pb = self.plate_rho0 * self.c0**2

        self.dim = 2

        self.artificial_stress_eps = 0.3

        # edac constants
        self.edac_alpha = 1.
        self.edac_nu = self.edac_alpha * self.c0 * self.h / 8
        print("real alpha is", self.edac_nu)

        self.clamp = self.options.clamp
        self.clamp_factor = self.options.clamp_factor

        self.fx = 0.
        self.fy = -4. / (0.2 / self.dx_plate)
        self.fz = 0.
        print("force per particle is", self.fy)

        # the incremental force to be applied
        timesteps = self.gradual_force_time / self.dt
        self.step_force_x = self.fx / timesteps
        self.step_force_y = self.fy / timesteps
        self.step_force_z = self.fz / timesteps

        # now distribute such force over a group of particles
        self.delta_fx = self.step_force_x
        self.delta_fy = self.step_force_y
        self.delta_fz = self.step_force_z

        if self.distributed_load is True:
            self.delta_fx = self.step_force_x / (self.N + 1)
            self.delta_fy = self.step_force_y / (self.N + 1)
            self.delta_fz = self.step_force_z / (self.N + 1)

        if self.distributed_load is True:
            self.fx = 0.
            self.fy = self.fy / (self.N + 1)
            self.fz = 0.

        # boundary equations
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

    def create_particles(self):
        if self.clamp is True:
            xp, yp, xw, yw = get_beam_clamp(self.L, self.H,
                                            20. * self.dx_plate,
                                            3. * self.H,
                                            self.wall_layers + 1,
                                            self.dx_plate)
        else:
            xp, yp, xw, yw = get_beam_no_clamp(self.L, self.H,
                                               4. * self.dx_plate,
                                               2. * self.H,
                                               self.wall_layers + 1,
                                               self.dx_plate)

        m = self.plate_rho0 * self.dx_plate**2.

        plate = get_particle_array(x=xp,
                                   y=yp,
                                   m=m,
                                   h=self.h,
                                   rho=self.plate_rho0,
                                   name="plate",
                                   constants={
                                       'n': 4,
                                       'spacing0': self.dx_plate
                                   })

        # create the particle array
        wall = get_particle_array(x=xw,
                                  y=yw,
                                  m=m,
                                  h=self.h,
                                  rho=self.plate_rho0,
                                  name="wall",
                                  constants={
                                      'n': 4,
                                      'spacing0': self.dx_plate,
                                  })
        remove_overlap_particles(wall, plate, self.dx_plate)

        wall.add_property('E')
        wall.add_property('nu')
        plate.add_property('E')
        plate.add_property('nu')
        plate.add_constant('is_sandwich', 0.)
        plate.add_constant('is_two_layer', 0.)

        if self.options.sandwich is True:
            # set Young's modulus
            indices = np.where(plate.y >= 3. * 1e-3)
            plate.E[indices] = self.plate_E

            indices = np.where(plate.y <= -3. * 1e-3)
            plate.E[indices] = self.plate_E

            indices = (plate.y <= 3. * 1e-3) & (plate.y >= -3. * 1e-3)
            plate.E[indices] = self.plate_E / 2.

            plate.nu[:] = self.plate_nu
            plate.is_sandwich[0] = 1.

        elif self.options.two_layer is True:
            # set Young's modulus
            indices = np.where(plate.y >= 0.)
            plate.E[indices] = self.plate_E

            indices = np.where(plate.y < 0.)
            plate.E[indices] = self.plate_E / 2.

            # set Poisson's ratio
            plate.nu[:] = self.plate_nu
            plate.is_two_layer[0] = 1.

        else:
            plate.E[:] = self.plate_E
            plate.nu[:] = self.plate_nu

        # setup properties
        self.scheme.setup_properties([wall, plate])

        if self.gradual_force is True:
            setup_properties_for_gradual_force(plate)
            plate.gradual_force_time[0] = self.gradual_force_time
        else:
            setup_properties_for_sudden_force(plate)

        # find the particle indices where the force have to be applied
        if self.distributed_load is True:
            UDL_force_index_distributed(plate, self.clamp)
        else:
            UDL_force_index_single(plate, self.clamp)

        # add post processing variables.
        find_displacement_index(plate)

        ##################################
        # Add output arrays
        ##################################
        plate.add_output_arrays(['sigma00', 'force_idx', 'is_boundary',
                                 'tip_displacemet_index', 'E', 'G', 'cs'])

        return [plate, wall]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=500)

        self.scheme.configure(edac_nu=self.edac_nu)

    def create_scheme(self):
        solid = SolidsScheme(solids=['plate'],
                             boundaries=['wall'],
                             dim=2,
                             pb=self.pb,
                             edac_nu=self.edac_nu,
                             mach_no=self.mach_no,
                             hdx=self.hdx,
                             gy=0)

        gray = SolidsSchemeGray(solids=['plate'],
                                boundaries=['wall'],
                                dim=2,
                                pb=self.pb,
                                edac_nu=self.edac_nu,
                                mach_no=self.mach_no,
                                hdx=self.hdx,
                                gy=0)

        s = SchemeChooser(default='solid', solid=solid,
                          gray=gray)
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()

        if self.options.integrator == "gtvf":
            # Apply external force
            force_eqs = []
            if self.gradual_force is True:
                force_eqs.append(
                    ApplyForceGradual(dest="plate", sources=None,
                                      delta_fx=self.delta_fx,
                                      delta_fy=self.delta_fy,
                                      delta_fz=self.delta_fz))
            else:
                force_eqs.append(
                    ApplyForceSudden(dest="plate", sources=None,
                                     fx=self.fx,
                                     fy=self.fy,
                                     fz=self.fz))

            eqns.groups[-1].append(Group(force_eqs))

        if self.options.integrator == "rk2":
            # Apply external force
            force_eqs = []
            if self.gradual_force is True:
                force_eqs.append(
                    ApplyForceGradual(dest="plate", sources=None,
                                      delta_fx=self.delta_fx/2.,
                                      delta_fy=self.delta_fy/2.,
                                      delta_fz=self.delta_fz/2.))
            else:
                force_eqs.append(
                    ApplyForceSudden(dest="plate", sources=None,
                                     fx=self.fx/2.,
                                     fy=self.fy/2.,
                                     fz=self.fz/2.))

            eqns.append(Group(force_eqs))

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

    def _plot_force_curve(self):
        files = get_files(self.fname)
        files = files[0::10]
        t = []
        total_force_applied_x = []
        total_force_applied_y = []
        total_force_applied_z = []
        # index = plate

        for sd, plate in iter_output(files, 'plate'):
            _t = sd['t']
            t.append(_t)
            total_force_applied_x.append(plate.total_force_applied_x[0])
            total_force_applied_y.append(plate.total_force_applied_y[0])
            total_force_applied_z.append(plate.total_force_applied_z[0])

            # total_disp_at_mid_point.append(plate.tip_displacemet_index[0])

        import os
        from matplotlib import pyplot as plt

        plt.clf()
        plt.plot(t, total_force_applied_x, "-", label='Simulated')
        plt.xlabel('t')
        plt.ylabel('total applied force in x direction')
        plt.legend()
        fig = os.path.join(os.path.dirname(self.fname),
                           "total_applied_force_in_x_with_t.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, total_force_applied_y, "-", label='Simulated')

        plt.xlabel('t')
        plt.ylabel('total applied force in y direction')
        plt.legend()
        fig = os.path.join(os.path.dirname(self.fname),
                           "total_applied_force_in_y_with_t.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, total_force_applied_z, "-", label='Simulated')

        plt.xlabel('t')
        plt.ylabel('total applied force in z direction')
        plt.legend()
        fig = os.path.join(os.path.dirname(self.fname),
                           "total_applied_force_in_z_with_t.png")
        plt.savefig(fig, dpi=300)

    def post_process(self, fname):
        from pysph.solver.utils import iter_output, load
        from pysph.solver.utils import get_files
        import os

        info = self.read_info(fname)
        files = self.output_files

        data = load(files[0])
        # solver_data = data['solver_data']
        arrays = data['arrays']
        pa = arrays['plate']
        index = np.where(pa.tip_displacemet_index == 1)[0][0]
        y_0 = pa.y[index]

        files = files[0::1]
        # print(len(files))
        t_ctvf, amplitude_ctvf = [], []
        for sd, plate in iter_output(files, 'plate'):
            _t = sd['t']
            t_ctvf.append(_t)
            print(_t)
            amplitude_ctvf.append((plate.y[index] - y_0))

        # matplotlib.use('Agg')
        # analytical solution
        t_analytical = np.linspace(0., max(t_ctvf), 1000)
        if plate.is_sandwich[0] == 1.:
            amplitude_analytical = -2.5165 * 1e-5 * np.ones_like(t_analytical)
            amplitude_khayyer = -2.8958 * 1e-5 * np.ones_like(t_analytical)
        elif plate.is_two_layer[0] == 1.:
            amplitude_analytical = -3.4108 * 1e-5 * np.ones_like(t_analytical)
            amplitude_khayyer = -3.8145 * 1e-5 * np.ones_like(t_analytical)
        else:
            amplitude_analytical = -2.41 * 1e-5 * np.ones_like(t_analytical)
            amplitude_khayyer = -2.707 * 1e-5 * np.ones_like(t_analytical)

        res = os.path.join(os.path.dirname(fname), "results.npz")
        # res = os.path.join(fname, "results.npz")
        np.savez(res, amplitude_analytical=amplitude_analytical,
                 t_analytical=t_analytical,
                 amplitude_khayyer=amplitude_khayyer,
                 amplitude_ctvf=amplitude_ctvf,
                 t_ctvf=t_ctvf)

        from matplotlib import pyplot as plt

        plt.clf()
        plt.plot(t_analytical, amplitude_analytical, label='analytical')
        plt.plot(t_analytical, amplitude_khayyer, label='Khayyer (d=1.0E-3 m)')
        if plate.is_sandwich[0] == 1.:
            plt.plot(t_ctvf, amplitude_ctvf, "-", label='Simulated Sandwich')
        elif plate.is_two_layer[0] == 1.:
            plt.plot(t_ctvf, amplitude_ctvf, "-", label='Simulated Two layers')
        else:
            plt.plot(t_ctvf, amplitude_ctvf, "-", label='Simulated Homogeneous')

        plt.xlabel('t')
        plt.ylabel('amplitude')
        # plt.ylim(-5 * 1e-5, 0.0)
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "amplitude_with_t.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = ClampedElasticPlateUDL()
    app.run()
    app.post_process(app.info_filename)
