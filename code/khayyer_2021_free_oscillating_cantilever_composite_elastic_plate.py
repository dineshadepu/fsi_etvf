import numpy as np
from numpy import cos, sin, sinh, cosh

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
                                     UDL_force_index_distributed,
                                     UDL_force_index_single,
                                     tip_load_force_index_distributed,
                                     tip_load_force_index_single)
from pysph.solver.utils import iter_output, load
from pysph.solver.utils import get_files


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
    """
    TODO
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


def get_beam_no_clamp(beam_length, beam_height, boundary_length,
                      boundary_height, boundary_layers, spacing):
    # create a block first
    xb, yb = get_2d_block(dx=spacing,
                          length=beam_length,
                          height=beam_height)

    xs, ys = get_2d_block(dx=spacing,
                          length=boundary_length,
                          height=boundary_height)

    # move the beam to the front of the wall
    # xb += max(xb) - min(xs1) + spacing
    xs -= max(xs) - min(xb) + spacing
    return xb, yb, xs, ys


def find_displacement_index(pa):
    x = pa.x
    max_x = max(x)
    max_x_indices = np.where(x == max_x)[0]
    index = max_x_indices[int(len(max_x_indices)/2)]
    pa.add_property('tip_displacemet_index', type='int',
                    data=np.zeros(len(pa.x)))
    pa.tip_displacemet_index[index] = 1


class OscillatingPlate(Application):
    def initialize(self):
        # dummy value to make the scheme work
        self.plate_rho0 = 1000.
        self.plate_E = 2.0 * 1e6
        self.plate_nu = 0.4
        self.c0 = get_speed_of_sound(self.plate_E, self.plate_nu,
                                     self.plate_rho0)
        self.pb = self.plate_rho0 * self.c0**2

        self.edac_alpha = 0.5
        self.hdx = 1.2

        # this is dummpy value
        self.h = 0.001
        self.edac_nu = self.edac_alpha * self.c0 * self.h / 8

        # attributes for Sun PST technique
        # dummy value, will be updated in consume user options
        self.u_max = 2.85
        self.mach_no = self.u_max / self.c0

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

        add_bool_argument(group, 'sandwich', dest='sandwich',
                          default=False, help='Assume sandwich plate')

        add_bool_argument(group, 'two-layer', dest='two_layer',
                          default=False, help='Assume two layer composite')

    def consume_user_options(self):
        self.N = self.options.N
        self.clamp = self.options.clamp
        self.clamp_factor = self.options.clamp_factor
        self.sandwich = self.options.sandwich
        self.two_layer = self.options.two_layer

        self.rho = 1000.
        self.L = 0.2
        self.H = 0.02

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
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0 * self.h / 8

        self.clamp = self.options.clamp
        self.wall_pst = self.options.wall_pst
        self.clamp_factor = self.options.clamp_factor

        # boundary equations
        if self.wall_pst is True:
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
                                            4. * self.dx_plate,
                                            2. * self.H,
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
                                       'spacing0': self.dx_plate,
                                       'rho_ref': self.plate_rho0
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
                                      'rho_ref': self.plate_rho0
                                  })

        # translate the model
        scale = - min(plate.x)
        plate.x[:] += scale
        wall.x[:] += scale

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
            indices = np.where(plate.y > 0.)
            plate.E[indices] = self.plate_E

            indices = np.where(plate.y < 0.)
            plate.E[indices] = self.plate_E / 2.

            # set Poisson's ratio
            plate.nu[:] = self.plate_nu
            plate.is_two_layer[0] = 1.

        else:
            plate.E[:] = self.plate_E
            plate.nu[:] = self.plate_nu

        self.scheme.setup_properties([wall, plate])

        # add post processing variables.
        find_displacement_index(plate)

        ##################################
        # vertical velocity of the plate #
        ##################################
        # initialize with zero at the beginning
        v = np.zeros_like(xp)
        v = v.ravel()

        # set the vertical velocity for particles which are only
        # out of the wall
        self.KL = 1.875
        self.K = 1.875 / self.L
        K = self.K
        KL = self.KL
        M = sin(KL) + sinh(KL)
        N = cos(KL) + cosh(KL)
        Q = 2 * (cos(KL) * sinh(KL) - sin(KL) * cosh(KL))

        x = plate.x
        fltr = plate.x >= 0
        tmp1 = (cos(K * x[fltr]) - cosh(K * x[fltr]))
        tmp2 = (sin(K * x[fltr]) - sinh(K * x[fltr]))
        v[fltr] = 0.01 * self.c0 * (M * tmp1 - N * tmp2) / Q

        # set vertical velocity
        plate.v = v

        ##################################
        # Add output arrays
        ##################################
        plate.add_output_arrays(['sigma00', 'is_boundary',
                                 'tip_displacemet_index',
                                 'E', 'G', 'cs'])

        return [plate, wall]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=500)

    def create_scheme(self):
        solid = SolidsScheme(solids=['plate'],
                             boundaries=['wall'],
                             dim=2,
                             pb=self.pb,
                             edac_nu=self.edac_nu,
                             mach_no=self.mach_no,
                             hdx=self.hdx,
                             gy=0)

        s = SchemeChooser(default='solid', solid=solid)
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()

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
        np.savez(res, ampiltude_analytical=amplitude_analytical,
                 t_analytical=t_analytical,
                 amplitude_khayyer=amplitude_khayyer,
                 amplitude_ctvf=amplitude_ctvf,
                 t_ctvf=t_ctvf)

        from matplotlib import pyplot as plt

        plt.clf()
        plt.plot(t_analytical, amplitude_analytical, label='analytical')
        plt.plot(t_analytical, amplitude_khayyer, label='Khayyer (d=1.0E-3 m)')
        plt.plot(t_ctvf, amplitude_ctvf, "-", label='Simulated')

        plt.xlabel('t')
        plt.ylabel('amplitude')
        # plt.ylim(-5 * 1e-5, 0.0)
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "amplitude_with_t.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = OscillatingPlate()
    app.run()
    app.post_process(app.info_filename)
