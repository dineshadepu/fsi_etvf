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

from solid_mech import SetHIJForInsideParticles, SolidsScheme

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)
from pysph.tools.geometry import get_2d_tank, get_2d_block
from solid_mech_common import AddGravityToStructure
from pysph.sph.scheme import add_bool_argument


def get_fixed_beam(beam_length, beam_height, beam_inside_length,
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
    import matplotlib.pyplot as plt
    # create a block first
    xb, yb = get_2d_block(dx=spacing,
                          length=beam_length + beam_inside_length,
                          height=beam_height)

    # create a (support) block with required number of layers
    xs1, ys1 = get_2d_block(dx=spacing,
                            length=beam_inside_length,
                            height=boundary_layers * spacing)
    xs1 += np.min(xb) - np.min(xs1)
    ys1 += np.min(yb) - np.max(ys1) - spacing

    # create a (support) block with required number of layers
    xs2, ys2 = get_2d_block(dx=spacing,
                            length=beam_inside_length,
                            height=boundary_layers * spacing)
    xs2 += np.min(xb) - np.min(xs2)
    ys2 += np.max(ys2) - np.min(yb) + spacing

    xs = np.concatenate([xs1, xs2])
    ys = np.concatenate([ys1, ys2])

    xs3, ys3 = get_2d_block(dx=spacing,
                            length=boundary_layers * spacing,
                            height=np.max(ys) - np.min(ys))
    xs3 += np.min(xb) - np.max(xs3) - 1. * spacing
    # ys3 += np.max(ys2) - np.min(yb) + spacing

    xs = np.concatenate([xs, xs3])
    ys = np.concatenate([ys, ys3])
    # plt.scatter(xs, ys)
    # plt.scatter(xb, yb)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.show()

    return xb, yb, xs, ys


def get_fixed_beam_no_clamp(beam_length, beam_height, beam_inside_length,
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
                          length=beam_length,
                          height=beam_height)

    # create a (support) block with required number of layers
    xs1, ys1 = get_2d_block(dx=spacing,
                            length=boundary_layers*spacing,
                            height=beam_height + beam_height)

    xs2, ys2 = get_2d_block(dx=spacing,
                            length=boundary_layers*spacing,
                            height=beam_height + beam_height)

    xs1 -= np.max(xs1) - np.min(xb) + spacing
    xs2 += np.max(xb) - np.min(xs2) + spacing

    xs = np.concatenate([xs1, xs2])
    ys = np.concatenate([ys1, ys2])

    return xb, yb, xs, ys


def get_fixed_beam_clamp(beam_length, beam_height, beam_inside_length,
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
                          length=beam_length + 2. * beam_inside_length,
                          height=beam_height)

    # create the left (support) block
    xs1, ys1 = get_2d_block(dx=spacing,
                            length=beam_inside_length - spacing,
                            height=boundary_layers*spacing)
    xs1 -= np.min(xs1) - np.min(xb)
    ys1 += np.max(yb) - np.min(ys1) + spacing

    xs2, ys2 = get_2d_block(dx=spacing,
                            length=beam_inside_length - spacing,
                            height=boundary_layers*spacing)
    xs2 -= np.min(xs2) - np.min(xb)
    ys2 -= np.max(ys2) - np.min(yb) + spacing

    xs5, ys5 = get_2d_block(dx=spacing,
                            length=boundary_layers*spacing,
                            height=np.max(ys1) - np.min(ys2))
    xs5 -= np.max(xs5) - np.min(xb) + spacing

    # create a right (support) block
    xs3, ys3 = get_2d_block(dx=spacing,
                            length=beam_inside_length - spacing,
                            height=boundary_layers*spacing)
    xs3 += np.max(xb) - np.max(xs3)
    ys3 += np.max(yb) - np.min(ys3) + spacing

    xs4, ys4 = get_2d_block(dx=spacing,
                            length=beam_inside_length - spacing,
                            height=boundary_layers*spacing)
    xs4 += np.max(xb) - np.max(xs4)
    ys4 -= np.max(ys4) - np.min(yb) + spacing

    xs6, ys6 = get_2d_block(dx=spacing,
                            length=boundary_layers*spacing,
                            height=np.max(ys1) - np.min(ys2))
    xs6 += np.max(xb) - np.min(xs6) + spacing

    xs = np.concatenate([xs1, xs2, xs3, xs4, xs5, xs6])
    ys = np.concatenate([ys1, ys2, ys3, ys4, ys5, ys6])

    return xb, yb, xs, ys


class ApplyForceGradual(Equation):
    def __init__(self, dest, sources, delta_fx, delta_fy, delta_fz):
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


class ApplyForce(Equation):
    def __init__(self, dest, sources, fx=0, fy=0, fz=0):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        super(ApplyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_force_idx, d_m, t):
        if d_force_idx[d_idx] == 1:
            d_au[d_idx] += self.fx / d_m[d_idx]
            d_av[d_idx] += self.fy / d_m[d_idx]
            d_aw[d_idx] += self.fz / d_m[d_idx]


class BeamUnderUDF(Application):
    def initialize(self):
        # dummy value to make the scheme work
        self.plate_rho0 = 2700.
        self.plate_E = 67.5 * 1e9
        self.plate_nu = 0.34
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
        self.u_max = 2.8513
        self.mach_no = self.u_max / self.c0

        self.cylinder_r = 0.05

        # for pre step
        self.seval = None

        # boundary equations
        # self.boundary_equations = get_boundary_identification_etvf_equations(
            # destinations=["plate"], sources=["plate"])
        self.boundary_equations = get_boundary_identification_etvf_equations(
            destinations=["plate"],
            sources=["plate", "wall"],
            boundaries=["wall"])

    def add_user_options(self, group):
        group.add_argument("--rho",
                           action="store",
                           type=float,
                           dest="rho",
                           default=1000.,
                           help="Density of the particle (Defaults to 1000.)")

        group.add_argument("--length",
                           action="store",
                           type=float,
                           dest="length",
                           default=1.,
                           help="Length of the plate")

        group.add_argument("--height",
                           action="store",
                           type=float,
                           dest="height",
                           default=0.05,
                           help="height of the plate")

        group.add_argument("--N",
                           action="store",
                           type=int,
                           dest="N",
                           default=25,
                           help="No of particles in the height direction")

        add_bool_argument(group, 'gradual-force', dest='gradual_force',
                          default=False, help='Apply gradual force')

        add_bool_argument(group, 'clamp', dest='clamp',
                          default=True, help='Clamped beam')

        group.add_argument("--clamp-factor",
                           action="store",
                           type=float,
                           dest="clamp_factor",
                           default=2.5,
                           help="Amount of beam to be clamped")

    def consume_user_options(self):
        self.rho = self.options.rho
        self.L = self.options.length
        self.H = self.options.height
        self.N = self.options.N

        self.dx_plate = 0.05 / 5
        # print("dx_plate[ is ]")
        # print(self.dx_plate)
        # self.fac = self.dx_plate / 2.
        self.h = self.hdx * self.dx_plate
        # print(self.h)
        self.plate_rho0 = self.rho

        self.wall_layers = 2

        # compute the timestep
        self.tf = 1.0
        self.dt = 0.25 * self.h / (
            (self.plate_E / self.plate_rho0)**0.5 + 2.85)
        # self.dt = 0.5 * self.h / (
        #     (self.plate_E / self.plate_rho0)**0.5 + 2.85)
        # self.dt = 0.25 * self.h / (
        #     (self.plate_E / self.plate_rho0)**0.5 + 2.85)

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

        self.clamp = self.options.clamp
        self.clamp_factor = self.options.clamp_factor

        pressure = 1000. * 9.81 * 2.
        area = self.dx_plate
        self.fx = 0.
        self.fy = -pressure * area
        self.fz = 0.

        # the incremental force to be applied
        self.final_force_time = 0.1
        timesteps = self.final_force_time / self.dt

        # now distribute such force over a group of particles
        self.delta_fx = self.fx / timesteps
        self.delta_fy = self.fy / timesteps
        self.delta_fz = self.fz / timesteps
        self.gradual_force = self.options.gradual_force

    def create_particles(self):
        if self.clamp is True:
            xp, yp, xw, yw = get_fixed_beam_clamp(self.L, self.H, self.L/2.5,
                                                  self.wall_layers + 1, self.dx_plate)
        else:
            xp, yp, xw, yw = get_fixed_beam_no_clamp(self.L, self.H, self.L/6.,
                                                     self.wall_layers + 1, self.dx_plate)

        # make sure that the beam intersection with wall starts at the 0.
        # xw -= max(xw) - min(xp) + self.dx_plate

        m = self.plate_rho0 * self.dx_plate**2.

        plate = get_particle_array(x=xp,
                                   y=yp,
                                   m=m,
                                   h=self.h,
                                   rho=self.plate_rho0,
                                   name="plate",
                                   constants={
                                       'E': self.plate_E,
                                       'n': 4,
                                       'nu': self.plate_nu,
                                       'spacing0': self.dx_plate,
                                       'rho_ref': self.plate_rho0
                                   })
        # ============================================ #
        # find all the indices which are at the top
        # ============================================ #
        max_y = max(yp)
        indices = np.where((yp > max_y - self.dx_plate / 2.) & (xp <= 0.5) & (xp >= -0.5))
        # indices1 = np.where(plate.x < 0.5)
        # indices2 = np.where(plate.x > - -0.5)
        # indices = np.intersect1d(indices0, indices1, indices2)
        # print(indices)

        # print(indices[0])
        force_idx = np.zeros_like(plate.x)
        plate.add_property('force_idx', type='int', data=force_idx)
        # print(beam.zero_force_idx)
        plate.force_idx[indices] = 1

        plate.add_constant('final_force_time',
                           np.array([self.final_force_time]))
        plate.add_constant('total_force_applied_x', np.array([self.delta_fx]))
        plate.add_constant('total_force_applied_y', np.array([self.delta_fy]))
        plate.add_constant('total_force_applied_z', np.array([self.delta_fz]))

        # create the particle array
        wall = get_particle_array(x=xw,
                                  y=yw,
                                  m=m,
                                  h=self.h,
                                  rho=self.plate_rho0,
                                  name="wall",
                                  constants={
                                      'E': self.plate_E,
                                      'n': 4,
                                      'nu': self.plate_nu,
                                      'spacing0': self.dx_plate,
                                      'rho_ref': self.plate_rho0
                                  })

        self.scheme.setup_properties([wall, plate])

        xp_max = max(xp)
        fltr = np.argwhere(xp == xp_max)
        fltr_idx = int(len(fltr) / 2.)
        amplitude_idx = fltr[fltr_idx][0]

        plate.add_constant("amplitude_idx", amplitude_idx)

        if self.clamp is True:
            if self.N == 25:
                plate.amplitude_idx[0] = 2700
                if self.clamp_factor == 8:
                    print("here")
                    plate.amplitude_idx[0] = 2161

        ##################################
        # Add output arrays
        ##################################
        plate.add_output_arrays(['sigma00'])
        plate.add_output_arrays(['force_idx'])

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
                             gy=-9.81)

        s = SchemeChooser(default='solid', solid=solid)
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # Apply external force
        force_eqs = []
        if self.gradual_force is True:
            force_eqs.append(
                ApplyForceGradual("plate", sources=None,
                                  delta_fx=self.delta_fx,
                                  delta_fy=self.delta_fy,
                                  delta_fz=self.delta_fz))
        else:
            force_eqs.append(
                ApplyForce("plate", sources=None, fx=self.fx, fy=self.fy,
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
        if self.options.pst in ['sun2019', 'ipst']:
            if solver.count % 1 == 0:
                t = solver.t
                dt = solver.dt

                arrays = self.particles
                a_eval = self._make_accel_eval(self.boundary_equations, arrays)

                # When
                a_eval.evaluate(t, dt)

    def post_process(self, fname):
        from pysph.solver.utils import iter_output
        from pysph.solver.utils import get_files, load

        files = get_files(fname)

        idx = 290

        # initial position of the gate
        # index = 479
        data = load(files[0])
        arrays = data['arrays']
        plate = arrays['plate']
        y_initial = plate.y[idx]
        x_initial = plate.x[idx]
        t, y_amplitude = [], []

        for sd, array in iter_output(files[::3], 'plate'):
            _t = sd['t']

            if _t < 0.3:
                t.append(_t)
                y_amplitude.append(array.y[idx] - y_initial)

        import matplotlib
        import os
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt

        if "info" in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        # np.savez(res, t=t,  x_ampiltude=x_amplitude, y_ampiltude=y_amplitude)

        # gtvf data
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # data = np.loadtxt(os.path.join(directory, 'turek_fem_y_data.csv'),
        #                   delimiter=',')
        # t_fem, amplitude_fem = data[:, 0], data[:, 1]

        # np.savez(res, t=t,  x_ampiltude=x_amplitude, y_amplitude=y_amplitude,
        #          t_fem=t_fem,  y_amplitude_fem=amplitude_fem)

        plt.clf()

        t_exact = np.linspace(0., 0.7, 1000)
        exact_sol = - np.ones_like(t_exact) * 7.047 * 1e-5
        plt.plot(t_exact, exact_sol, label='exact')
        plt.plot(t, y_amplitude, "-r", label='Simulated')

        # print("heeee haaaa")
        plt.xlabel('t')
        plt.ylabel('Amplitude')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "amplitude.png")
        # print(fig)
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = BeamUnderUDF()
    app.run()
    app.post_process(app.info_filename)
    # app.create_rings_geometry()
