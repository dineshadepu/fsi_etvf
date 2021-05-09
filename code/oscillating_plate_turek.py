import numpy as np
from math import cos, sin, sinh, cosh

# SPH equations
from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.scheme import SchemeChooser

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
    xs, ys = get_2d_block(dx=spacing,
                          length=boundary_layers*spacing,
                          height=beam_height + beam_height)

    return xb, yb, xs, ys


class OscillatingPlate(Application):
    def initialize(self):
        # dummy value to make the scheme work
        self.plate_rho0 = 1000.
        self.plate_E = 1.4 * 1e6
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
                           default=0.35,
                           help="Length of the plate")

        group.add_argument("--height",
                           action="store",
                           type=float,
                           dest="height",
                           default=0.02,
                           help="height of the plate")

        group.add_argument("--N",
                           action="store",
                           type=int,
                           dest="N",
                           default=25,
                           help="No of particles in the height direction")

        add_bool_argument(group, 'clamp', dest='clamp',
                          default=True, help='Clamped beam')

        group.add_argument("--clamped",
                           action="store",
                           type=int,
                           dest="N",
                           default=25,
                           help="No of particles in the height direction")

    def consume_user_options(self):
        self.rho = self.options.rho
        self.L = self.options.length
        self.H = self.options.height
        self.N = self.options.N

        self.dx_plate = self.cylinder_r / self.N
        # print("dx_plate[ is ]")
        # print(self.dx_plate)
        # self.fac = self.dx_plate / 2.
        self.h = self.hdx * self.dx_plate
        # print(self.h)
        self.plate_rho0 = self.rho

        self.wall_layers = 2

        # compute the timestep
        self.tf = 10.0
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

    def create_particles(self):
        if self.clamp is True:
            xp, yp, xw, yw = get_fixed_beam(self.L, self.H, self.L/2.5,
                                            self.wall_layers, self.dx_plate)
            # make sure that the beam intersection with wall starts at the 0.
            min_xp = np.min(xp)

            # add this to the beam and wall
            xp += abs(min_xp)
            xw += abs(min_xp)

            max_xw = np.max(xw)
            xp -= abs(max_xw)
            xw -= abs(max_xw)

        else:
            xp, yp, xw, yw = get_fixed_beam_no_clamp(self.L, self.H, self.L/2.5,
                                                     self.wall_layers +1, self.dx_plate)
            # make sure that the beam intersection with wall starts at the 0.
            xw -= max(xw) - min(xp) + self.dx_plate

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

        ##################################
        # Add output arrays
        ##################################
        plate.add_output_arrays(['sigma00'])

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
                             gy=-2)

        s = SchemeChooser(default='solid', solid=solid)
        return s

    # def create_equations(self):
    #     eqns = self.scheme.get_equations()

    #     g9 = []
    #     g9.append(AddGravityToStructure(dest="plate", sources=None,
    #                                     gx=0.,
    #                                     gy=2.,
    #                                     gz=0.))

    #     # equation = eqns.groups[-1][4].equations[3]
    #     eqns.groups[-1].append(Group(equations=g9))
    #     print(eqns)

    #     return eqns

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
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        t, x_amplitude, y_amplitude = [], [], []
        for sd, array in iter_output(files, 'plate'):
            _t = sd['t']
            t.append(_t)
            x_amplitude.append(array.x[array.amplitude_idx[0]])
            y_amplitude.append(array.y[array.amplitude_idx[0]])

        import matplotlib
        import os
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res, t=t, x_amplitude=x_amplitude, y_amplitude=y_amplitude)

        # gtvf data
        # path = os.path.abspath(__file__)
        # directory = os.path.dirname(path)

        # data = np.loadtxt(os.path.join(directory, 'oscillating_plate.csv'), delimiter=',')
        # t_gtvf, amplitude_gtvf = data[:, 0], data[:, 1]

        # plt.clf()

        # plt.plot(t_gtvf, amplitude_gtvf, "s-", label='GTVF Paper')
        plt.plot(t, y_amplitude, "-", label='Simulated')

        # print("heeee haaaa")
        plt.xlabel('t')
        plt.ylabel('Amplitude')
        plt.legend()
        fig = os.path.join(self.output_dir, "amplitude.png")
        print(fig)
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = OscillatingPlate()
    app.run()
    app.post_process(app.info_filename)
    # app.create_rings_geometry()
