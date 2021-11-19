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


def find_displacement_index(pa):
    x = pa.x
    y = pa.y
    max_x = max(x)
    max_x_indices = np.where(x == max_x)[0]
    index = max_x_indices[int(len(max_x_indices)/2)]
    pa.add_property('tip_displacemet_index', type='int',
                    data=np.zeros(len(pa.x)))
    pa.tip_displacemet_index[index] = 1
    pa.add_output_arrays(['tip_displacemet_index'])


class OscillatingPlate(Application):
    def add_user_options(self, group):
        group.add_argument("--N",
                           action="store",
                           type=int,
                           dest="N",
                           default=25,
                           help="No of particles in the height direction")

        add_bool_argument(group, 'clamp', dest='clamp',
                          default=True, help='Clamped beam')

        group.add_argument("--clamp-factor",
                           action="store",
                           type=float,
                           dest="clamp_factor",
                           default=2.5,
                           help="Amount of beam to be clamped")

    def consume_user_options(self):
        self.N = self.options.N
        self.clamp = self.options.clamp
        self.clamp_factor = self.options.clamp_factor

        # =============================
        # general simulation parameters
        # =============================
        self.hdx = 1.2
        self.dim = 2
        self.seval = None  # for pre step
        self.gx = 0.
        self.gy = -2.
        self.gz = 0.

        # ====================
        # structure properties
        # ====================
        self.plate_rho0 = 1000.
        self.plate_E = 1.4 * 1e6
        self.plate_nu = 0.4
        # attributes for Sun PST technique
        self.c0 = get_speed_of_sound(self.plate_E, self.plate_nu,
                                     self.plate_rho0)
        self.pb = self.plate_rho0 * self.c0**2
        self.u_max = 2.8513
        self.mach_no = self.u_max / self.c0
        self.L = 0.35
        self.H = 0.02
        self.dx_plate = self.H / self.N
        self.h = self.hdx * self.dx_plate
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

        # ===============
        # wall properties
        # ===============
        self.wall_layers = 2

        # compute the timestep
        self.tf = 10.
        self.dt = 0.25 * self.h / (
            (self.plate_E / self.plate_rho0)**0.5 + 2.85)

    def create_particles(self):
        if self.clamp is True:
            xp, yp, xw, yw = get_fixed_beam(self.L, self.H, self.L/self.clamp_factor,
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

        # add post processing variables.
        find_displacement_index(plate)

        ##################################
        # Add output arrays
        ##################################
        plate.add_output_arrays(['sigma00'])

        return [plate, wall]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=500)

        scheme = self.scheme
        scheme.configure(mach_no=self.mach_no, gy=self.gy)

    def create_scheme(self):
        solid = SolidsScheme(solids=['plate'],
                             boundaries=['wall'],
                             dim=2,
                             mach_no=1.,
                             gy=0.)

        s = SchemeChooser(default='solid', solid=solid)
        return s

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
        print("index is", index)
        y_0 = pa.y[index]

        files = files[0::1]
        # print(len(files))
        t_ctvf, amplitude_ctvf = [], []
        for sd, plate in iter_output(files, 'plate'):
            _t = sd['t']
            t_ctvf.append(_t)
            amplitude_ctvf.append((plate.y[index] - y_0))

        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)
        data = np.loadtxt(os.path.join(directory, 'turek_fem_y_data.csv'),
                          delimiter=',')
        t_fem, amplitude_fem = data[:, 0], data[:, 1]

        res = os.path.join(os.path.dirname(fname), "results.npz")
        # res = os.path.join(fname, "results.npz")
        np.savez(res, amplitude_ctvf=amplitude_ctvf, t_ctvf=t_ctvf,
                 t_fem=t_fem, y_amplitude_fem=amplitude_fem)

        from matplotlib import pyplot as plt

        plt.clf()
        plt.scatter(t_fem, amplitude_fem, label='analytical')
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
    # app.create_rings_geometry()
