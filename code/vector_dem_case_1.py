"""Case 1 of two particles bonded under axial force
"""
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

from vector_based_dem import VectorBasedDEM

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.sph.scheme import add_bool_argument
from force_application_utils import (setup_properties_for_gradual_force,
                                     setup_properties_for_sudden_force)


class ApplyForceGradual(Equation):
    def __init__(self, dest, sources, delta_fx, delta_fy, delta_fz):
        self.delta_fx = delta_fx
        self.delta_fy = delta_fy
        self.delta_fz = delta_fz
        super(ApplyForceGradual, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz, d_gradual_force_time,
                   d_total_force_applied_x, d_total_force_applied_y,
                   d_total_force_applied_z, d_force_idx, d_m, t):
        if t < d_gradual_force_time[0]:
            if d_idx == 0:
                d_total_force_applied_x[0] += self.delta_fx
                d_total_force_applied_y[0] += self.delta_fy
                d_total_force_applied_z[0] += self.delta_fz

        if d_force_idx[d_idx] == 1:
            d_fx[d_idx] += d_total_force_applied_x[0]
            d_fy[d_idx] += d_total_force_applied_y[0]
            d_fz[d_idx] += d_total_force_applied_z[0]


class ApplyForceSudden(Equation):
    def __init__(self, dest, sources, fx, fy, fz):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        super(ApplyForceSudden, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz, d_force_idx, d_m, t):
        if d_force_idx[d_idx] == 1:
            d_fx[d_idx] += self.fx
            d_fy[d_idx] += self.fy
            d_fz[d_idx] += self.fz


def tip_load_force_index_single(plate):
    max_y = np.max(plate.y)
    max_x = np.max(plate.x)
    indices_1 = np.where(max_x == plate.x)[0]
    max_x = np.max(plate.x)

    for i in indices_1:
        if plate.y[i] == max_y:
            break
    plate.force_idx[i] = 1


def tip_load_force_index_distributed(plate):
    max_x = np.max(plate.x)
    indices = np.where(max_x == plate.x)[0]

    plate.force_idx[indices] = 1


def get_fixed_beam(beam_length, beam_height, spacing):
    """


    ======================================================|
    ======================================================|Beam
    ======================================================|height


    <---------------------------------------------------->
                      Beam length

    """
    x = np.array([])
    y = np.array([])

    _y = np.arange(0., beam_height, spacing)

    y_tmp = 0.
    for i in range(len(_y)):
        x_layer = np.arange(0., beam_length+spacing/2., spacing)
        y_layer = np.ones_like(x_layer) * y_tmp

        if i % 2 == 1:
            x_layer += spacing / 2.

        x = np.concatenate([x, x_layer])
        y = np.concatenate([y, y_layer])

        y_tmp += np.sqrt(3) * spacing / 2.

    return x, y


def find_displacement_index(pa):
    x = pa.x
    max_x = max(x)
    max_x_indices = np.where(x == max_x)[0]
    index = max_x_indices[int(len(max_x_indices)/2)]
    pa.add_property('tip_displacemet_index', type='int',
                    data=np.zeros(len(pa.x)))
    pa.tip_displacemet_index[index] = 1


def set_boundary_particles(plate):
    min_x = np.min(plate.x)
    indices = np.where(min_x == plate.x)[0]
    plate.boundary_particle[indices] = 1


class StaticCantileverBeamUnderTipLoad(Application):
    def initialize(self):
        # dummy value to make the scheme work
        self.plate_rho0 = 100.
        self.plate_E = 10. * 1e6
        self.plate_nu = 0.3
        self.c0 = get_speed_of_sound(self.plate_E, self.plate_nu,
                                     self.plate_rho0)

    def add_user_options(self, group):
        group.add_argument("--Nx",
                           action="store",
                           type=int,
                           dest="Nx",
                           default=40,
                           help="No of particles in the length direction")

        group.add_argument("--Ny",
                           action="store",
                           type=int,
                           dest="Ny",
                           default=3,
                           help="No of particles in the height direction")

        add_bool_argument(group, 'gradual-force', dest='gradual_force',
                          default=True, help='Apply gradual force')

        group.add_argument("--gradual-force-time",
                           action="store",
                           type=float,
                           dest="gradual_force_time",
                           default=0.1,
                           help="Total time of gradual force")

        add_bool_argument(group, 'distributed-load', dest='distributed_load',
                          default=False, help='Apply load to a set of particles')

    def consume_user_options(self):
        self.Nx = self.options.Nx
        self.gradual_force = self.options.gradual_force
        self.gradual_force_time = self.options.gradual_force_time
        self.distributed_load = self.options.distributed_load

        self.rho = 100.
        self.L = 1.
        self.H = 0.075

        self.dx_plate = 0.025
        self.h = 1 * self.dx_plate
        self.plate_rho0 = self.rho
        m = self.plate_rho0 * np.pi * (self.dx_plate/2)**2.

        # compute the timestep
        self.tf = 10.
        self.dt = 0.25 * self.h / (
            (self.plate_E / self.plate_rho0)**0.5 + 2.85)

        self.dim = 2

        self.fx = 100.
        self.fy = 0.
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
            self.delta_fx = self.step_force_x / (self.Nx + 1)
            self.delta_fy = self.step_force_y / (self.Nx + 1)
            self.delta_fz = self.step_force_z / (self.Nx + 1)

        if self.distributed_load is True:
            self.fx = self.fx / (self.Nx + 1)
            self.fy = self.fy / (self.Nx + 1)
            self.fz = 0.

        # compute the time step
        self.dt = 0.05 * np.sqrt(m / self.plate_E)
        print("timestep is", self.dt)

    def create_particles(self):
        xp = np.array([0., 0.025])
        yp = np.array([0., 0.])

        rad = (self.dx_plate)/2.
        # Need to check this mass
        m = np.pi * self.plate_rho0 * rad**2.
        moi = 0.4 * m * rad**2.

        plate = get_particle_array(x=xp,
                                   y=yp,
                                   m=m,
                                   moi=moi,
                                   h=self.h,
                                   rho=self.plate_rho0,
                                   rad_s=self.dx_plate/2.,
                                   name="plate",
                                   constants={
                                       'E': self.plate_E,
                                       'nu': self.plate_nu,
                                   })
        # print("total mass is", np.sum(plate.m))
        self.scheme.setup_properties([plate])

        # mark boundary particles
        set_boundary_particles(plate)

        ##################################
        # Apply the force
        ##################################
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
        ##################################
        # Apply the force ends.
        ##################################

        ##################################
        # add post processing variables.
        ##################################
        find_displacement_index(plate)
        ######################################
        # add post processing variables ends
        ######################################

        ##################################
        # Add output arrays
        ##################################
        plate.add_output_arrays(['force_idx', 'tip_displacemet_index'])

        return [plate]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=500)

    def create_scheme(self):
        vector = VectorBasedDEM(solids=['plate'], dim=2, gy=0)

        s = SchemeChooser(default='vector', vector=vector)
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

    # def customize_output(self):
    #     self._mayavi_config('''
    #     b = particle_arrays['plate']
    #     b.plot.glyph.glyph_source.glyph_source = b.plot.glyph.glyph_source.glyph_dict['sphere_source']
    #     b.plot.glyph.glyph_source.glyph_source.radius = {s_rad}
    #     b.scalar = 'fy'
    #     '''.format(s_rad=self.dx_plate/2.))


if __name__ == '__main__':
    app = StaticCantileverBeamUnderTipLoad()
    app.run()
    app.post_process(app.info_filename)
