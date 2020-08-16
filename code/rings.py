"""Dam break solved with DTSPH.

python dam_break_2d.py --openmp --integrator gtvf --no-internal-flow --pst sun2019 --no-set-solid-vel-project-uhat --no-set-uhat-solid-vel-to-u --no-vol-evol-solid --no-edac-solid --surf-p-zero -d dam_break_2d_etvf_integrator_gtvf_pst_sun2019_output --pfreq 1 --detailed-output


"""
import numpy as np

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.base.utils import get_particle_array

from pysph.examples import cavity as LDC
from pysph.sph.equation import Equation, Group
# from fluids import ETVFScheme
# from solid_mech import ComputeAuHatETVF

from pysph.sph.solid_mech.basic import (get_particle_array_elastic_dynamics,
                                        ElasticSolidsScheme)
from pysph.base.kernels import CubicSpline
from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import SchemeChooser
from pysph.tools.geometry import rotate

# from boundary_particles import (add_boundary_identification_properties,
#                                 get_boundary_identification_etvf_equations)
from fluid_structure_interaction import FSIScheme


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


class Rings(Application):
    def initialize(self):
        # constants
        self.E = 1e7
        self.nu = 0.3975
        self.rho0 = 1.2 * 1e3

        self.dx = 0.001
        self.hdx = 1.3
        self.h = self.hdx * self.dx
        self.fac = self.dx / 2.
        self.kernel_fac = 3

        # geometry
        self.ri = 0.03
        self.ro = 0.04

        self.spacing = 0.041
        self.dim = 2

        # compute the timestep
        self.dt = 0.25 * self.h / ((self.E / self.rho0)**0.5 + 2.85)
        print(self.dt)

        # self.tf = 6.0000000000007015e-05
        self.tf = 1.18 * 1e-2

        self.c0 = np.sqrt(self.E / (3 * (1. - 2 * self.nu) * self.rho0))
        self.pb = self.rho0 * self.c0**2.

        self.artificial_vis_alpha = 1.0
        self.artificial_vis_beta = 0.0

        self.seval = None

        # attributes for Sun PST technique
        self.u_f = 0.059

    def create_rings_geometry(self):
        import matplotlib.pyplot as plt
        x, y = np.array([]), np.array([])

        radii = np.arange(self.ri, self.ro, self.dx)
        # radii = np.arange(self.ri, self.ri+self.dx, self.dx)

        for radius in radii:
            points = np.arange(0., 2 * np.pi * radius, self.dx)
            theta = np.arange(0., 2. * np.pi, 2. * np.pi / len(points))
            xr, yr = radius * np.cos(theta), radius * np.sin(theta)
            x = np.concatenate((x, xr))
            y = np.concatenate((y, yr))

        plt.scatter(x, y)
        # plt.show()
        return x, y

    def create_particles(self):
        spacing = self.spacing  # spacing = 2*5cm

        x, y = self.create_rings_geometry()
        x = x.ravel()
        y = y.ravel()

        dx = self.dx
        hdx = self.hdx
        m = self.rho0 * dx * dx
        h = np.ones_like(x) * hdx * dx
        rho = self.rho0

        ring_1 = get_particle_array(x=-x - spacing,
                                    y=y,
                                    m=m,
                                    rho=rho,
                                    h=h,
                                    name="ring_1",
                                    constants={
                                        'E': self.E,
                                        'n': 4,
                                        'nu': self.nu,
                                        'spacing0': self.dx,
                                        'rho_ref': self.rho0
                                    })

        ring_2 = get_particle_array(x=x + spacing,
                                    y=y,
                                    m=m,
                                    rho=rho,
                                    h=h,
                                    name="ring_2",
                                    constants={
                                        'E': self.E,
                                        'n': 4,
                                        'nu': self.nu,
                                        'spacing0': self.dx,
                                        'rho_ref': self.rho0
                                    })

        self.scheme.setup_properties([ring_1, ring_2])

        u_f = self.u_f
        ring_1.u = ring_1.cs * u_f
        ring_2.u = -ring_1.cs * u_f

        return [ring_1, ring_2]

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=500)

    def create_scheme(self):
        fsi = FSIScheme(fluids=[],
                        solids=['ring_1', 'ring_2'],
                        boundaries=[],
                        solid_supports=[],
                        dim=self.dim,
                        rho0=1000.,
                        c0=0.,
                        p0=0.,
                        h0=self.h,
                        hdx=self.hdx)

        s = SchemeChooser(default='fsi', fsi=fsi)
        return s


if __name__ == '__main__':
    app = Rings()

    app.run()
    # app.create_rings_geometry()
