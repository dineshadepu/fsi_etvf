"""Hydrostatic tank

python dam_break_2d.py --openmp --integrator gtvf --no-internal-flow --pst sun2019 --no-set-solid-vel-project-uhat --no-set-uhat-solid-vel-to-u --no-vol-evol-solid --no-edac-solid --surf-p-zero -d dam_break_2d_etvf_integrator_gtvf_pst_sun2019_output --pfreq 1 --detailed-output


"""
import numpy as np

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.base.utils import get_particle_array

from pysph.examples import cavity as LDC
from pysph.sph.equation import Equation, Group
from fluids_wcsph import FluidsWCSPHScheme

from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import SchemeChooser

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)


fluid_column_height = 2.0
fluid_column_width = 1.0
container_height = 4.0
container_width = 4
nboundary_layers = 4

g = 9.81
ro = 1000.0
vref = np.sqrt(2*g*fluid_column_height)
co = 10.0 * vref
mach_no = vref / co
nu = 0.0
tf = 1.0
p0 = ro*co**2


def get_hydrostatic_tank_with_fluid(fluid_length=1., fluid_height=2.,
                                    tank_height=2.3, tank_layers=2,
                                    fluid_spacing=0.1):
    xf, yf = get_2d_block(dx=fluid_spacing,
                          length=fluid_length,
                          height=fluid_height)

    xt_1, yt_1 = get_2d_block(dx=fluid_spacing,
                              length=tank_layers*fluid_spacing,
                              height=tank_height+fluid_spacing/2.)
    xt_1 -= max(xt_1) - min(xf) + fluid_spacing
    yt_1 += min(yf) - min(yt_1)

    xt_2, yt_2 = get_2d_block(dx=fluid_spacing,
                              length=tank_layers*fluid_spacing,
                              height=tank_height+fluid_spacing/2.)
    xt_2 += max(xf) - min(xt_2) + fluid_spacing
    yt_2 += min(yf) - min(yt_2)

    xt_3, yt_3 = get_2d_block(dx=fluid_spacing,
                              length=max(xt_2) - min(xt_1),
                              height=tank_layers*fluid_spacing)
    yt_3 -= max(yt_3) - min(yf)
    yt_3 -= fluid_spacing

    xt = np.concatenate([xt_1, xt_2, xt_3])
    yt = np.concatenate([yt_1, yt_2, yt_3])

    return xf, yf, xt, yt


class HydrostaticTank(DB.DamBreak2D):
    def initialize(self):
        super(HydrostaticTank, self).initialize()

        self.hdx = 1.
        self.dim = 2
        # print(self.boundary_equations)

    def create_particles(self):
        xf, yf, xt, yt = get_hydrostatic_tank_with_fluid(
            fluid_length=fluid_column_width, fluid_height=fluid_column_height,
            tank_height=container_height, tank_layers=2, fluid_spacing=self.dx)

        # xt += 2.0
        # xf += 2.0
        # xf += self.dx
        # yf += self.dx
        h = self.hdx * self.dx
        self.h = h
        m = self.dx**2 * ro
        m_tank = (self.dx)**2 * ro
        h_tank = self.hdx * self.dx
        fluid = get_particle_array(name='fluid', x=xf, y=yf, h=h, m=m, rho=ro)
        tank = get_particle_array(name='boundary', x=xt, y=yt, h=h_tank,
                                  m=m_tank, rho=ro)
        self.scheme.setup_properties([fluid, tank])

        if self.options.pst == "sun2019":
            kernel = self.scheme.scheme.kernel(dim=2)
            wdeltap = kernel.kernel(rij=self.dx, h=self.h)
            fluid.wdeltap[0] = wdeltap
            fluid.n[0] = 4

        return [fluid, tank]

    def consume_user_options(self):
        self.options.scheme = 'wcsph'
        super(HydrostaticTank, self).consume_user_options()

    def configure_scheme(self):
        # super().configure_scheme()

        # dt = 0.125*self.h/(co + vref)
        dt = 5e-5
        h0 = self.hdx * self.dx
        scheme = self.scheme
        if self.options.scheme == 'wcsph':
            scheme.configure(pb=p0, nu=nu, h=h0)

            times = [0.4, 0.6, 0.8]
            self.scheme.configure_solver(dt=dt, tf=tf, output_at_times=times)

    def create_scheme(self):
        h0 = None
        wcsph = FluidsWCSPHScheme(
            ['fluid'], ['boundary'], dim=2, rho0=ro, c0=co, nu=None,
            pb=p0, h=None, u_max=3. * vref, mach_no=mach_no,
            gy=-9.81, alpha=0.05)
        schemes = super().create_scheme()
        schemes.schemes['wcsph'] = wcsph
        schemes.default = 'wcsph'
        return schemes

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')

    def _make_accel_eval(self, equations, pa_arrays):
        if self.seval is None:
            kernel = QuinticSpline(dim=self.dim)
            seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                                 dim=self.dim, kernel=kernel)
            self.seval = seval
            return self.seval
        else:
            self.seval.update()
            return self.seval
        return seval


if __name__ == '__main__':
    app = HydrostaticTank()
    app.run()
    app.post_process(app.info_filename)
