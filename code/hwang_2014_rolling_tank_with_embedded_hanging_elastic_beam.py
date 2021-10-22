"""This is also there is Sun2021, An accurate FSI-SPH modeling of challenging
fluid-structure interaction problems in two and three dimensions. Which gives
us the expression for the rolling tank's theta.


1. Development of a fully Lagrangian MPS-based coupled method for simulation of
fluid-structure interaction problems. 3.2.2
https://doi.org/10.1016/j.jfluidstructs.2014.07.007

"""
import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.examples.solid_mech.impact import add_properties
# from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
#                                                                create_fluid,
#                                                                create_sphere)
from pysph.tools.geometry import get_2d_block, rotate

from fsi_coupling import FSIETVFSubSteppingScheme

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)


def create_tank_2d_from_block_2d(xf, yf, tank_length, tank_height,
                                 tank_spacing, tank_layers, close=False):
    """
    This is mainly used by granular flows

    Tank particles radius is spacing / 2.
    """
    ####################################
    # create the left wall of the tank #
    ####################################
    xleft, yleft = get_2d_block(dx=tank_spacing,
                                length=(tank_layers - 1) * tank_spacing,
                                height=tank_height + 1. * tank_spacing,
                                center=[0., 0.])
    xleft += min(xf) - max(xleft) - tank_spacing
    yleft += min(yf) - min(yleft)

    xright = xleft + abs(min(xleft)) + tank_length
    xright -= min(xright) - max(xf) - tank_spacing
    yright = yleft

    xbottom, ybottom = get_2d_block(dx=tank_spacing,
                                    length=max(xright) - min(xleft),
                                    height=(tank_layers - 1) * tank_spacing,
                                    center=[0., 0.])
    xbottom += min(xleft) - min(xbottom)
    ybottom -= max(ybottom) - min(yf) + tank_spacing

    if close is True:
        xtop, ytop = get_2d_block(dx=tank_spacing,
                                  length=max(xright) - min(xleft),
                                  height=(tank_layers - 1) * tank_spacing,
                                  center=[0., 0.])
        xtop += min(xleft) - min(xtop)
        ytop += max(yleft) - min(ytop) + 1. * tank_spacing

    x = np.concatenate([xleft, xright, xbottom, xtop])
    y = np.concatenate([yleft, yright, ybottom, ytop])
    # x = np.concatenate([xleft, xright])
    # y = np.concatenate([yleft, yright])

    return x, y


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

    xt = np.concatenate([xt_1, xt_2, xt_3])
    yt = np.concatenate([yt_1, yt_2, yt_3])

    return xf, yf, xt, yt


def get_elastic_plate_with_support(beam_length, beam_height, boundary_layers,
                                   spacing):
    # create a block first
    xb, yb = get_2d_block(dx=spacing, length=beam_length,
                          height=beam_height)

    # create a (support) block with required number of layers
    xs, ys = get_2d_block(dx=spacing, length=(3. * beam_length),
                          height=boundary_layers * spacing)
    ys += np.max(yb) - np.min(ys) + spacing
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
    pa.add_output_arrays(['tip_displacemet_index'])


def set_rotation_point(pa):
    y = pa.y
    min_y = min(y)
    min_y_indices = np.where(y == min_y)[0]
    index = min_y_indices[int(len(min_y_indices)/2)]
    pa.xcm[0] = pa.x[index]
    pa.xcm[1] = pa.y[index]


def set_body_frame_position_vectors(pa):
    """Save the position vectors w.r.t body frame"""
    # loop over all the bodies
    add_properties(pa, 'dx0', 'dy0', 'dz0')
    cm_i = pa.xcm
    for j in range(len(pa.x)):
        pa.dx0[j] = pa.x[j] - cm_i[0]
        pa.dy0[j] = pa.y[j] - cm_i[1]
        pa.dz0[j] = pa.z[j] - cm_i[2]


class ElasticGate(Application):
    def add_user_options(self, group):
        group.add_argument("--d0", action="store", type=float, dest="d0",
                           default=1e-3,
                           help="No of particles in the height direction")

    def consume_user_options(self):
        self.dim = 2
        self.d0 = self.options.d0

        # ================================================
        # properties related to the only fluids
        # ================================================
        spacing = self.d0
        self.hdx = 1.0

        # ================================================
        # Fluid properties
        # ================================================
        self.fluid_length = 609 * 1e-3
        self.fluid_height = 57.4 * 1e-3
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing
        self.rho0_fluid = self.fluid_density
        self.vref_fluid = np.sqrt(2 * 9.81 * self.fluid_height)
        self.u_max_fluid = self.vref_fluid
        self.c0_fluid = 10 * self.vref_fluid
        self.mach_no_fluid = self.vref_fluid / self.c0_fluid
        self.p0_fluid = self.fluid_density * self.c0_fluid**2.
        self.pb_fluid = self.p0_fluid
        self.alpha = 0.1
        self.gy = -9.81
        self.seval = None
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "tank", "gate",
                                             "gate_support"],
            boundaries=["tank", "gate", "gate_support"])

        # ================================================
        # Tank properties
        # ================================================
        self.tank_height = 344.5 * 1e-3
        self.tank_length = 609 * 1e-3
        self.tank_layers = 3
        self.tank_spacing = spacing

        self.h_fluid = self.hdx * self.fluid_spacing

        # ================================================
        # properties related to the elastic gate
        # ================================================
        # elastic gate is made of rubber
        self.gate_length = 4 * 1e-3
        self.gate_height = 287.1 * 1e-3
        self.gate_spacing = self.fluid_spacing
        self.gate_rho0 = 1900
        self.gate_E = 4 * 1e6
        self.gate_nu = 0.49
        self.c0_gate = get_speed_of_sound(self.gate_E, self.gate_nu,
                                          self.gate_rho0)
        self.pb_gate = self.gate_rho0 * self.c0_gate**2
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0_gate * self.h_fluid / 8
        self.u_max_gate = 0.05
        self.mach_no_gate = self.u_max_gate / self.c0_gate
        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["gate"], sources=["gate", "gate_support"],
            boundaries=["gate_support"])

        # ================================================
        # common properties
        # ================================================
        self.boundary_equations = self.boundary_equations_1 + self.boundary_equations_2
        self.dt_fluid = 0.25 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        self.dt_solid = 0.25 * self.h_fluid / (
            (self.gate_E / self.gate_rho0)**0.5 + self.u_max_gate)

    def create_particles(self):
        # ===================================
        # Create fluid
        # ===================================
        xf, yf = get_2d_block(dx=self.fluid_spacing, length=self.fluid_length,
                              height=self.fluid_height)

        xt, yt = create_tank_2d_from_block_2d(
            xf, yf, self.tank_length, self.tank_height, self.tank_spacing,
            self.tank_layers, close=True)

        m_fluid = self.fluid_density * self.fluid_spacing**2.

        # =============================================
        # Only fluids part particle properties
        # =============================================

        # ===================================
        # Create fluid
        # ===================================
        fluid = get_particle_array(x=xf,
                                   y=yf,
                                   m=m_fluid,
                                   h=self.h_fluid,
                                   rho=self.fluid_density,
                                   name="fluid")

        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_density * self.gy * (max(fluid.y) - fluid.y[:])

        # ===================================
        # Create tank
        # ===================================
        tank = get_particle_array(x=xt,
                                  y=yt,
                                  m=m_fluid,
                                  m_fluid=m_fluid,
                                  h=self.h_fluid,
                                  rho=self.fluid_density,
                                  rad_s=self.fluid_spacing/2.,
                                  name="tank")
        # add properties to rotate the tank about some point
        tank.add_constant('theta', 0.)
        tank.add_constant('xcm', np.zeros(3, dtype=float))
        set_rotation_point(tank)
        set_body_frame_position_vectors(tank)

        # =============================================
        # Only structures part particle properties
        # =============================================
        xp, yp, xw, yw = get_elastic_plate_with_support(self.gate_length,
                                                        self.gate_height,
                                                        self.tank_layers,
                                                        self.fluid_spacing)
        # scale of shift of gate support
        scale = max(tank.y) - max(yw) + self.fluid_spacing
        yp += scale
        yw += scale

        m = self.gate_rho0 * self.fluid_spacing**2.

        # ===================================
        # Create elastic gate
        # ===================================
        gate = get_particle_array(
            x=xp, y=yp, m=m, h=self.h_fluid, rho=self.gate_rho0, name="gate",
            constants={
                'E': self.gate_E,
                'n': 4.,
                'nu': self.gate_nu,
                'spacing0': self.gate_spacing,
                'rho_ref': self.gate_rho0
            })
        # add post processing variables.
        find_displacement_index(gate)

        # ===================================
        # Create elastic gate support
        # ===================================
        gate_support = get_particle_array(
            x=xw, y=yw, m=m, h=self.h_fluid, rho=self.gate_rho0, name="gate_support",
            constants={
                'E': self.gate_E,
                'n': 4.,
                'nu': self.gate_nu,
                'spacing0': self.gate_spacing,
                'rho_ref': self.gate_rho0
            })

        self.scheme.setup_properties([fluid, tank,
                                      gate, gate_support])

        gate.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate.rho_fsi[:] = self.fluid_density

        gate_support.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate_support.rho_fsi[:] = self.fluid_density

        return [fluid, tank, gate, gate_support]

    def create_scheme(self):
        substep = FSIETVFSubSteppingScheme(fluids=['fluid'],
                                           solids=['tank'],
                                           structures=['gate'],
                                           structure_solids=['gate_support'],
                                           dt_fluid=1.,
                                           dt_solid=1.,
                                           dim=2,
                                           h_fluid=0.,
                                           rho0_fluid=0.,
                                           pb_fluid=0.,
                                           c0_fluid=0.,
                                           nu_fluid=0.,
                                           mach_no_fluid=0.,
                                           mach_no_structure=0.,
                                           gy=0.)

        s = SchemeChooser(default='substep', substep=substep)

        return s

    def configure_scheme(self):
        # dt = 0.125 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        # TODO: This has to be changed for solid
        dt = 0.25 * self.h_fluid / (
            (self.gate_E / self.gate_rho0)**0.5 + self.u_max_gate)
        dt = 0.25 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        print("time step fluid dt", dt)

        print("DT: %s" % dt)
        tf = 5.

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=2000)

        self.scheme.configure(
            dim=2,
            h_fluid=self.h_fluid,
            rho0_fluid=self.rho0_fluid,
            pb_fluid=self.pb_fluid,
            c0_fluid=self.c0_fluid,
            nu_fluid=0.0,
            mach_no_fluid=self.mach_no_fluid,
            mach_no_structure=self.mach_no_gate,
            gy=self.gy,
            alpha_fluid=0.1,
            alpha_solid=1.,
            beta_solid=0,
            dt_fluid=self.dt_fluid,
            dt_solid=self.dt_solid
        )

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # if self.options.scheme == 'etvf':
        #     equation = eqns.groups[-1][5].equations[4]
        #     equation.sources = ["tank", "fluid", "gate", "gate_support"]
        # print(equation)

        return eqns

    def _make_accel_eval(self, equations, pa_arrays):
        from pysph.base.kernels import (QuinticSpline)
        from pysph.tools.sph_evaluator import SPHEvaluator
        if self.seval is None:
            # kernel = self.options.kernel(dim=self.dim)
            kernel = QuinticSpline(dim=self.dim)
            seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                                 dim=self.dim, kernel=kernel)
            self.seval = seval
            return self.seval
        else:
            self.seval.update()
            return self.seval
        return seval

    def pre_step(self, solver):
        if solver.count % 1 == 0:
            t = solver.t
            dt = solver.dt

            arrays = self.particles
            a_eval = self._make_accel_eval(self.boundary_equations, arrays)

            # When
            a_eval.evaluate(t, dt)

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        for pa in self.particles:
            if pa.name == 'tank':
                theta = 2. * np.sin(2. * np.pi * 0.607 * t) * np.pi / 180
                theta_dot = (2. * 2. * np.pi * 0.607 *
                             np.cos(2. * np.pi * 0.607 * t) * np.pi) / 180

                # rotate the position of the vector in the body frame to
                # global frame
                R0 = np.cos(theta)
                R1 = -np.sin(theta)
                R2 = np.sin(theta)
                R3 = np.cos(theta)
                for i in range(len(pa.x)):
                    pa.x[i] = pa.xcm[0] + R0 * pa.dx0[i] + R1 * pa.dy0[i]
                    pa.y[i] = pa.xcm[1] + R2 * pa.dx0[i] + R3 * pa.dy0[i]
                    pa.u[i] = - theta_dot * (pa.xcm[1] - pa.y[i])
                    pa.v[i] = theta_dot * (pa.xcm[0] - pa.x[i])
                    pa.uhat[i] = pa.u[i]
                    pa.vhat[i] = pa.v[i]

    def post_process(self, fname):
        from pysph.solver.utils import iter_output, load
        import os
        from matplotlib import pyplot as plt

        info = self.read_info(fname)
        files = self.output_files

        data = load(files[0])
        arrays = data['arrays']
        pa = arrays['gate']
        index = np.where(pa.tip_displacemet_index == 1)[0][0]
        y_0 = pa.y[index]

        files = files[0::1]
        t_ctvf, y_ctvf = [], []
        for sd, gate in iter_output(files, 'gate'):
            _t = sd['t']
            t_ctvf.append(_t)
            y_ctvf.append((gate.y[index] - y_0) * 1)

        t_analytical = np.linspace(0., 1., 1000)
        y_analytical = -6.849 * 1e-5 * np.ones_like(t_analytical)

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res, t_analytical=t_analytical, y_analytical=y_analytical,
                 y_ctvf=y_ctvf, t_ctvf=t_ctvf)

        plt.clf()
        plt.plot(t_analytical, y_analytical, label='Analytical')
        plt.plot(t_ctvf, y_ctvf, "-", label='Simulated')

        plt.xlabel('t')
        plt.ylabel('y-amplitude')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "amplitude_with_t.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = ElasticGate()
    app.run()
    app.post_process(app.info_filename)
