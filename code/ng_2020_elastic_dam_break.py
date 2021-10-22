"""A coupled Smoothed Particle Hydrodynamics-Volume Compensated Particle Method
(SPH-VCPM) for Fluid Structure Interaction (FSI) modelling
"""

import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

# from rigid_fluid_coupling import RigidFluidCouplingScheme
from geometry import hydrostatic_tank_2d, create_tank_2d_from_block_2d

from pysph.examples.solid_mech.impact import add_properties
# from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
#                                                                create_fluid,
#                                                                create_sphere)
from pysph.tools.geometry import get_2d_block, rotate

from fsi_coupling import FSIETVFScheme, FSIETVFSubSteppingScheme

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)


def find_displacement_index(pa):
    x = pa.x
    y = pa.y
    min_y = min(y)
    max_x = max(x)
    index = (pa.x == max_x) & (pa.y == min_y)
    pa.add_property('tip_displacemet_index', type='int',
                    data=np.zeros(len(pa.x)))
    pa.tip_displacemet_index[index] = 1


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
    xb, yb = get_2d_block(dx=spacing, length=beam_length + beam_inside_length,
                          height=beam_height)

    # create a (support) block with required number of layers
    xs1, ys1 = get_2d_block(dx=spacing, length=beam_inside_length,
                            height=boundary_layers * spacing)
    xs1 += np.min(xb) - np.min(xs1)
    ys1 += np.min(yb) - np.max(ys1) - spacing

    # create a (support) block with required number of layers
    xs2, ys2 = get_2d_block(dx=spacing, length=beam_inside_length,
                            height=boundary_layers * spacing)
    xs2 += np.min(xb) - np.min(xs2)
    ys2 += np.max(ys2) - np.min(yb) + spacing

    xs = np.concatenate([xs1, xs2])
    ys = np.concatenate([ys1, ys2])

    xs3, ys3 = get_2d_block(dx=spacing, length=boundary_layers * spacing,
                            height=np.max(ys) - np.min(ys))
    xs3 += np.min(xb) - np.max(xs3) - 1. * spacing
    # ys3 += np.max(ys2) - np.min(yb) + spacing

    xs = np.concatenate([xs, xs3])
    ys = np.concatenate([ys, ys3])
    # plt.scatter(xs, ys, s=1)
    # plt.scatter(xb, yb, s=1)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig("geometry", dpi=300)
    # plt.show()

    return xb, yb, xs, ys


def get_fixed_beam_without_clamp(beam_length, beam_height, boundary_height, spacing):
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
    xb, yb = get_2d_block(dx=spacing, length=beam_length,
                          height=beam_height)

    # create a (support) block with required number of layers
    xs1, ys1 = get_2d_block(dx=spacing, length=beam_length,
                            height=boundary_height)

    ys1 += np.min(yb) - np.max(ys1) - spacing

    xs = np.concatenate([xs1])
    ys = np.concatenate([ys1])
    # plt.scatter(xs, ys, s=1)
    # plt.scatter(xb, yb, s=1)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig("geometry", dpi=300)
    # plt.show()

    return xb, yb, xs, ys


def set_normals_tank(pa, spacing):
    min_x = min(pa.x)
    max_x = max(pa.x)
    min_y = min(pa.y)
    # left wall
    fltr = (pa.x < min_x + 3. * spacing) & (pa.y > min_y + 3. * spacing)
    for i in range(len(fltr)):
        if fltr[i] == True:
            pa.normal[3*i] = 1.
            pa.normal[3*i+1] = 0.
            pa.normal[3*i+2] = 0.

    # right wall
    fltr = (pa.x > max_x - 3. * spacing) & (pa.y > min_y + 3. * spacing)
    for i in range(len(fltr)):
        if fltr[i] == True:
            pa.normal[3*i] = -1.
            pa.normal[3*i+1] = 0.
            pa.normal[3*i+2] = 0.

    # bottom wall
    fltr = (pa.x > min_x + 5. * spacing) & (pa.x < max_x - 5. * spacing)
    for i in range(len(fltr)):
        if fltr[i] == True:
            pa.normal[3*i] = 0.
            pa.normal[3*i+1] = 1.
            pa.normal[3*i+2] = 0.


class ElasticGate(Application):
    def add_user_options(self, group):
        group.add_argument("--d0", action="store", type=float, dest="d0",
                           default=1e-3,
                           help="Spacing between the particles")

    def consume_user_options(self):
        # ================================================
        # consume the user options first
        # ================================================
        self.d0 = self.options.d0
        spacing = self.d0

        # ================================================
        # common properties
        # ================================================
        self.hdx = 1.0
        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 2
        self.seval = None

        # ================================================
        # Fluid properties
        # ================================================
        self.fluid_length = 0.1
        self.fluid_height = 0.14
        self.fluid_spacing = spacing
        self.h_fluid = self.hdx * self.fluid_spacing
        self.vref_fluid = np.sqrt(2 * 9.81 * self.fluid_height)
        self.c0_fluid = 10 * self.vref_fluid
        self.nu_fluid = 0.
        self.rho0_fluid = 1000.0
        self.fluid_density = 1000.0
        self.mach_no_fluid = self.vref_fluid / self.c0_fluid
        self.pb_fluid = self.rho0_fluid * self.c0_fluid**2.
        self.alpha_fluid = 0.1
        self.edac_alpha = 0.5  # these variable are doubt
        self.edac_nu = self.edac_alpha * self.c0_fluid * self.h_fluid / 8  # these variable are doubt
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "tank", "gate",
                                             "gate_support"])

        # ================================================
        # Tank properties
        # ================================================
        self.tank_height = 0.15
        self.tank_length = 0.2
        self.tank_layers = 3
        self.tank_spacing = spacing

        # ================================================
        # properties related to the elastic gate
        # ================================================
        # elastic gate is made of rubber
        self.L = 0.005
        self.H = 0.079
        self.gate_spacing = self.fluid_spacing
        self.gate_rho0 = 1100
        self.gate_E = 11958923.292360354
        self.gate_nu = 0.4
        self.c0_gate = get_speed_of_sound(self.gate_E, self.gate_nu,
                                          self.gate_rho0)
        self.pb_gate = self.gate_rho0 * self.c0_gate**2
        self.u_max_gate = 50
        self.mach_no_gate = self.u_max_gate / self.c0_gate
        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["gate"], sources=["gate", "gate_support"],
            boundaries=["gate_support"])

        # ================================================
        # common properties
        # ================================================
        self.boundary_equations = (self.boundary_equations_1 +
                                   self.boundary_equations_2)

        self.dt_fluid = 0.25 * self.fluid_spacing * self.hdx / (
            self.c0_fluid * 1.1)
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
            self.tank_layers)

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

        # ===================================
        # Create tank
        # ===================================
        tank = get_particle_array(x=xt,
                                  y=yt,
                                  m=m_fluid,
                                  m_fluid=m_fluid,
                                  h=self.h_fluid,
                                  rho=self.fluid_density,
                                  rad_s=self.tank_spacing/2.,
                                  name="tank")

        # =============================================
        # Only structures part particle properties
        # =============================================
        with_out_clamp = True
        if with_out_clamp is True:
            xp, yp, xw, yw = get_fixed_beam_without_clamp(self.H, self.L,
                                                          self.L, self.fluid_spacing)

        else:
            xp, yp, xw, yw = get_fixed_beam(self.H, self.L, self.H/2.5,
                                            self.wall_layers, self.fluid_spacing)

        # make sure that the beam intersection with wall starts at the 0.
        # min_xp = np.min(xp)

        # # add this to the beam and wall
        # xp += abs(min_xp)
        # xw += abs(min_xp)

        # max_xw = np.max(xw)
        # xp -= abs(max_xw)
        # xw -= abs(max_xw)

        m = self.gate_rho0 * self.fluid_spacing**2.

        # ===================================
        # Create elastic gate
        # ===================================
        xp += self.fluid_length
        gate = get_particle_array(
            x=xp, y=yp, m=m, h=self.h_fluid, rho=self.gate_rho0, name="gate",
            constants={
                'E': self.gate_E,
                'n': 4.,
                'nu': self.gate_nu,
                'spacing0': self.gate_spacing,
                'rho_ref': self.gate_rho0,
                'rho_ref_fluid': self.fluid_density,
            })

        # ===================================
        # Create elastic gate support
        # ===================================
        xw += self.fluid_length
        # xw += max(xf) + max(xf) / 2.
        gate_support = get_particle_array(
            x=xw, y=yw, m=m, h=self.h_fluid, rho=self.gate_rho0, name="gate_support",
            constants={
                'E': self.gate_E,
                'n': 4.,
                'nu': self.gate_nu,
                'spacing0': self.gate_spacing,
                'rho_ref': self.gate_rho0
            })
        # ================================
        # Adjust the geometry
        # ================================
        # rotate the particles
        axis = np.array([0.0, 0.0, 1.0])
        angle = -90
        xp, yp, zp = rotate(gate.x, gate.y, gate.z, axis, angle)
        gate.x, gate.y, gate.z = xp[:], yp[:], zp[:]

        xw, yw, zw = rotate(gate_support.x, gate_support.y,
                            gate_support.z, axis, angle)
        gate_support.x, gate_support.y, gate_support.z = xw[:], yw[:], zw[:]

        # translate gate and gate support
        x_translate = (max(fluid.x) - min(gate_support.x)) - self.fluid_spacing * 2.
        gate.x += x_translate
        gate_support.x += x_translate

        y_translate = (max(tank.y) - max(gate_support.y)) + 3. * self.fluid_spacing
        gate.y += y_translate
        gate_support.y += y_translate

        if with_out_clamp is True:
            # set the gate and gate support x variables
            # translate gate and gate support
            x_translate = (max(fluid.x) - min(gate.x)) + self.fluid_spacing
            gate.x += x_translate

            x_translate = (max(fluid.x) - min(gate_support.x)) + self.fluid_spacing
            gate_support.x += x_translate

            y_translate = (min(tank.y) - min(gate.y)) + 3. * self.fluid_spacing
            gate.y += y_translate

            y_translate = (max(gate.y) - min(gate_support.y))
            gate_support.y += y_translate + self.fluid_spacing

        # gate.x[:] += 3. * self.fluid_spacing
        # gate_support.x[:] += 3. * self.fluid_spacing

        # ================================
        # Adjust the geometry ends
        # ================================

        self.scheme.setup_properties([fluid, tank,
                                      gate, gate_support])

        # add post processing variables.
        find_displacement_index(gate)

        gate.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate.rho_fsi[:] = self.fluid_density

        gate_support.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate_support.rho_fsi[:] = self.fluid_density

        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_density * self.gy * (max(fluid.y) - fluid.y[:])

        gate.add_output_arrays(['tip_displacemet_index'])

        # ================================
        # set the normals of the tank
        # ================================

        set_normals_tank(tank, self.fluid_spacing)

        return [fluid, tank, gate, gate_support]

    def create_scheme(self):
        ctvf = FSIETVFScheme(fluids=['fluid'],
                             solids=['tank'],
                             structures=['gate'],
                             structure_solids=['gate_support'],
                             dim=2,
                             h_fluid=0.,
                             c0_fluid=0.,
                             nu_fluid=0.,
                             rho0_fluid=0.,
                             mach_no_fluid=0.,
                             mach_no_structure=0.)

        substep = FSIETVFSubSteppingScheme(fluids=['fluid'],
                                           solids=['tank'],
                                           structures=['gate'],
                                           structure_solids=['gate_support'],
                                           dim=2,
                                           h_fluid=0.,
                                           c0_fluid=0.,
                                           nu_fluid=0.,
                                           rho0_fluid=0.,
                                           mach_no_fluid=0.,
                                           mach_no_structure=0.)

        s = SchemeChooser(default='ctvf', substep=substep, ctvf=ctvf)

        return s

    def configure_scheme(self):
        # dt = 0.125 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        dt = 0.25 * self.h_fluid / (
            (self.gate_E / self.gate_rho0)**0.5 + (self.u_max_gate/50.))

        # print("DT: %s" % dt)
        tf = 0.5

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
        x_0 = pa.x[index]

        t_ctvf, y_ctvf, x_ctvf = [], [], []
        for sd, gate in iter_output(files[::5], 'gate'):
            _t = sd['t']
            t_ctvf.append(_t)
            y_ctvf.append(gate.y[index] - y_0)
            x_ctvf.append(gate.x[index] - x_0)

        # gtvf data
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # load the data
        data_x_disp_antoci_exp = np.loadtxt(os.path.join(directory, 'ng_2020_elastic_dam_break_x_displacement_antoci_2007_experiment.csv'),
                                            delimiter=',')
        data_x_disp_khayyer_2018 = np.loadtxt(os.path.join(directory, 'ng_2020_elastic_dam_break_x_displacement_khayyer_2018_isph_sph.csv'),
                                              delimiter=',')
        data_x_disp_yang_2012 = np.loadtxt(os.path.join(directory, 'ng_2020_elastic_dam_break_x_displacement_yang_2012_sph_fem.csv'),
                                           delimiter=',')
        data_x_disp_ng_2020 = np.loadtxt(os.path.join(directory, 'ng_2020_elastic_dam_break_x_displacement_ng_2020_sph_vcpm_alpha_1.csv'),
                                         delimiter=',')
        data_x_disp_wcsph_pysph = np.loadtxt(os.path.join(directory, 'ng_2020_elastic_dam_break_x_displacement_wcsph_pysph.csv'),
                                             delimiter=',')

        data_y_disp_antoci_exp = np.loadtxt(os.path.join(directory, 'ng_2020_elastic_dam_break_y_displacement_antoci_2007_experiment.csv'),
                                            delimiter=',')
        data_y_disp_khayyer_2018 = np.loadtxt(os.path.join(directory, 'ng_2020_elastic_dam_break_y_displacement_khayyer_2018_isph_sph.csv'),
                                              delimiter=',')
        data_y_disp_yang_2012 = np.loadtxt(os.path.join(directory, 'ng_2020_elastic_dam_break_y_displacement_yang_2012_sph_fem.csv'),
                                           delimiter=',')
        data_y_disp_ng_2020 = np.loadtxt(os.path.join(directory, 'ng_2020_elastic_dam_break_y_displacement_ng_2020_sph_vcpm_alpha_1.csv'),
                                         delimiter=',')

        txant, xdant = data_x_disp_antoci_exp[:, 0], data_x_disp_antoci_exp[:, 1]
        txkha, xdkha = data_x_disp_khayyer_2018[:, 0], data_x_disp_khayyer_2018[:, 1]
        txyan, xdyan = data_x_disp_yang_2012[:, 0], data_x_disp_yang_2012[:, 1]
        txng, xdng = data_x_disp_ng_2020[:, 0], data_x_disp_ng_2020[:, 1]
        txwcsph, xdwcsph = data_x_disp_wcsph_pysph[:, 0], data_x_disp_wcsph_pysph[:, 1]
        txwcsph += 0.05

        tyant, ydant = data_y_disp_antoci_exp[:, 0], data_y_disp_antoci_exp[:, 1]
        tykha, ydkha = data_y_disp_khayyer_2018[:, 0], data_y_disp_khayyer_2018[:, 1]
        tyyan, ydyan = data_y_disp_yang_2012[:, 0], data_y_disp_yang_2012[:, 1]
        tyng, ydng = data_y_disp_ng_2020[:, 0], data_y_disp_ng_2020[:, 1]

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res, txant=txant, xdant=xdant, txkha=txkha, xdkha=xdkha,
                 txyan=txyan, xdyan=xdyan, txng=txng, xdng=xdng, tyant=tyant,
                 ydant=ydant, tykha=tykha, ydkha=ydkha, tyyan=tyyan,
                 ydyan=ydyan, tyng=tyng, ydng=ydng, txwcsph=txwcsph,
                 xdwcsph=xdwcsph, t_ctvf=t_ctvf, x_ctvf=x_ctvf, y_ctvf=y_ctvf)

        # ========================
        # x amplitude figure
        # ========================
        plt.clf()
        plt.plot(txant, xdant, "o", label='Antoci 2008, Experiment')
        plt.plot(txkha, xdkha, "^", label='Khayyer 2018, ISPH-SPH')
        plt.plot(txyan, xdyan, "+", label='Yang 2012, SPH-FEM')
        plt.plot(txng, xdng, "v", label='Ng 2020, SPH-VCPM')
        plt.plot(txwcsph, xdwcsph, "*", label='WCSPH PySPH')
        plt.plot(t_ctvf, x_ctvf, "-", label='CTVF')

        plt.title('x amplitude')
        plt.xlabel('t')
        plt.ylabel('x amplitude')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "x_amplitude_with_t.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================

        # ========================
        # y amplitude figure
        # ========================
        plt.clf()
        plt.plot(tyant, ydant, "o", label='Antoci 2008, Experiment')
        plt.plot(tykha, ydkha, "v", label='Khayyer 2018, ISPH-SPH')
        plt.plot(tyyan, ydyan, "o", label='Yang 2012, SPH-FEM')
        plt.plot(tyng, ydng, "o", label='Ng 2020, SPH-VCPM')
        plt.plot(t_ctvf, y_ctvf, "-", label='CTVF')

        plt.title('y amplitude')
        plt.xlabel('t')
        plt.ylabel('y amplitude')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "y_amplitude_with_t.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # y amplitude figure
        # ========================


if __name__ == '__main__':
    app = ElasticGate()
    app.run()
    app.post_process(app.info_filename)
