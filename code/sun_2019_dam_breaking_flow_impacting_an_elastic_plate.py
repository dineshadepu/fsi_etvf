"""[1] A fully Lagrangian method for fluid-structure interaction problems with
deformable floating structure

https://doi.org/10.1016/j.jfluidstructs.2019.07.005

Section 3.1.2: Water impact on a deformable obstruction


[2] Numerical simulation of hydro-elastic problems with smoothed particle hydro-
dynamics method

DOI: 10.1016/S1001-6058(13)60412-6

3.3 Water impact onto a forefront elastic plate


[3] Unified Lagrangian formulation for elastic solids and incompressible fluids:
Application to fluid-structure interaction problems via the PFEM

https://doi.org/10.1016/j.cma.2007.06.004


7.2 Impact of sea waves on solid object

# Image for dimensions

https://ars.els-cdn.com/content/image/1-s2.0-S0889974618308673-gr7_lrg.jpg

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

# from fsi_coupling import FSIETVFScheme, FSIETVFSubSteppingScheme
from fsi_wcsph import FSIWCSPHScheme

from fsi_wcsph import (FSIWCSPHScheme, FSIWCSPHFluidsScheme,
                       FSIWCSPHFluidsScheme, FSIWCSPHFluidsSubSteppingScheme)

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)

from pysph.sph.scheme import add_bool_argument
from pysph.tools.geometry import (remove_overlap_particles)


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
    # create a block first
    xb, yb = get_2d_block(dx=spacing, length=beam_length, height=beam_height)

    # create a (support) block with required number of layers
    xs, ys = get_2d_block(dx=spacing, length=beam_length,
                          height=boundary_layers * spacing)
    ys -= np.max(yb) - np.min(ys) + spacing

    # import matplotlib.pyplot as plt
    # plt.scatter(xs, ys, s=1)
    # plt.scatter(xb, yb, s=1)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig("geometry", dpi=300)
    # plt.show()

    return xb, yb, xs, ys


def get_fixed_beam_with_clamp(beam_length, beam_height, beam_inside_length,
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


def get_fixed_beam_with_out_clamp(beam_length, beam_height, beam_inside_length,
                                  boundary_length, boundary_height,
                                  boundary_layers, spacing):
    # create a block first
    xb, yb = get_2d_block(dx=spacing, length=beam_length,
                          height=beam_height)

    xs, ys = get_2d_block(dx=spacing, length=boundary_length,
                          height=boundary_height)
    ys -= max(ys) - min(yb) + spacing

    return xb, yb, xs, ys


def find_displacement_index(pa):
    y = pa.y
    max_y = max(y)
    max_y_indices = np.where(y == max_y)[0]
    index = max_y_indices[int(len(max_y_indices)/2)]
    pa.add_property('tip_displacemet_index', type='int',
                    data=np.zeros(len(pa.x)))
    pa.tip_displacemet_index[index] = 1


class Sun2019DamBreakingFLowImpactingAnElasticPlate(Application):
    def add_user_options(self, group):
        group.add_argument("--d0", action="store", type=float, dest="d0",
                           default=1e-3,
                           help="Spacing between the particles")

    def consume_user_options(self):
        # ================================================
        # consume the user options first
        # ================================================
        spacing = self.options.d0

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
        self.fluid_length = 0.146
        self.fluid_height = 0.292
        self.fluid_spacing = spacing
        self.h_fluid = self.hdx * self.fluid_spacing
        self.vref_fluid = np.sqrt(2 * 9.81 * self.fluid_height)
        self.u_max_fluid = np.sqrt(2 * 9.81 * self.fluid_height)
        self.c0_fluid = 10 * self.vref_fluid
        self.nu_fluid = 0.
        self.fluid_density = 1000.0
        self.mach_no_fluid = self.vref_fluid / self.c0_fluid
        self.pb_fluid = self.fluid_density * self.c0_fluid**2.
        self.alpha_fluid = 0.1
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0_fluid * self.h_fluid / 8
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "tank", "gate",
                                             "gate_support"],
            boundaries=["tank", "gate", "gate_support"])

        # ================================================
        # Tank properties
        # ================================================
        self.tank_height = 0.32
        self.tank_length = 0.584
        self.tank_layers = 3
        self.tank_spacing = spacing
        self.wall_layers = 3

        # ================================================
        # properties related to the elastic gate
        # ================================================
        # elastic gate is made of rubber
        self.gate_length = 0.012
        self.gate_height = 0.08
        self.gate_spacing = self.fluid_spacing
        self.gate_rho0 = 2500.
        self.gate_E = 1e6
        self.gate_nu = 0.0
        self.c0_gate = get_speed_of_sound(self.gate_E, self.gate_nu,
                                          self.gate_rho0)
        self.u_max_gate = self.u_max_fluid
        self.mach_no_gate = self.u_max_gate / self.c0_gate
        self.alpha_solid = 1.
        self.beta_solid = 0.
        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["gate"], sources=["gate", "gate_support"],
            boundaries=["gate_support"])

        # ================================================
        # common properties
        # ================================================
        self.boundary_equations = (self.boundary_equations_1 +
                                   self.boundary_equations_2)

        self.dt_fluid = 0.25 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        self.dt_solid = 0.25 * self.h_fluid / (
            (self.gate_E / self.gate_rho0)**0.5 + self.u_max_gate)

    def create_particles(self):
        # ===================================
        # Create fluid
        # ===================================
        xf, yf = get_2d_block(dx=self.fluid_spacing, length=self.fluid_length,
                              height=self.fluid_length)

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
        # xp, yp, xw, yw = get_fixed_beam(self.L, self.H, self.H/2.5,
        #                                 self.wall_layers, self.fluid_spacing)

        xp, yp, xw, yw = get_fixed_beam_with_out_clamp(self.gate_length, self.gate_height,
                                                       self.gate_height/2.5,
                                                       self.gate_length * 3.,
                                                       (self.wall_layers + 1) * self.fluid_spacing,
                                                       self.wall_layers,
                                                       self.fluid_spacing)
        # move the wall onto the tank
        scale = max(yw) - min(tank.y) - (self.wall_layers - 1) * self.fluid_spacing
        yw -= scale
        yp -= scale

        scale = max(fluid.x) - min(xp) + self.fluid_spacing
        xp += scale
        xw += scale

        m = self.gate_rho0 * self.fluid_spacing**2.

        # ===================================
        # Create elastic gate
        # ===================================
        shift = 0.14 + self.fluid_spacing/2.
        xp += shift
        gate = get_particle_array(
            x=xp, y=yp, m=m, h=self.h_fluid, rho=self.gate_rho0, name="gate",
            E=self.gate_E, nu=self.gate_nu, rho_ref=self.gate_rho0,
            constants={
                'n': 4.,
                'spacing0': self.gate_spacing,
            })
        # add post processing variables.
        find_displacement_index(gate)
        gate.add_output_arrays(['tip_displacemet_index'])

        # ===================================
        # Create elastic gate support
        # ===================================
        xw += shift
        gate_support = get_particle_array(
            x=xw, y=yw, m=m, h=self.h_fluid, rho=self.gate_rho0,
            name="gate_support", E=self.gate_E, nu=self.gate_nu,
            rho_ref=self.gate_rho0,
            constants={
                'n': 4.,
                'spacing0': self.gate_spacing,
            })

        # ===========================
        # Adjust the geometry
        # ===========================
        remove_overlap_particles(tank, gate_support, self.fluid_spacing/2.)

        self.scheme.setup_properties([fluid, tank,
                                      gate, gate_support])

        gate.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate.rho_fsi[:] = self.fluid_density

        gate_support.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate_support.rho_fsi[:] = self.fluid_density

        return [fluid, tank, gate, gate_support]

    def create_scheme(self):
        # ctvf = FSIETVFScheme(fluids=['fluid'],
        #                      solids=['tank'],
        #                      structures=['gate'],
        #                      structure_solids=['gate_support'],
        #                      dim=2,
        #                      h_fluid=0.,
        #                      c0_fluid=0.,
        #                      nu_fluid=0.,
        #                      rho0_fluid=0.,
        #                      mach_no_fluid=0.,
        #                      mach_no_structure=0.)

        # substep = FSIETVFSubSteppingScheme(fluids=['fluid'],
        #                                    solids=['tank'],
        #                                    structures=['gate'],
        #                                    structure_solids=['gate_support'],
        #                                    dim=2,
        #                                    h_fluid=0.,
        #                                    c0_fluid=0.,
        #                                    nu_fluid=0.,
        #                                    rho0_fluid=0.,
        #                                    mach_no_fluid=0.,
        #                                    mach_no_structure=0.)

        wcsph = FSIWCSPHScheme(fluids=['fluid'],
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

        wcsph_fluids = FSIWCSPHFluidsScheme(fluids=['fluid'],
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

        # s = SchemeChooser(default='wcsph', substep=substep, wcsph=wcsph)
        s = SchemeChooser(default='wcsph', wcsph=wcsph,
                          wcsph_fluids=wcsph_fluids)

        return s

    def configure_scheme(self):
        dt = self.dt_fluid
        tf = 1.

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=2000)

        self.scheme.configure(
            dim=2,
            h_fluid=self.h_fluid,
            rho0_fluid=self.fluid_density,
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

        # print(eqns)
        # equation = eqns.groups[-1][5].equations[4]
        # equation.sources = ["tank", "fluid", "gate", "gate_support"]
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
        x_0 = pa.x[index]

        t_ctvf, x_ctvf = [], []
        for sd, gate in iter_output(files[::1], 'gate'):
            _t = sd['t']
            t_ctvf.append(_t)
            x_ctvf.append(gate.x[index] - x_0)

        # Numerical data
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # load the data
        data_x_disp_sun_2019 = np.loadtxt(
            os.path.join(directory, 'sun_2019_dam_breaking_flow_impacting_an_elastic_plate_x_displacement_sun_2019_mps_dem.csv'),
            delimiter=',')

        data_x_disp_bogaers_2016 = np.loadtxt(
            os.path.join(directory, 'sun_2019_dam_breaking_flow_impacting_an_elastic_plate_x_displacement_bogaers_2016_qn_ls.csv'),
            delimiter=',')

        data_x_disp_idelsohn_2008 = np.loadtxt(
            os.path.join(directory, 'sun_2019_dam_breaking_flow_impacting_an_elastic_plate_x_displacement_idelsohn_2008_pfrem.csv'),
            delimiter=',')

        data_x_disp_liu_2013 = np.loadtxt(
            os.path.join(directory, 'sun_2019_dam_breaking_flow_impacting_an_elastic_plate_x_displacement_liu_2013_sph.csv'),
            delimiter=',')

        txsun, xdsun = data_x_disp_sun_2019[:, 0], data_x_disp_sun_2019[:, 1]
        txbogaers, xdbogaers = data_x_disp_bogaers_2016[:, 0], data_x_disp_bogaers_2016[:, 1]
        txidelsohn, xdidelsohn = data_x_disp_idelsohn_2008[:, 0], data_x_disp_idelsohn_2008[:, 1]
        txliu, xdliu = data_x_disp_liu_2013[:, 0], data_x_disp_liu_2013[:, 1]

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res,
                 txsun=txsun,
                 xdsun=xdsun,
                 txbogaers=txbogaers,
                 xdbogaers=xdbogaers,
                 txidelsohn=txidelsohn,
                 xdidelsohn=xdidelsohn,
                 txliu=txliu,
                 xdliu=xdliu,
                 t_ctvf=t_ctvf, x_ctvf=x_ctvf)

        # ========================
        # x amplitude figure
        # ========================
        plt.clf()
        plt.plot(txsun, xdsun, "o", label='Sun et al 2019, MPS-DEM')
        plt.plot(txbogaers, xdbogaers, "^", label='Bogaers 2016, QN-LS')
        plt.plot(txidelsohn, xdidelsohn, "+", label='Idelsohn 2008, PFEM')
        plt.plot(txliu, xdliu, "v", label='Liu 2013, SPH')

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


if __name__ == '__main__':
    app = Sun2019DamBreakingFLowImpactingAnElasticPlate()
    app.run()
    app.post_process(app.info_filename)
