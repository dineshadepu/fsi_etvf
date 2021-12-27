"""A delta SPH-SPIM coupled method for fluid-structure interaction
problems

https://doi.org/10.1016/j.jfluidstructs.2020.103210

Also at

Coupling edge-based smoothed finite element method with smoothed particle
hydrodynamics for fluid structure interaction problems

https://doi.org/10.1016/j.oceaneng.2021.108772

An enhanced ISPH-SPH coupled method for simulation of incompressible
fluid-elastic structure interactions

https://doi.org/10.1016/j.cpc.2018.05.012

# Wedge geometry description
A non-intrusive partitioned approach to couple smoothed particle hydrodynamics
and finite element methods for transient fluid-structure interaction problems
with large interface motion

https://doi.org/10.1007/s00466-015-1131-8

"""

import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

# from rigid_fluid_coupling import RigidFluidCouplingScheme
# from geometry import hydrostatic_tank_2d, create_tank_2d_from_block_2d

from pysph.examples.solid_mech.impact import add_properties
# from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
#                                                                create_fluid,
#                                                                create_sphere)
from pysph.tools.geometry import get_2d_block, rotate

# from fsi_coupling import FSIETVFScheme, FSIETVFSubSteppingScheme
from fsi_wcsph import (FSIWCSPHScheme, FSIWCSPHSubSteppingScheme,
                       FSIWCSPHFluidsScheme, FSIWCSPHFluidsSubSteppingScheme)
from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)


def find_displacement_index(pa):
    x = pa.x
    y = pa.y
    min_y = min(y)
    min_y_indices = np.where(y == min_y)[0]
    index = min_y_indices[int(len(min_y_indices)/2)]
    pa.add_property('A_index', type='int',
                    data=np.zeros(len(pa.x)))
    pa.A_index[index] = 1


def find_velocity_indices(pa):
    min_x = min(pa.x)
    max_x = max(pa.x)
    pa.add_property('velocity_indices', type='int',
                    data=np.zeros(len(pa.x)))

    index = (pa.x == min_x)
    pa.velocity_indices[index] = 1

    index = (pa.x == max_x)
    pa.velocity_indices[index] = 1

    min_y = min(pa.y)
    x_min_y = pa.x[pa.y == min_y]
    index = pa.x == x_min_y
    pa.velocity_indices[index] = 1


def get_hydrostatic_tank_with_fluid(fluid_length=1., fluid_height=2.,
                                    tank_height=2.3, tank_layers=2,
                                    fluid_spacing=0.1):
    import matplotlib.pyplot as plt

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

    # create the bottom part
    xt_3, yt_3 = get_2d_block(dx=fluid_spacing,
                              length=max(xt_2) - min(xt_1),
                              height=tank_layers*fluid_spacing)
    yt_3 += min(yf) - max(yt_3) - fluid_spacing

    xt = np.concatenate([xt_1, xt_2, xt_3])
    yt = np.concatenate([yt_1, yt_2, yt_3])

    # plt.scatter(xf, yf, s=10)
    # plt.scatter(xt, yt, s=10)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig("geometry", dpi=300)
    # plt.show()

    return xf, yf, xt, yt


def get_wedge(wedge_size, wedge_width, slope, spacing):
    dx = spacing
    layers = wedge_width / spacing
    no_of_layers = layers
    _x = np.arange(-wedge_size, wedge_size, dx)
    _y = np.arange(-dx, (no_of_layers)*dx, dx)
    xw, yw = np.meshgrid(_x, _y)
    xw = xw.ravel()
    yw = yw.ravel()
    yw += - min(yw)

    # find the index on right panel in middle (fig 21)
    max_x = max(xw)

    B_index = (yw == min(yw)) & (
        (xw > max_x / 2. - spacing / 2.) & (xw < max_x / 2. + spacing / 2.))

    # find the indices of the right panel (fig 22)
    right_panel = (yw == min(yw)) & (xw > 0.)

    # find the indices of B, C (Fig 19 b)
    C_index = (yw == min(yw)) & (
        (xw > max_x - wedge_size/4. - spacing / 2.) &
        (xw < max_x - wedge_size/4. + spacing / 2.))

    yw = yw + xw * np.tan(np.sign(xw)*slope*np.pi/180)

    return xw, yw, B_index, right_panel, C_index


class WaterEntryOfElasticWedge(Application):
    def add_user_options(self, group):
        group.add_argument("--d0", action="store", type=float, dest="d0",
                           default=5 * 1e-3,
                           help="Spacing between the particles")

    def consume_user_options(self):
        # ================================================
        # consume the user options first
        # ================================================
        self.d0 = self.options.d0
        spacing = self.d0
        print("Spacing is", spacing)
        self.hdx = 1.0
        self.dim = 2

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
        self.fluid_length = 6.
        self.fluid_height = 2.2
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing
        self.h_fluid = self.hdx * self.fluid_spacing
        self.vref_fluid = 30
        # Coupling edge-based smoothed finite element method with smoothed
        # particle hydrodynamics for fluid structure interaction problems
        self.c0_fluid = 1500.
        self.nu_fluid = 0.
        self.rho0_fluid = 1000.0
        self.mach_no_fluid = self.vref_fluid / self.c0_fluid
        self.pb_fluid = self.fluid_density * self.c0_fluid**2.
        self.alpha_fluid = 0.1
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0_fluid * self.h_fluid / 8
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "tank", "wedge"],
            boundaries=["tank", "wedge"])

        # ================================================
        # Tank properties
        # ================================================
        self.tank_height = 7.
        self.tank_length = 3.
        self.tank_layers = 2
        self.tank_spacing = spacing

        # for boundary particles
        self.seval = None

        # ================================================
        # properties related to the elastic wedge
        # ================================================
        # elastic wedge is made of rubber
        # self.wedge_size = 0.60925
        self.wedge_size = 0.60
        self.wedge_width = 0.04
        self.wedge_spacing = self.fluid_spacing
        self.wedge_rho0 = 2700
        self.wedge_E = 67.5 * 1e9
        self.wedge_nu = 0.34
        self.c0_wedge = get_speed_of_sound(self.wedge_E, self.wedge_nu,
                                           self.wedge_rho0)
        # print(self.c0_wedge)
        # self.c0 = 5960
        print("speed of sound is")
        print(self.c0_wedge)
        self.pb_wedge = self.wedge_rho0 * self.c0_wedge**2

        self.edac_alpha = 0.5

        self.edac_nu = self.edac_alpha * self.c0_wedge * self.h_fluid / 8

        # attributes for Sun PST technique
        # dummy value, will be updated in consume user options
        self.u_max_wedge = 30.
        self.mach_no_wedge = self.u_max_wedge / self.c0_wedge

        # for pre step
        # self.seval = None

        # boundary equations
        # self.boundary_equations = get_boundary_identification_etvf_equations(
        #     destinations=["wedge"], sources=["wedge"])
        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["wedge"], sources=["wedge"],
            boundaries=None)

        self.boundary_equations = self.boundary_equations_1 + self.boundary_equations_2

        self.wall_layers = 2

        self.artificial_stress_eps = 0.3

        self.dt_fluid = 0.25 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        print("dt fluid is", self.dt_fluid)
        self.dt_solid = 0.25 * self.h_fluid / (
            (self.wedge_E / self.wedge_rho0)**0.5 + self.u_max_wedge)

    def create_particles(self):
        # ===================================
        # Create fluid
        # ===================================
        xf, yf, xt, yt = get_hydrostatic_tank_with_fluid(self.fluid_length,
                                                         self.fluid_height,
                                                         self.tank_length,
                                                         self.tank_layers,
                                                         self.fluid_spacing)

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
        fluid.p[:] = - self.fluid_density * self.gy * (max(fluid.y) -
                                                       fluid.y[:])

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

        # =============================================
        # Only structures part particle properties
        # =============================================
        xp, yp, B_index, right_panel_indices, C_index = get_wedge(self.wedge_size,
                                                                  self.wedge_width,
                                                                  10.,
                                                                  self.fluid_spacing)
        # make sure that the wedge intersection with wall starts at the 0.
        min_xp = np.min(xp)

        # add this to the wedge and wall
        xp += abs(min_xp)

        m = self.wedge_rho0 * self.fluid_spacing**2.

        # ===================================
        # Create elastic wedge
        # ===================================
        xp += self.fluid_length
        wedge = get_particle_array(
            x=xp, y=yp, m=m, h=self.h_fluid, rho=self.wedge_rho0,
            E=self.wedge_E, nu=self.wedge_nu, rho_ref=self.wedge_rho0,
            name="wedge",
            constants={
                'n': 4.,
                'spacing0': self.wedge_spacing,
            })
        # add post processing variables.
        find_displacement_index(wedge)
        find_velocity_indices(wedge)

        wedge.add_property('vhat')
        wedge.v[:] = -30.
        wedge.vhat[:] = -30.

        wedge.y += max(fluid.y) - min(wedge.y) + self.fluid_spacing
        wedge.x -= min(wedge.x) - min(fluid.x)

        # wedge.x += - self.wedge_size
        wedge.x += self.fluid_length / 2. - self.wedge_size

        self.scheme.setup_properties([fluid, tank, wedge])

        wedge.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        wedge.rho_fsi[:] = self.fluid_density

        # setup properties for post processing
        # wedge_post_process_properties(wedge)
        wedge.add_property('right_panel_indices')
        wedge.add_property('A_index')
        wedge.add_property('B_index')
        wedge.add_property('C_index')
        wedge.right_panel_indices[:] = right_panel_indices
        wedge.B_index[:] = B_index
        wedge.C_index[:] = C_index

        wedge.add_output_arrays(['right_panel_indices',
                                 'A_index',
                                 'B_index',
                                 'C_index',
                                 'velocity_indices'])
        wedge.add_output_arrays(['is_boundary', 'p_fsi'])
        return [fluid, tank, wedge]

    def create_scheme(self):
        # ctvf = FSIETVFScheme(fluids=['fluid'],
        #                      solids=['tank'],
        #                      structures=['wedge'],
        #                      structure_solids=None,
        #                      dim=2,
        #                      h_fluid=0.,
        #                      c0_fluid=0.,
        #                      nu_fluid=0.,
        #                      rho0_fluid=0.,
        #                      mach_no_fluid=0.,
        #                      mach_no_structure=0.)

        # substep = FSIETVFSubSteppingScheme(fluids=['fluid'],
        #                                    solids=['tank'],
        #                                    structures=['wedge'],
        #                                    structure_solids=None,
        #                                    dim=2,
        #                                    h_fluid=0.,
        #                                    c0_fluid=0.,
        #                                    nu_fluid=0.,
        #                                    rho0_fluid=0.,
        #                                    mach_no_fluid=0.,
        #                                    mach_no_structure=0.)

        wcsph = FSIWCSPHScheme(fluids=['fluid'],
                               solids=['tank'],
                               structures=['wedge'],
                               structure_solids=None,
                               dim=2,
                               h_fluid=0.,
                               c0_fluid=0.,
                               nu_fluid=0.,
                               rho0_fluid=0.,
                               mach_no_fluid=0.,
                               mach_no_structure=0.)

        wcsph_fluids = FSIWCSPHFluidsScheme(fluids=['fluid'],
                                            solids=['tank'],
                                            structures=['wedge'],
                                            structure_solids=None,
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
        dt = 0.125 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        # TODO: This has to be changed for solid
        # dt = 0.25 * self.h_fluid / (
        #     (self.wedge_E / self.wedge_rho0)**0.5 + self.u_max_wedge)
        # dt = 0.125 * self.fluid_spacing * self.hdx / (self.c0_fluid * 1.1)
        # print("time step fluid dt", dt)

        # print("DT: %s" % dt)
        dt = 2. * 1e-7
        tf = 0.0025

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=200)

        # print(self.scheme.name)
        self.scheme.configure(
            dim=2,
            h_fluid=self.h_fluid,
            rho0_fluid=self.rho0_fluid,
            pb_fluid=self.pb_fluid,
            c0_fluid=self.c0_fluid,
            nu_fluid=0.0,
            mach_no_fluid=self.mach_no_fluid,
            mach_no_structure=self.mach_no_wedge,
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
        #     equation.sources = ["tank", "fluid", "wedge", "wedge_support"]
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
        for pa in self.particles:
            if pa.name == 'wedge':
                pa.u[pa.velocity_indices == 1] = 0.
                pa.v[pa.velocity_indices == 1] = -30.

                pa.uhat[pa.velocity_indices == 1] = 0.
                pa.vhat[pa.velocity_indices == 1] = -30.

                pa.au[pa.velocity_indices == 1] = 0.
                pa.av[pa.velocity_indices == 1] = 0.

                pa.auhat[pa.velocity_indices == 1] = 0.
                pa.avhat[pa.velocity_indices == 1] = 0.

                pa.as00[pa.velocity_indices == 1] = 0.
                pa.as01[pa.velocity_indices == 1] = 0.
                pa.as02[pa.velocity_indices == 1] = 0.
                pa.as11[pa.velocity_indices == 1] = 0.
                pa.as12[pa.velocity_indices == 1] = 0.
                pa.as22[pa.velocity_indices == 1] = 0.

                pa.arho[pa.velocity_indices == 1] = 0.

    def post_process(self, fname):
        from pysph.solver.utils import iter_output, load
        import os
        from matplotlib import pyplot as plt

        info = self.read_info(fname)
        files = self.output_files

        # Fig 21 of Sun 2021, \delta SPH - SPIM
        data = load(files[0])
        arrays = data['arrays']
        pa = arrays['wedge']
        # Displacement of the middle of the wedge (Index B)
        index_B = np.where(pa.B_index == 1)[0][0]
        y_0 = pa.y[index_B]

        # for pressure computation
        index_C = np.where(pa.C_index == 1)[0][0]
        index_A = np.where(pa.A_index == 1)[0][0]
        patActvf, patCctvf = [], []

        files = files[0::1]
        t_ctvf, yatBctvf = [], []
        for sd, wedge in iter_output(files, 'wedge'):
            _t = sd['t']
            t_ctvf.append(_t)

            # y_tmp = y_0 - 30. * _t
            y_tmp = y_0
            yatBctvf.append(wedge.y[index_B] - y_tmp)

            # save the pressure data
            patActvf.append(wedge.p_fsi[index_A])
            patCctvf.append(wedge.p_fsi[index_C])

        # Numerical data
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # ================================ #
        # Plot the displacement at point B
        # ================================ #
        # load the data
        # The data of mid point deflection is extracted from "A coupled
        # smoothed  particle hydrodynamic and finite particle method: An
        # efficient approach for fluid-solid interaction problems involving
        # free-surface flow and solid failure"
        data_y_at_b_disp_analytical = np.loadtxt(
            os.path.join(directory, 'zhang_2021_high_speed_water_entry_of_an_elastic_wedge_y_displacement_analytical.csv'),
            delimiter=',')

        data_y_at_b_disp_fourey_2010 = np.loadtxt(
            os.path.join(directory, 'zhang_2021_high_speed_water_entry_of_an_elastic_wedge_y_displacement_fourey_2010_sph_fem.csv'),
            delimiter=',')

        data_y_at_b_disp_li_2015 = np.loadtxt(
            os.path.join(directory, 'zhang_2021_high_speed_water_entry_of_an_elastic_wedge_y_displacement_li_2015_sph_fem.csv'),
            delimiter=',')

        tyatBanalytical, yatBanalytical = data_y_at_b_disp_analytical[:, 0], data_y_at_b_disp_analytical[:, 1]
        tyatBfourey, yatBfourey = data_y_at_b_disp_fourey_2010[:, 0], data_y_at_b_disp_fourey_2010[:, 1]
        tyatBli, yatBli = data_y_at_b_disp_li_2015[:, 0], data_y_at_b_disp_li_2015[:, 1]

        plt.clf()

        # yatBanalytical reads y_at_b_analytical
        plt.plot(tyatBanalytical, yatBanalytical, "o-", label='Analytical')
        plt.plot(tyatBfourey, yatBfourey, "^-", label='Fourey 2010, SPH-FEM')
        plt.plot(tyatBli, yatBli, "+-", label='Li 2015, SPH-FEM')
        plt.plot(t_ctvf, yatBctvf, "-", label='CTVF')
        # plt.plot(t_ctvf, y_wedge, "-", label='Rigid')
        plt.title('Displacement at mid point of wedge')

        plt.xlabel('t')
        plt.ylabel('Displacement')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "b_displacement_with_t.png")
        plt.savefig(fig, dpi=300)
        # ================================ #
        # Plot the displacement at point B
        # ================================ #

        # ============================ #
        # Plot the pressure at point A
        # ============================ #
        # load the data
        # The data of pressure at point A and C deflection is extracted from
        # " A SPH-SPIM coupled method for fluid-structure interaction problems"

        data_p_at_A_analytical = np.loadtxt(
            os.path.join(directory, 'zhang_2021_high_speed_water_entry_of_an_elastic_wedge_pressure_point_A_analytical.csv'),
            delimiter=',')

        data_p_at_A_khayyer_2018_sph = np.loadtxt(
            os.path.join(directory, 'zhang_2021_high_speed_water_entry_of_an_elastic_wedge_pressure_point_A_khayyer_2018_sph.csv'),
            delimiter=',')

        data_p_at_A_oger_2010_sph = np.loadtxt(
            os.path.join(directory, 'zhang_2021_high_speed_water_entry_of_an_elastic_wedge_pressure_point_A_oger_2010_sph.csv'),
            delimiter=',')

        tpatAanalytical, patAanalytical = data_p_at_A_analytical[:, 0], data_p_at_A_analytical[:, 1]
        tpatAkhayyer, patAkhayyer = data_p_at_A_khayyer_2018_sph[:, 0], data_p_at_A_khayyer_2018_sph[:, 1]
        tpatAoger, patAoger = data_p_at_A_oger_2010_sph[:, 0], data_p_at_A_oger_2010_sph[:, 1]

        patAanalytical *= 1e7
        patAkhayyer *= 1e7
        patAoger *= 1e7

        plt.clf()

        # yatBanalytical reads y_at_b_analytical
        plt.plot(tpatAanalytical, patAanalytical, "o-", label='Analytical')
        plt.plot(tpatAkhayyer, patAkhayyer, "^-", label='Khayyer 2018, SPH')
        plt.plot(tpatAoger, patAoger, "+-", label='Oger 2010, SPH')
        plt.plot(t_ctvf, patActvf, "-", label='CTVF')
        plt.title('Pressure at point A of wedge')

        plt.xlabel('t')
        plt.ylabel('pressure')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "A_pressure_with_t.png")
        plt.savefig(fig, dpi=300)

        # ============================ #
        # Plot the pressure at point A
        # ============================ #

        # ============================ #
        # Plot the pressure at point C
        # ============================ #
        # load the data
        # The data of pressure at point A and C deflection is extracted from
        # " A SPH-SPIM coupled method for fluid-structure interaction problems"

        data_p_at_C_analytical = np.loadtxt(
            os.path.join(directory, 'zhang_2021_high_speed_water_entry_of_an_elastic_wedge_pressure_point_C_analytical.csv'),
            delimiter=',')

        data_p_at_C_khayyer_2018_sph = np.loadtxt(
            os.path.join(directory, 'zhang_2021_high_speed_water_entry_of_an_elastic_wedge_pressure_point_C_khayyer_2018_sph.csv'),
            delimiter=',')

        data_p_at_C_oger_2010_sph = np.loadtxt(
            os.path.join(directory, 'zhang_2021_high_speed_water_entry_of_an_elastic_wedge_pressure_point_C_oger_2010_sph.csv'),
            delimiter=',')

        tpatCanalytical, patCanalytical = data_p_at_C_analytical[:, 0], data_p_at_C_analytical[:, 1]
        tpatCkhayyer, patCkhayyer = data_p_at_C_khayyer_2018_sph[:, 0], data_p_at_C_khayyer_2018_sph[:, 1]
        tpatCoger, patCoger = data_p_at_C_oger_2010_sph[:, 0], data_p_at_C_oger_2010_sph[:, 1]

        patCanalytical *= 1e7
        patCkhayyer *= 1e7
        patCoger *= 1e7

        plt.clf()

        # yatBanalytical reads y_at_b_analytical
        plt.plot(tpatCanalytical, patCanalytical, "o-", label='Analytical')
        plt.plot(tpatCkhayyer, patCkhayyer, "^-", label='Khayyer 2018, SPH')
        plt.plot(tpatCoger, patCoger, "+-", label='Oger 2010, SPH')
        plt.plot(t_ctvf, patCctvf, "-", label='CTVF')
        plt.title('Pressure at point C of wedge')

        plt.xlabel('t')
        plt.ylabel('pressure')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "C_pressure_with_t.png")
        plt.savefig(fig, dpi=300)

        # ============================ #
        # Plot the pressure at point C
        # ============================ #

        # save all the data
        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res,
                 tyatBanalytical=tyatBanalytical,
                 yatBanalytical=yatBanalytical,

                 tyatBfourey=tyatBfourey,
                 yatBfourey=yatBfourey,

                 tyatBli=tyatBli,
                 yatBli=yatBli,

                 t_ctvf=t_ctvf,
                 yatBctvf=yatBctvf,

                 tpatCanalytical=tpatCanalytical,
                 patCanalytical=patCanalytical,

                 tpatCkhayyer=tpatCkhayyer,
                 patCkhayyer=patCkhayyer,

                 tpatCoger=tpatCoger,
                 patCoger=patCoger,

                 patCctvf=patCctvf,

                 tpatAanalytical=tpatAanalytical,
                 patAanalytical=patAanalytical,

                 tpatAkhayyer=tpatAkhayyer,
                 patAkhayyer=patAkhayyer,

                 tpatAoger=tpatAoger,
                 patAoger=patAoger,

                 patActvf=patActvf)


if __name__ == '__main__':
    app = WaterEntryOfElasticWedge()
    app.run()
    app.post_process(app.info_filename)
