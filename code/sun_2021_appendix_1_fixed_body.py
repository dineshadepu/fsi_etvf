"""
Appendix 1

https://www.sciencedirect.com/science/article/pii/S0029801820314608#appsec1
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

from fsi_coupling import FSIETVFScheme, FSIETVFSubSteppingScheme
from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)

from pysph.sph.solid_mech.basic import (get_speed_of_sound, get_bulk_mod,
                                        get_shear_modulus)
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import add_bool_argument

from geometry import hydrostatic_tank_2d, create_tank_2d_from_block_2d
from pysph.tools.geometry import (remove_overlap_particles)


def get_hydrostatic_tank_with_fluid(fluid_length=1., fluid_height=2., tank_height=2.3,
                                    tank_layers=2, fluid_spacing=0.1):
    import matplotlib.pyplot as plt

    xf, yf = get_2d_block(dx=fluid_spacing,
                          length=fluid_length+fluid_spacing/2.,
                          height=fluid_height+fluid_spacing/2.)

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

    xt = np.concatenate([xt_1, xt_2])
    yt = np.concatenate([yt_1, yt_2])

    # plt.scatter(xf, yf, s=10)
    # plt.scatter(xt, yt, s=10)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.savefig("geometry", dpi=300)
    # plt.show()

    return xf, yf, xt, yt


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


class BeamInQuiescentTank(Application):
    def add_user_options(self, group):
        group.add_argument("--d0", action="store", type=float, dest="d0",
                           default=0.005,
                           help="Spacing between the particles")

        group.add_argument("--gate-rho", action="store", type=float, dest="gate_rho",
                           default=1500.,
                           help="Density of the gate (Defaults to 1500.)")

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
        self.gy = -1.
        self.gz = 0.
        self.dim = 2
        self.seval = None

        # ================================================
        # Fluid properties
        # ================================================
        self.fluid_length = 0.8
        self.fluid_height = 0.6
        self.fluid_spacing = spacing
        self.h_fluid = self.hdx * self.fluid_spacing
        self.vref_fluid = np.sqrt(2 * 9.81 * self.fluid_height)
        self.c0_fluid = 10 * self.vref_fluid
        self.nu_fluid = 0.
        self.rho0_fluid = 1000.0
        self.fluid_density = self.rho0_fluid
        self.mach_no_fluid = self.vref_fluid / self.c0_fluid
        self.pb_fluid = self.rho0_fluid * self.c0_fluid**2.
        self.alpha_fluid = 1.
        self.edac_alpha = 0.5
        self.edac_nu = self.edac_alpha * self.c0_fluid * self.h_fluid / 8
        self.boundary_equations_1 = get_boundary_identification_etvf_equations(
            destinations=["fluid"], sources=["fluid", "tank", "gate"],
            boundaries=["tank", "gate"])

        # ================================================
        # Tank properties
        # ================================================
        self.tank_height = 0.8
        self.tank_length = 0.6
        self.tank_layers = 3
        self.tank_spacing = spacing
        self.wall_layers = 2

        # ================================================
        # properties related to the elastic gate
        # ================================================
        # elastic gate is made of rubber
        self.gate_length = 0.35
        self.gate_height = 0.02
        self.gate_spacing = self.fluid_spacing
        self.gate_rho0 = self.options.gate_rho
        self.gate_E = 1.4 * 1e6
        self.gate_nu = 0.4
        self.c0_gate = get_speed_of_sound(self.gate_E, self.gate_nu,
                                          self.gate_rho0)
        self.u_max_gate = 0.004
        self.mach_no_gate = self.u_max_gate / self.c0_gate
        self.alpha_solid = 1.
        self.beta_solid = 0.
        self.boundary_equations_2 = get_boundary_identification_etvf_equations(
            destinations=["gate"], sources=["gate"], boundaries=None)

        self.boundary_equations = self.boundary_equations_1 + self.boundary_equations_2

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
        xf, yf, xt, yt = hydrostatic_tank_2d(
            self.fluid_length, self.fluid_height, self.tank_height,
            self.tank_layers, self.fluid_spacing, self.fluid_spacing)

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
                                  rad_s=self.fluid_spacing/2.,
                                  name="tank")

        # =============================================
        # Only structures part particle properties
        # =============================================
        xp, yp, xw, yw = get_fixed_beam_no_clamp(self.gate_length,
                                                 self.gate_height,
                                                 self.gate_length/4.,
                                                 self.tank_layers,
                                                 self.fluid_spacing)
        # translate the beam into the tank
        xp += -0.10 - min(xp)
        yp += 0.3 - max(yp)

        m = self.gate_rho0 * self.fluid_spacing**2.

        # ===================================
        # Create elastic gate
        # ===================================
        # xp += self.fluid_length
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

        self.scheme.setup_properties([fluid, tank, gate])

        gate.m_fsi[:] = self.fluid_density * self.fluid_spacing**2.
        gate.rho_fsi[:] = self.fluid_density
        gate.m_frac[:] = gate.m_fsi[:] / gate.m[:]

        remove_overlap_particles(fluid, gate, self.fluid_spacing)

        gate.x[:] += self.fluid_spacing/2.

        gate.add_output_arrays(['tip_displacemet_index'])

        # ================================
        # set the normals of the tank
        # ================================
        set_normals_tank(tank, self.fluid_spacing)

        return [fluid, tank, gate]

    def create_scheme(self):
        substep = FSIETVFSubSteppingScheme(fluids=['fluid'],
                                           solids=['tank'],
                                           structures=['gate'],
                                           structure_solids=None,
                                           dim=2,
                                           h_fluid=0.,
                                           c0_fluid=0.,
                                           nu_fluid=0.,
                                           rho0_fluid=0.,
                                           mach_no_fluid=0.,
                                           mach_no_structure=0.)

        s = SchemeChooser(default='substep', substep=substep)

        return s

    def configure_scheme(self):
        dt = self.dt_fluid
        tf = 20.0
        print("timestep is", dt)

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
            alpha_fluid=1.,
            alpha_solid=1.,
            beta_solid=0,
            dt_fluid=self.dt_fluid,
            dt_solid=self.dt_solid
        )

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

        files = files[0::1]
        t_ctvf, amplitude_ctvf = [], []
        for sd, gate in iter_output(files, 'gate'):
            _t = sd['t']
            t_ctvf.append(_t)
            amplitude_ctvf.append((gate.y[index] - y_0) * 1)

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res,
                 amplitude_ctvf=amplitude_ctvf,
                 t_ctvf=t_ctvf)

        plt.clf()
        plt.plot(t_ctvf, amplitude_ctvf, "-", label='PySPH')

        plt.xlabel('t')
        plt.ylabel('amplitude')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "amplitude_with_t.png")
        plt.savefig(fig, dpi=300)

        plt.clf()


if __name__ == '__main__':
    app = BeamInQuiescentTank()
    app.run()
    app.post_process(app.info_filename)
