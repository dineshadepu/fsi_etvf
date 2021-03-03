"""
Lid driven cavity




python lid_driven_cavity.py --openmp --integrator pec --internal-flow --pst ipst --ipst-interval 1 --re 100 -d lid_driven_cavity_etvf_integrator_pec_pst_ipst_interval_1_re_100_output --detailed-output



1.sun2019
2.ipst
3.tvf


"""
import numpy
import numpy as np

from pysph.examples import cavity as LDC
from pysph.sph.equation import Equation, Group
from pysph.sph.basic_equations import SummationDensity
from pysph.tools.sph_evaluator import SPHEvaluator
from fluids import ETVFScheme
from solid_mech import ComputeAuHatETVF


class MoveParticles(Equation):
    def __init__(self, dest, sources):
        self.conv = -1
        super().__init__(dest, sources)

    def post_loop(self, d_idx, d_x, d_y, d_auhat, d_avhat, dt):
        fac = dt*dt
        d_x[d_idx] += fac*d_auhat[d_idx]
        d_y[d_idx] += fac*d_avhat[d_idx]

    def reduce(self, dst, t, dt):
        rho_max = numpy.max(dst.rho)
        rho_min = numpy.min(dst.rho)
        if (rho_max - rho_min)/rho_max < 1e-2:
            self.conv = 1
        else:
            self.conv = -1

    def converged(self):
        return self.conv


def initial_packing(pa, dx, domain):
    np.random.seed(3)
    x = pa.x
    dt = 0.25*dx/(LDC.c0 + 1.0)
    pa.x += np.random.random(len(x))*dx*0.3
    pa.y += np.random.random(len(x))*dx*0.3
    name = pa.name

    eqs = [
        Group(equations=[SummationDensity(dest=name, sources=[name])]),
        Group(
            equations=[
                ComputeAuHatETVF(dest=name, sources=[name], pb=LDC.p0),
                MoveParticles(dest=name, sources=[name])
            ],
            iterate=True, min_iterations=1, max_iterations=500,
            update_nnps=True
        )
    ]
    seval = SPHEvaluator(
        arrays=[pa], equations=eqs, dim=2, domain_manager=domain
    )
    seval.evaluate(0.0, dt)
    print("Finished packing.")


class LidDrivenCavity(LDC.LidDrivenCavity):
    def add_user_options(self, group):
        super().add_user_options(group)
        group.add_argument(
            '--packing', action="store_true", default=False,
            help="Use packing to initialize particles."
        )
        group.add_argument(
            "--pb-factor", action="store", type=float, dest="pb_factor",
            default=1.0,
            help="Use fraction of the background pressure (default: 1.0)."
        )

    def create_particles(self):
        [fluid, solid] = super().create_particles()

        self.scheme.setup_properties([fluid, solid])

        if self.options.packing:
            initial_packing(fluid, self.dx, self.create_domain())
            pi = np.pi
            b = -8.0*pi*pi / self.options.re
            u0, v0, p0 = LDC.exact_solution(
                U=LDC.U, b=b, t=0.0, x=fluid.x, y=fluid.y
            )
            fluid.u[:] = u0
            fluid.v[:] = v0
            fluid.p[:] = p0

        solid.u[:] = 0.
        for i in range(solid.get_number_of_particles()):
            if solid.y[i] > LDC.L:
                if solid.x[i] > 0. and solid.x[i] < LDC.L:
                    solid.u[i] = LDC.Umax

        if self.options.pst == "sun2019":
            # from pysph.base.kernels import (QuinticSpline)
            kernel = self.scheme.scheme.kernel(dim=2)
            # print(kernel)
            wdeltap = kernel.kernel(rij=self.dx, h=LDC.hdx*self.dx)
            fluid.wdeltap[0] = wdeltap
            fluid.n[0] = 4

        return [fluid, solid]

    def create_scheme(self):
        h0 = None
        etvf = ETVFScheme(
            fluids=['fluid'], solids=['solid'], dim=2, rho0=LDC.rho0, c0=LDC.c0, nu=None,
            pb=LDC.p0, h=h0, u_max=5. * LDC.Umax, mach_no=LDC.Umax/LDC.c0,
        )
        schemes = super().create_scheme()
        schemes.schemes['etvf'] = etvf
        schemes.default = 'etvf'
        return schemes

    def configure_scheme(self):
        h0 = LDC.hdx * self.dx
        scheme = self.scheme
        if self.options.scheme == 'etvf':
            scheme.configure(pb=self.options.pb_factor * LDC.p0, nu=self.nu,
                             h=h0)
        super().configure_scheme()


if __name__ == '__main__':
    app = LidDrivenCavity()
    app.run()
    app.post_process(app.info_filename)
