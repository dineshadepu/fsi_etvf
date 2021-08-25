from pysph.sph.equation import Equation


class ShepardInterpolateCharacteristics(Equation):
    def initialize(self, d_idx, d_J1, d_J2u, d_J3u, d_J2v, d_J3v):
        d_J1[d_idx] = 0.0
        d_J2u[d_idx] = 0.0
        d_J3u[d_idx] = 0.0

        d_J2v[d_idx] = 0.0
        d_J3v[d_idx] = 0.0

    def loop(self, d_idx, d_J1, d_J2u, s_J1, s_J2u, d_J3u, s_J3u, s_idx, s_J2v,
             s_J3v, d_J2v, d_J3v, WIJ):
        d_J1[d_idx] += s_J1[s_idx] * WIJ
        d_J2u[d_idx] += s_J2u[s_idx] * WIJ
        d_J3u[d_idx] += s_J3u[s_idx] * WIJ

        d_J2v[d_idx] += s_J2v[s_idx] * WIJ
        d_J3v[d_idx] += s_J3v[s_idx] * WIJ

    def post_loop(self, d_idx, d_J1, d_J2u, d_wij, d_avg_j2u, d_avg_j1, d_J3u,
                  d_avg_j3u, d_J2v, d_J3v, d_avg_j2v, d_avg_j3v):
        if d_wij[d_idx] > 1e-14:
            d_J1[d_idx] /= d_wij[d_idx]
            d_J2u[d_idx] /= d_wij[d_idx]
            d_J3u[d_idx] /= d_wij[d_idx]

            d_J2v[d_idx] /= d_wij[d_idx]
            d_J3v[d_idx] /= d_wij[d_idx]

        else:
            d_J1[d_idx] = d_avg_j1[0]
            d_J2u[d_idx] = d_avg_j2u[0]
            d_J3u[d_idx] = d_avg_j3u[0]

            d_J2v[d_idx] = d_avg_j2v[0]
            d_J3v[d_idx] = d_avg_j3v[0]

    def reduce(self, dst, t, dt):
        dst.avg_j1[0] = numpy.average(dst.J1[dst.wij > 0.0001])
        dst.avg_j2u[0] = numpy.average(dst.J2u[dst.wij > 0.0001])
        dst.avg_j3u[0] = numpy.average(dst.J3u[dst.wij > 0.0001])

        dst.avg_j2v[0] = numpy.average(dst.J2v[dst.wij > 0.0001])
        dst.avg_j3v[0] = numpy.average(dst.J3v[dst.wij > 0.0001])


class EvaluateCharacterisctics(Equation):
    def __init__(self, dest, sources, c_ref, rho_ref, u_ref, v_ref, p_ref):
        self.c_ref = c_ref
        self.rho_ref = rho_ref
        self.u_ref = u_ref
        self.v_ref = v_ref
        self.p_ref = p_ref
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_p, d_rho, d_J1, d_J2u, d_J3u, d_J2v,
                   d_J3v, d_y):
        a = self.c_ref
        rho_ref = self.rho_ref

        rho = d_rho[d_idx]
        p_ref = self.p_ref
        pdiff = d_p[d_idx] - p_ref
        udiff = d_u[d_idx] - self.u_ref
        vdiff = d_v[d_idx] - self.v_ref

        d_J1[d_idx] = -a * a * (rho - rho_ref) + pdiff
        d_J2u[d_idx] = rho * a * udiff + pdiff
        d_J3u[d_idx] = -rho * a * udiff + pdiff

        d_J2v[d_idx] = rho * a * vdiff + pdiff
        d_J3v[d_idx] = -rho * a * vdiff + pdiff


class EvaluatePropertyfromCharacteristics(Equation):
    def __init__(self, dest, sources, c_ref, rho_ref, u_ref, v_ref, p_ref):
        self.c_ref = c_ref
        self.rho_ref = rho_ref
        self.u_ref = u_ref
        self.v_ref = v_ref
        self.p_ref = p_ref
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_J1, d_J2u, d_J2v, d_J3u, d_J3v, d_rho,
                   d_u, d_v, d_xn, d_yn):
        a = self.c_ref
        a2_1 = 1.0/(a*a)
        rho = d_rho[d_idx]
        xn = d_xn[d_idx]
        yn = d_yn[d_idx]

        # Characteristic in the downstream direction.
        if xn > 0.5 or yn > 0.5:
            J1 = d_J1[d_idx]
            J3 = 0.0
            if xn > 0.5:
                J2 = d_J2u[d_idx]
                d_u[d_idx] = self.u_ref + (J2 - J3) / (2 * rho * a)
            else:
                J2 = d_J2v[d_idx]
                d_v[d_idx] = self.v_ref + (J2 - J3) / (2 * rho * a)

            d_rho[d_idx] = self.rho_ref + a2_1 * (-J1 + 0.5 * (J2 + J3))
            d_p[d_idx] = self.p_ref + 0.5 * (J2 + J3)
        # Characteristic in the upstream direction.
        else:
            J1 = 0.0
            J2 = 0.0
            if xn < -0.5:
                J3 = d_J3u[d_idx]
                d_u[d_idx] = self.u_ref + (J2 - J3) / (2 * rho * a)
            else:
                J3 = d_J3v[d_idx]
                d_v[d_idx] = self.v_ref + (J2 - J3) / (2 * rho * a)

            d_rho[d_idx] = self.rho_ref + a2_1 * (-J1 + 0.5 * (J2 + J3))
            d_p[d_idx] = self.p_ref + 0.5 * (J2 + J3)


class EvaluateNumberDensity(Equation):
    def initialize(self, d_idx, d_wij, d_wij2):
        d_wij[d_idx] = 0.0
        d_wij2[d_idx] = 0.0

    def loop(self, d_idx, d_wij, d_wij2, XIJ, HIJ, RIJ, SPH_KERNEL):
        wij = SPH_KERNEL.kernel(XIJ, RIJ, HIJ)
        wij2 = SPH_KERNEL.kernel(XIJ, RIJ, 0.5*HIJ)
        d_wij[d_idx] += wij
        d_wij2[d_idx] += wij2
