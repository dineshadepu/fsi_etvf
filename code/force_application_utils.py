import numpy as np
from pysph.sph.equation import Equation


class ApplyForceGradual(Equation):
    def __init__(self, dest, sources, delta_fx, delta_fy, delta_fz):
        self.delta_fx = delta_fx
        self.delta_fy = delta_fy
        self.delta_fz = delta_fz
        super(ApplyForceGradual, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_gradual_force_time,
                   d_total_force_applied_x, d_total_force_applied_y,
                   d_total_force_applied_z, d_force_idx, d_m, t):
        if t < d_gradual_force_time[0]:
            if d_idx == 0:
                d_total_force_applied_x[0] += self.delta_fx
                d_total_force_applied_y[0] += self.delta_fy
                d_total_force_applied_z[0] += self.delta_fz

        if d_force_idx[d_idx] == 1:
            d_au[d_idx] += d_total_force_applied_x[0] / d_m[d_idx]
            d_av[d_idx] += d_total_force_applied_y[0] / d_m[d_idx]
            d_aw[d_idx] += d_total_force_applied_z[0] / d_m[d_idx]


class ApplyForceSudden(Equation):
    def __init__(self, dest, sources, fx, fy, fz):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        super(ApplyForceSudden, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_force_idx, d_m, t):
        if d_force_idx[d_idx] == 1:
            d_au[d_idx] += self.fx / d_m[d_idx]
            d_av[d_idx] += self.fy / d_m[d_idx]
            d_aw[d_idx] += self.fz / d_m[d_idx]


def tip_load_force_index_single(plate):
    max_y = np.max(plate.y)
    indices_1 = np.where(max_y == plate.y)[0]
    max_x = np.max(plate.x)

    for i in indices_1:
        if plate.x[i] == max_x:
            break
    plate.force_idx[i] = 1


def tip_load_force_index_distributed(plate):
    max_x = np.max(plate.x)
    indices = np.where(max_x == plate.x)[0]

    plate.force_idx[indices] = 1


def UDL_force_index_single(plate, clamp):
    max_y = np.max(plate.y)
    indices = np.where(max_y == plate.y)[0]

    plate.force_idx[indices] = 1


def UDL_force_index_distributed(plate, clamp):
    plate.force_idx[:] = 1


def setup_properties_for_gradual_force(pa):
    pa.add_constant('total_force_applied_x', 0.)
    pa.add_constant('total_force_applied_y', 0.)
    pa.add_constant('total_force_applied_z', 0.)

    pa.add_constant('gradual_force_time', 0.)

    force_idx = np.zeros_like(pa.x)
    pa.add_property('force_idx', type='int', data=force_idx)


def setup_properties_for_sudden_force(pa):
    force_idx = np.zeros_like(pa.x)
    pa.add_property('force_idx', type='int', data=force_idx)
