import numpy as np
import matplotlib.pyplot as plt
from pysph.tools.geometry import get_2d_tank, get_2d_block


def hydrostatic_tank_2d(fluid_length, fluid_height, tank_height, tank_layers,
                        fluid_spacing, tank_spacing):
    xt, yt = get_2d_tank(dx=tank_spacing, length=fluid_length + 2. * tank_spacing,
                         height=tank_height,
                         num_layers=tank_layers)
    xf, yf = get_2d_block(dx=fluid_spacing, length=fluid_length,
                          height=fluid_height, center=[-1.5, 1])

    xf += (np.min(xt) - np.min(xf))
    yf -= (np.min(yf) - np.min(yt))

    # now adjust inside the tank
    xf += tank_spacing * (tank_layers)
    yf += tank_spacing * (tank_layers)

    return xf, yf, xt, yt


def test_hydrostatic_tank():
    xf, yf, xt, yt = hydrostatic_tank_2d(1., 1., 1.5, 3, 0.1, 0.1/2.)
    plt.scatter(xt, yt)
    plt.scatter(xf, yf)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()


# test_hydrostatic_tank()
