import numpy as np
from . import sphere_torus_helpers as sth
from . import sphere_torus_data_gen as dg


def simulate(NUM_POINTS):
    try:
        data = np.load("cached_data/sphere_simulate_data.npy")
    except:
        dt = 0.001
        DT = 0.01
        top = 100
        paths = 100
        x0 = sth.generateInitialConditions_(paths, random=False)
        data_ = sth.generateData_(x0, top, dt)
        periodArray = sth.computeFirstRepeatIndexArray_(data_)
        data_ = sth.truncatePath_(data_, periodArray)

        # globalCoordField = dg.pathsToCloud(globalCoordField_, AMBIENT_DIM)
        data = dg.pathsToCloud(data_, 3)

        np.save("cached_data/sphere_simulate_data.npy", data)

    randomMask = np.full(data.shape[0], False)
    randomMask[:NUM_POINTS] = True
    np.random.shuffle(randomMask)

    data = data[randomMask]
    globalCoordField = sth.f(data)

    return data, globalCoordField
