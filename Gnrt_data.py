from shapes import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def padding(x, width, value=0):
    return np.pad(x, width, pad_with, padder=value)


''' Generate a signal of specific shape '''

# Circle
def Gnrt_circle(img_size, center, radius):
    canvas = np.zeros(img_size)
    x, y = center
    C = circle(canvas, x, y, radius)
    return C


# Circles
def Gnrt_circles(img_size, center_list, radius_list):
    canvas = np.zeros(img_size)
    C = canvas.copy()
    for i in range(len(radius_list)):
        x, y = center_list[i]
        r = radius_list[i]
        C = circle(C, x, y, r)
    return C


def Gnrt_3D(dims, shape="ball", center=None, radius=None):
    canvas = np.zeros(dims)
    if shape == "ball":
        gnrt_shape = ball(canvas, center, radius)
    elif shape == 'balls':
        gnrt_shape = ball(canvas, [24, 24, 24], 7)
        gnrt_shape = ball(gnrt_shape, [24, 40, 40], 5)
        gnrt_shape = ball(gnrt_shape, [40, 40, 24], 4)
    else:
        raise
    return gnrt_shape


''' Generate an integrated dataset, based on the signal of specific shape '''


def Gnrt_data(signal, sample_size, sigma_of_X=1, sigma_of_noise=1, mask=np.array([1]), seed=666):
    rng = np.random.RandomState(seed)
    C = signal
    N = sample_size
    shape = list(np.array(C.shape))
    shape.insert(0, N)
    X = rng.normal(0, sigma_of_X, size=shape)
    X = [mask * x for x in X]

    Y = np.array([np.sum(np.vdot(Xi, C)) for Xi in X]) + rng.normal(0, sigma_of_noise, N)
    Y = Y.reshape(-1, 1)

    return X, Y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from operators import Kron_k
    shape = (128, 128)
    N, r = 500, 0.8
    # C = Gnrt_circle(shape, (20, 44), 10)

    Bs = []
    Bs.append(np.array([[1, 1], [1, 1]]))
    Bs.append(np.array([[1, 1], [1, 1]]))
    Bs.append(np.array([[1, 1], [1, 1]]))
    Bs.append(np.array([[1, 1], [1, 1]]))
    Bs.append(np.array([[0, 1], [0, 0]]))
    Bs.append(np.array([[0, 0], [1, 0]]))
    Bs.append(np.array([[0, 1], [0, 0]]))

    C = Kron_k(Bs)
    X, Y = Gnrt_data(C, N)

    # padding_width = 5
    # mask = np.ones(list(np.array(shape) - 2 * padding_width))
    # mask = padding(mask, padding_width)

    # mask = Gnrt_circle(shape, (31, 31), 32)
    # X, Y = Gnrt_data(C, N, mask=mask)

