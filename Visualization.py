from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import nibabel as nib
from nilearn import plotting
from nilearn.plotting import plot_stat_map


def Tri_Views(Xi, cut_coords=None):
    new_image = nib.Nifti1Image(Xi, np.eye(4))
    fig = plt.figure(figsize=(9, 3), facecolor='w')
    plotting.plot_img(img=new_image, colorbar=True, figure=fig, cut_coords=cut_coords)


def my_plot3(data):
    fig = plt.figure(figsize=(40, 20))
    time = np.arange(0, 256, 1)
    channels = np.arange(0, 64, 1)

    X, Y = np.meshgrid(channels, time)
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, data.T, rstride=1, cstride=1, cmap='rainbow', linewidth=1)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    # adjust angels, first is up/down, second is left/right
    # ax.view_init(0, 0)
    ax.set_xlabel("channels")
    ax.set_ylabel("time(Sec)")
    ax.set_zlabel("Voltage(mV)")
    plt.show()


def gridplot(mat, p, grid=True):
    plt.imshow(mat)
    plt.xticks(np.arange(0, mat.shape[0]+1, np.power(2, p)))
    plt.yticks(np.arange(0, mat.shape[1]+1, np.power(2, p)))
    if grid:
        plt.grid(linestyle='-.', linewidth=0.5, which="both")


def Vis(mat, p=3, sticks=True, grid=True, cmap=None):
    if cmap is None:
        plt.imshow(mat)
    else:
        plt.imshow(mat, cmap=cmap)
    if sticks:
        plt.xticks(np.arange(0, mat.shape[0]+1, np.power(2, p)))
        plt.yticks(np.arange(0, mat.shape[1]+1, np.power(2, p)))
    else:
        plt.xticks([])
        plt.yticks([])
    if grid:
        plt.grid(linestyle='-.', linewidth=0.5, which="both")

