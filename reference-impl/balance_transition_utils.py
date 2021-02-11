import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def unit_normalize():
    x = np.random.random((3, 3))
    print("Original Array:")
    print(x)
    xmax, xmin = x.max(), x.min()
    x = (x - xmin) / (xmax - xmin)
    print("After normalization:")
    print(x)


def random_rewards_distribution(num_channels):
    random_partitions = list(random.random() for _ in range(num_channels))
    max_entropy = sum(random_partitions)
    return list(map(lambda i: i / max_entropy, random_partitions))


def plot_2d_mat(matrix, xlabel='address_eigenvalues', ylabel='address_eigenvectors', zlabel='matrix_name'):
    ax = plt.axes(projection='3d')
    x_dim, y_dim = matrix.shape
    X, Y = np.meshgrid(np.arange(0, x_dim, 1), np.arange(0, y_dim, 1))
    ax.contour3D(X, Y, matrix.transpose(), 50, cmap='binary')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(zlabel)
    plt.show()
    return


def plot_evolution_3d(mat_a, mat_b, fig_name=100, title_a="title_a", title_b="title_b"):
    fig = plt.figure(fig_name, figsize=plt.figaspect(0.5))
    fig.suptitle(title_a + " + " + title_b)  # todo add legend for dims
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    x_dim, y_dim = mat_a.shape
    X, Y = np.meshgrid(np.arange(0, x_dim, 1), np.arange(0, y_dim, 1))
    ax1.plot_surface(X, Y, mat_a.transpose(), rstride=1, cstride=1, cmap=cm.coolwarm,
                     linewidth=0, antialiased=False)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, mat_b.transpose(), rstride=1, cstride=1, cmap=cm.coolwarm,
                             linewidth=0, antialiased=False)

    def on_move(event):
        if event.inaxes == ax1:
            if ax1.button_pressed in ax1._rotate_btn:
                ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            elif ax1.button_pressed in ax1._zoom_btn:
                ax2.set_xlim3d(ax1.get_xlim3d())
                ax2.set_ylim3d(ax1.get_ylim3d())
                ax2.set_zlim3d(ax1.get_zlim3d())
        elif event.inaxes == ax2:
            if ax2.button_pressed in ax2._rotate_btn:
                ax1.view_init(elev=ax2.elev, azim=ax2.azim)
            elif ax2.button_pressed in ax2._zoom_btn:
                ax1.set_xlim3d(ax2.get_xlim3d())
                ax1.set_ylim3d(ax2.get_ylim3d())
                ax1.set_zlim3d(ax2.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    return
