# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np
import matplotlib.pyplot as plt

### Works on whole DSMs ###

def sparsity(matrix, min_value=0):
    """
    https://redmine.epfl.ch/projects/python_cookbook/wiki/Matrix_sparsity_patterns
    """
    mat = matrix.matrix
    plt.spy(mat, precision=min_value, marker=',')
    plt.show()


def heatmap(matrix):
    row, col, data = matrix.row_col_data
    histogram, xedges, yedges = np.histogram2d(col, row, bins=50, normed=True, weights=data)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(histogram, extent=extent)
    plt.show()


def hexbin(matrix):
    row, col, data = matrix.row_col_data
    plt.hexbin(col, row, vmin=0)
    plt.show()


def pcolormesh(matrix):
    row, col, data = matrix.row_col_data
    histogram, xedges, yedges = np.histogram2d(col, row, bins=50, normed=True, weights=data)
    plt.pcolormesh(histogram)
    plt.show()


### Works on distributional vectors ###

def plot_vector(vector):
    if not vector.is_vector():
        raise ValueError("A vector can only have a row of length one.")
    plt.plot(vector.to_ndarray().A[0])
    plt.show()

__dsm__ = ["show_sparsity", "heatmap", "hexbin", "pcolormesh"]
__vector__ = ['plot_vector']
