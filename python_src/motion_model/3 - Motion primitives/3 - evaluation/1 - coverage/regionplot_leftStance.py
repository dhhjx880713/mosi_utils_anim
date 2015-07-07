# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 14:27:18 2015

@author: mamauer
"""
import numpy as np
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import os
import itertools


def region_bound(left_angle, right_angle, length):
    """ Calculate the region bounding for ploting"""
    xl = np.cos(left_angle) * length
    xr = np.cos(right_angle) * length
    X = np.linspace(xl, xr)
    l2 = length**2
    Y = [np.sqrt(l2 - x**2) for x in X]
    return X, Y


def plot_region(lengths=None, angles=None):
    """ Plot the desired region for walk:leftStance

    Parameters
    ----------
    lengths : array-like
        The array with min and max length of a step
    angles : array-like
        The array with the min and max angle of a step
    """

    if lengths is None:
        lengths = [50, 100]
    if angles is None:
        angles = [45, 180]
    _angles = [np.deg2rad(a) for a in angles]

    # Lower bounding
    X1, Y1 = region_bound(_angles[0], _angles[1], lengths[0])
    X2, Y2 = region_bound(_angles[0], _angles[1], lengths[1])

    plt.plot(X1, Y1, color='b')
    plt.plot(X2, Y2, color='b')
    plt.plot([X1[0], X2[0]], [Y1[0], Y2[0]], color='b')
    plt.plot([X1[-1], X2[-1]], [Y1[-1], Y2[-1]], color='b')


def in_region(x, y, lengths, angles):
    """ Check if the given (x, y) is in a valid steprange or not.

    Parameters
    ----------
    x : int
        The x coordinate
    y : int
        The y coordinate
    lengths : array-like
        The array with min and max length of a step
    angles : array-like
        The array with the min and max angle of a step

    Return
    ------
    boolean
        Wether the (x, y) coordinate is a valid postition for the LeftFoot in
        the last frame or not.
    """
    veclen = np.sqrt(x**2 + y**2)
    angle = np.rad2deg(np.arccos(x/veclen))

    return (angle < max(angles) and angle > min(angles) and
            veclen < max(lengths) and veclen > min(lengths))


def load_leftStance():
    """ Load data for walk_leftStance """

    inputpath = "output" + os.sep + 'walk_leftStance'

    cartesian = {}
    cartesian['LeftFood'] = np.load(inputpath + os.sep + 'LeftFoot.npy').item()
    return cartesian


def plot_hist(data):
    """ Plot a 3D - Histogram of the given data

    Parameters
    ----------
    data : numpy.ndarray
        X, Y, Z coordinates of the data points.
        Note: Only the X and Z coordinates are used.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 2], bins=30)
    elements = (len(xedges) - 1) * (len(yedges) - 1)
    xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(elements)
    dx = 2 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    #plt.imshow(H, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()

def plot_area(data, res=2.5, lengths=None, angles=None):
    """ Plot the coverage of the area

    Parameters
    ----------
    data : numpy.ndarray
        X, Y, Z coordinates of the data points.
        Note: Only the X and Z coordinates are used.
    res : number
        The resoultion of the squares in the area (width & hight of rectangles)
    lengths : array-like
        The array with min and max length of a step
    angles : array-like
        The array with the min and max angle of a step
    """

    if lengths is None:
        lengths = [50, 100]
    if angles is None:
        angles = [45, 180]

    # Mirror axis to the upper part...
    X = -1 * data[:, 0]
    Y = -1 * data[:, 2]

    # MAX STEPLENGTH: 1m, add a bit to it..
    xmax = int(max(X)) + 5
    while (xmax % res) > 1e-5:
        xmax += 1   # Add one until division with resolution is even
    xmin = -xmax

    ymax = int(max(Y)) + 5
    while (ymax % res) > 1e-5:
        ymax += 1   # Add one until division with resolution is even
    ymin = 0

    xmax = ymax = max(xmax, ymax)
    xmin = -xmax

    bin_x_count = (xmax - xmin) / res
    bin_y_count = (ymax - ymin) / res

    bin_x = np.linspace(xmin, xmax, bin_x_count+1)
    bin_y = np.linspace(ymin, ymax, bin_y_count+1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = np.array([['none'] * len(bin_y)] * len(bin_x), dtype='|S7')
    count = np.zeros((bin_x.size, bin_y.size))

    for x, y in itertools.izip(X, Y):
        cx = np.argmax(bin_x > x)
        cy = np.argmax(bin_y > y)
        count[cx, cy] += 1

    maxc = np.max(count)
    fact = int(255 / maxc)
    print fact

    for i in xrange(len(bin_x)):
        for j in xrange(len(bin_y)):
            in_reg = in_region(bin_x[i], bin_y[j], lengths, angles)
            if count[i, j] > 0:
                if in_reg:
                    color[i, j] = '#%02x%02x%02x' % (0, 255-count[i, j]*fact, 0)
                else:
                    color[i, j] = 'grey'
            else:
                if in_reg:
                    color[i, j] = 'red'

    for i in xrange(len(bin_x)):
        for j in xrange(len(bin_y)):
            x = bin_x[i]
            y = bin_y[j]
            ax.add_patch(
                patches.Rectangle((x-res/2, y-res/2), res, res,
                                  facecolor=color[i, j], edgecolor='none')
            )

    #plt.plot(X, Y, '*', color='r')
    #plot_region(lengths, angles)
    plt.plot(0, 0)
    plt.show()


def main():
    #plot_region()
    angles = [45, 180]
    lengths = [50, 100]
    res = 2.5
    cartesian = load_leftStance()
    plot_area(cartesian['LeftFood'][-1], res, lengths, angles)

if __name__ == '__main__':
    main()