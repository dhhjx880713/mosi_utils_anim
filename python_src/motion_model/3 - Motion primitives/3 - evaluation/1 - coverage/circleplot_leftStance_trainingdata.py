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


def plot_circlepart(astart, aend, lstart, lend, color='g'):
    phi = np.linspace(astart, aend)
    rho1 = np.array([lstart] * len(phi))
    rho2 = np.array([lend] * len(phi))

    xl, yl = pol2cart(rho1, phi)
    xu, yu = pol2cart(rho2, phi)

    plt.plot(xl, yl, color=color)
    plt.plot(xu, yu, color=color)
    plt.plot([xl[0], xu[0]], [yl[0], yu[0]], color=color)
    plt.plot([xl[-1], xu[-1]], [yl[-1], yu[-1]], color=color)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def load_leftStance():
    """ Load data for walk_leftStance """

    inputpath = "output" + os.sep + "original" + os.sep + 'walk_leftStance'

    cartesian = {}
    cartesian['LeftFood'] = np.load(inputpath + os.sep + 'LeftFoot.npy').item()
    return cartesian



def plot_circle(data, length=(50, 101, 10), alpha=(45, 180, 5)):
    X = -1 * data[:, 0]
    Y = -1 * data[:, 2]
    rho, phi = cart2pol(X, Y)
    alpha_rad = np.deg2rad(alpha)
    for a in np.arange(*alpha_rad):
        for l in np.arange(*length):
            cond1 = np.logical_and(phi>=a, phi<a+alpha_rad[2])
            cond2 = np.logical_and(rho>=l, rho<l+length[2])
            cond = np.logical_and(cond1, cond2)

            if np.any(cond):
                print sum(cond)
                c = '#%02x%02x%02x' % (max(255 - sum(cond)*6, 0), 255, 0)
                plot_circlepart(a, a+alpha_rad[2], l, l+length[2], c)
            else:
                plot_circlepart(a, a+alpha_rad[2], l, l+length[2], 'r')


def main():
    #plot_region()
    cartesian = load_leftStance()
    plot_circle(cartesian['LeftFood'][-1])

if __name__ == '__main__':
    main()