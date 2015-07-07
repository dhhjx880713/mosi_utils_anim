# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:29:19 2015

@author: mamauer
"""
import numpy as np
import os
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def plot3d(data, ax=None, color='b'):
    """ Plot the data in 3D Space

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 2], data[:, 1], color=color)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')


def plot2d(data, color='b'):
    """ Plot the data in 2D Space

    """
    plt.scatter(data[:, 0], data[:, 2], color=color)
    plt.xlabel('X')
    plt.ylabel('Z')


def main():
#    primitive = 'walk'
#    action = 'leftStance'
#    features = ['Hips', 'LeftFoot']
#    relevant_frames = [0, -1]

    primitive = 'pickRight'
    action = 'first'
    features = ['Hips', 'RightHand']
    relevant_frames = [-1]

    colors = {'Hips':{0:'grey', -1:'black'},
              'LeftFoot':{0:'b', -1:'r'},
              'RightHand':{0:'b', -1:'r'}}

    inputpath = "output" + os.sep + primitive + '_' + action

    cartesian = {}
    s = np.load(inputpath + os.sep + 's.npy')

    typ = '3D'
    if typ == '2D':
        fig = plt.figure()

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for f in features:
        cartesian[f] = np.load(inputpath + os.sep + f + '.npy').item()


        for rf in relevant_frames:
            color = 'r' if rf == -1 else 'b'

            if typ == '2D':
                plot2d(cartesian[f][rf], color=colors[f][rf])
            else:
                plot3d(cartesian[f][rf], ax, color=colors[f][rf])

    plt.show()


if __name__ == '__main__':
    main()
