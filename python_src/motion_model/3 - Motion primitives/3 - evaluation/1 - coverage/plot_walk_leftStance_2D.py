# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:29:19 2015

@author: mamauer
"""
import numpy as np
import os
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from xml.etree import ElementTree as ET


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


def plot2d(data, color='b', label=''):
    """ Plot the data in 2D Space

    """
    plt.scatter(data[:, 0], data[:, 2], color=color, label=label)
    plt.xlabel('X')
    plt.ylabel('Z')

def main():
    primitive = 'carryLeft'
    action = 'beginLeftStance'


    experiments = "experiments.xml"
    root = ET.parse(experiments).getroot()

    primitive_node = None
    action_node = None
    for node in root.iter('primitive'):
        if primitive in node.attrib['name']:
            primitive_node = node
            break

    for node in primitive_node.iter('action'):
        if action in node.attrib['name']:
            action_node = node
            break

    if primitive_node is None or action_node is None:
        print "Couldn't find %s:%s" % (primitive, action)

    features = []
    relevant_frames = {}
    colors = {}

    for feature_node in action_node:
        feature = feature_node.attrib['name']
        features.append(feature)
        colors[feature] = {}
        relevant_frames[feature] = []
        for frame_node in feature_node:
            index = int(frame_node.attrib['index'])
            colors[feature][index] = frame_node.attrib['color']
            relevant_frames[feature].append(index)


    inputpath = "output" + os.sep + primitive + '_' + action

    cartesian = {}
    s = np.load(inputpath + os.sep + 's.npy')

    typ = '2D'
    if typ == '2D':
        fig = plt.figure()

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for f in features:
        cartesian[f] = np.load(inputpath + os.sep + f + '.npy').item()

        for rf in relevant_frames[f]:
            if typ == '2D':
                plot2d(cartesian[f][rf], color=colors[f][rf],
                       label=f + ':' + str(rf))
            else:
                plot3d(cartesian[f][rf], ax, color=colors[f][rf])
    plt.legend()
    plt.title('%s - %s' %(primitive, action))
    plt.show()


if __name__ == '__main__':
    main()
