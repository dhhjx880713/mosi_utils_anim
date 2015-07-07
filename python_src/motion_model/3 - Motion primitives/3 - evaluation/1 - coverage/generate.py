# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:27:23 2015

@author: mamauer
"""
import os
import lib.motion_primitive
from lib.motion_editing import convert_quaternion_to_euler
from lib.motion_editing import get_cartesian_coordinates2
from lib.bvh2 import BVHReader, create_filtered_node_name_map
import numpy as np
from xml.etree import ElementTree as ET
from rpy2.rinterface import RRuntimeError
import json


def get_mm_path(primitive, action):
    """ Return the path to the mm from a motion primitive

    Parameters
    ----------
    * primitive : string
        The name of the Primitive

    * action : string
        The name of the action

    Returns
    -------
    * filepath : string
        The path to the json file
    """
    filepath_prefix = os.sep.join(('..', '..', '..', '..',
                                   'data', '3 - Motion primitives',
                                   'motion_primitives_quaternion_PCA95',
                                   'elementary_action_%s' % action))
    filepath = os.sep.join((filepath_prefix,
                            '%s_%s_quaternion_mm.json' % (action, primitive)))
    return filepath


def get_s_path(primitive, action):
    """ Return the path to the s vectors used to train the mm

    Parameters
    ----------
    * primitive : string
        The name of the Primitive

    * action : string
        The name of the action

    Returns
    -------
    * filepath : string
        The path to the json file
    """
    filepath_prefix = os.sep.join(('..', '..', '..', '..',
                                   'data', '2 - PCA',
                                   'combine_spatial_temporal'))
    filepath = os.sep.join((filepath_prefix,
                            '%s_%s_low_dimensional_motion_data.json'
                            % (action, primitive)))
    return filepath


def generate(primitive, action, features, n_samples=10000,
             relevant_frames=[0, -1]):
    """ Generate a set of s Vecotrs and calculate their [X, Y, Z] Position
    of all joints in 'features'

    Parameters
    ----------
    * primitive : string
        The name of the Primitive

    * action : string
        The name of the action

    * features : list of strings
        The names of the joints that are of interest

    * n_samples : integer
        The number of samples. If -1, the trainingdata is used.

    * relevant_frame : list of integer
        The number of the relevant frames

    Returns
    -------
    * s : numpy.ndarray
        All generated s vectors
    * cartesian : dict of numpy.ndarrays
        A dictionary containing a numpy.ndarray for each joint
    """
    mm_path = get_mm_path(primitive, action)
    mp = lib.motion_primitive.MotionPrimitive(mm_path)

    reader = BVHReader('lib' + os.sep + 'skeleton.bvh')
    nn_map = create_filtered_node_name_map(reader)

    cartesian = { }
    for f in features:
        cartesian[f] = {rf:[] for rf in relevant_frames}

    if n_samples == -1:
        s_path = get_s_path(primitive, action)
        with open(s_path) as fp:
            data = json.load(fp)
        s = []
        for k in data['motion_data']:
            s.append(data['motion_data'][k])
        s = np.array(s)
        n_samples = s.shape[0]
    else:
        s = mp.gmm.sample(n_samples)

    for i, s_i in enumerate(s):
        while True:
            try:
                ms = mp.back_project(s_i)
                quat = ms.get_motion_vector()
                #ms.save_motion_vector('bvhs' + os.sep + str(i) + '.bvh')
                break
            except RRuntimeError:
                if n_samples == -1:
                    raise ValueError("One of the original data couldn't be "
                                     "backprojected succesfully... ")
                s_i = mp.gmm.sample().ravel()
                s[i] = s_i

        for rf in relevant_frames:
            euler = convert_quaternion_to_euler([quat[rf]])
            euler = np.ravel(euler)

            for f in features:
                pos = get_cartesian_coordinates2(reader, f, euler, nn_map)
                cartesian[f][rf].append(pos)
        progress = int((float(i) / n_samples) * 100)
        print '\r[{0}] {1}%'.format('#'*(progress/2) + ' '*(50 - progress/2), progress),

    print '\r[{0}] {1}%'.format('#'*50, '100')
    for f in features:
        for rf in relevant_frames:
            cartesian[f][rf] = np.array(cartesian[f][rf])

    return s, cartesian


def create_experiments():
    experiments = "experiments.xml"
    root = ET.parse(experiments).getroot()

    for primitive_node in root:
        prim_names = primitive_node.attrib['name'].split(' ')
        params = {}
        for action_node in primitive_node:
            action = action_node.attrib['name']
            features = []
            relevant_frames = []

            for feature_node in action_node:
                features.append(feature_node.attrib['name'])
                for frame_node in feature_node:
                    if frame_node.attrib['index'] not in relevant_frames:
                        relevant_frames.append(int(frame_node.attrib['index']))
            params[action] = (features, relevant_frames)

        for primitive in prim_names:
            print "Starting primitive", primitive
            for action in params:

                outputpath = "output" + os.sep + primitive + '_' + action
                if os.path.exists(outputpath + os.sep + 's.npy'):
                    print action, "already done, skipping"
                    continue
                features = params[action][0]
                relevant_frames = params[action][1]
                print "Processing action", action
                try:
                    s, cartesian = generate(primitive, action,
                                            features, n_samples=10000,
                                            relevant_frames=relevant_frames)
                except IOError:
                    print("Couldn't find %s:%s. Skipping." % (action, primitive))
                    continue
                if not os.path.exists(outputpath):
                    os.mkdir(outputpath)
                np.save(outputpath + os.sep + 's', s)
                for f in features:
                    for rf in relevant_frames:
                        np.save(outputpath + os.sep + f, cartesian[f])


def main():
    s, cartesian = generate('leftStance', 'walk', ['Hips', 'LeftFoot'],
                            n_samples=-1, relevant_frames=[0, -1])
    outputpath = os.sep.join(("output", "original",
                             "%s_%s" % ('walk', 'leftStance')))
    np.save(outputpath + os.sep + 's', s)
    np.save(outputpath + os.sep + 'Hips', cartesian['Hips'])
    np.save(outputpath + os.sep + 'LeftFoot', cartesian['LeftFoot'])


if __name__ == '__main__':
    main()