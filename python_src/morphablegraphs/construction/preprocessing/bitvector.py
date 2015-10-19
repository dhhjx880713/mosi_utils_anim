# -*- coding: utf-8 -*-
"""
Created on Wed May 07 13:44:05 2014

@author: MAMAUER
"""
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os
import glob
from ...animation_data.skeleton import Skeleton
from ...animation_data.bvh import BVHReader
from ...animation_data.motion_editing import \
    get_cartesian_coordinates_from_euler_full_skeleton as get_cartesian_coords


def dict_and(a, b):
    """ FOR INTERNAL USE ONLY!
    Compare one Dictionaries with another and returns a
    Dictionaries containing the logical piecewise AND (only for Keys in both
    Dictionaries)

    Parameters
    ----------
    a : dict
        The first dict
    b : dict
        The second dict

    Returns
    -------
    A Dictionarie, containing the logical, piecewise AND of a and b.
    True is 1 and False is 0
    """
    keys = list(set.intersection(set(a.keys()), set(b.keys())))

    return {key: (1 if a[key] and b[key] else 0) for key in keys}


def dict_or(a, b):
    """ FOR INTERNAL USE ONLY!
    Compare one Dictionaries with another and returns a
    Dictionaries containing the logical piecewise OR (only for Keys in both
    Dictionaries)

    Parameters
    ----------
    a : dict
        The first dict
    b : dict
        The second dict

    Returns
    -------
    A Dictionarie, containing the logical, piecewise OR of a and b.
    True is 1 and False is 0
    """
    keys = list(set.intersection(set(a.keys()), set(b.keys())))

    return {key: (1 if a[key] or b[key] else 0) for key in keys}


def smooth_bitvectors(bitvectors, threshold=4):
    """ Smooth the bitvector by flating out every peak which is smaller than
    the given threshold

    Parameters
    ----------
    bitvectors: list of dicts
        The bitvector
    threshod: int
        The minimum number for a peak
    """
    features = bitvectors[0].keys()
    vectors = deepcopy(bitvectors)

    counter = 0
    at_start = True

    for feature in features:
        for i in xrange(1, len(bitvectors)):
            if vectors[i][feature] != vectors[i-1][feature]:
                if at_start:
                    at_start = False
                    counter = 0
                elif counter < threshold:
                    for j in xrange(1, counter+2):
                        vectors[i-j] = vectors[i]
                else:
                    counter = 0
            else:
                counter += 1
    return vectors


def calc_bitvector_walking(motion, features, skeleton=None, verbose=False,
                           threshold=0.8):
    """ Detect a bit vector for each frame in the motion

    Parameters
    ----------
    motion : numpy.ndarray
        The frames of the walking motion
    feature : list of str
        The Food features
    skeleton: animation_data.skeleton.Skeleton
        The skeleton to be used. If None, the default interact skeleton is
        used.
    verbose: bool
        Wether to print / plot debug information or not
    threshold: int
        T.B.D.

    Returns
    -------
    A list containing a bit vector for each frame. Each bit vector has one
    element for each feature, indicating wether this feature is on the ground
    or not.
    """
    if skeleton is None:
        reader = BVHReader('skeleton.bvh')
        skeleton = Skeleton(reader)

    if isinstance(features, basestring):
        features = [features]

    jointpositions = {}

    # Get cartesian position frame wise. Change this for performance?
    for feature in features:
        jointpositions[feature] = []
        for frame in motion:
            jointpositions[feature].append(get_cartesian_coords(reader, skeleton,
                                                                feature, frame))
        jointpositions[feature] = np.array(jointpositions[feature])

    velocity_bitvectors_xz = [{feature: 0 for feature in features}
                              for i in xrange(motion.shape[0])]

    xz_threshold = threshold

    relativ_velo_x = {}
    relativ_velo_y = {}
    relativ_velo_z = {}
    relativ_velo_xz = {}

    for feature in features:
        relativ_velo_x[feature] = []
        relativ_velo_y[feature] = []
        relativ_velo_z[feature] = []
        relativ_velo_xz[feature] = []
        n = len(jointpositions[feature])
        for i in xrange(n - 1):
            diff = jointpositions[feature][i] - jointpositions[feature][i+1]
            relativ_velocity_x = np.abs(diff[0])
            relativ_velocity_z = np.abs(diff[2])
            relativ_velocity_xz = relativ_velocity_x**2 + relativ_velocity_z**2
            relativ_velo_xz[feature].append(relativ_velocity_xz)
            if relativ_velocity_xz < xz_threshold:
                velocity_bitvectors_xz[i][feature] = 1

        velocity_bitvectors_xz[n-2][feature] = \
            velocity_bitvectors_xz[n-3][feature]
        velocity_bitvectors_xz[n-1][feature] = \
            velocity_bitvectors_xz[n-2][feature]

    height_bitvectors = [{feature: 0 for feature in features}
                         for i in xrange(motion.shape[0])]

    height_threshold = threshold

    jointpositions_y = {}
    for feature in features:
        jointpositions_y[feature] = [pos[1] for pos in jointpositions[feature]]
        for i in xrange(len(jointpositions_y[feature])):
            if jointpositions_y[feature][i] < height_threshold:
                height_bitvectors[i][feature] = 1


    bitvectors_smoothed = smooth_bitvectors(velocity_bitvectors_xz,
                                            threshold=3)
    bitvectors_smoothed = smooth_bitvectors(bitvectors_smoothed, threshold=2)

    first_feature = -1
    first_feature_pos = np.inf
    for feature in features:
        for i in xrange(len(bitvectors_smoothed)):
            if bitvectors_smoothed[i][feature] == 0:
                if first_feature_pos > i:
                    first_feature_pos = i
                    first_feature = feature
                break
    if first_feature_pos < 50:
        for i in xrange(first_feature_pos):
            bitvectors_smoothed[i][first_feature] = 0

    if verbose:
        # Plots:
        plt.figure()
        for feature in features:
            tmp = [vector[feature] for vector in bitvectors_smoothed]
            plt.plot(tmp, label=feature)
        plt.legend()
        plt.ylim([0, 2])
        plt.title('walk')
        plt.xlabel('frameindex')
        plt.ylabel('bitvalue')

#        plt.figure()
#        for feature in features:
#            tmp = [vector[feature] for vector in height_bitvectors]
#            plt.plot(tmp, label=feature)
#        plt.legend()
#        plt.ylim([0, 2])
#        plt.title('walk')
#        plt.xlabel('frameindex')
#        plt.ylabel('bitvalue (using height)')

        plt.figure()
        line_x = range(len(relativ_velo_xz[features[0]]))
        line_y = [xz_threshold] * len(line_x)
        plt.plot(line_x, line_y)
        for feature in features:
            plt.plot(relativ_velo_xz[feature], label=feature)
        plt.legend()
        plt.xlabel('frameindex')
        plt.ylabel('relativ velocity in xz')

#        plt.figure()
#        line_x = range(len(relativ_velo_xz[features[0]]))
#        line_y = [xz_threshold] * len(line_x)
#        plt.plot(line_x, line_y)
#        for feature in features:
#            plt.plot(jointpositions_y[feature], label=feature)
#        plt.legend()
#        plt.xlabel('frameindex')
#        plt.ylabel('relativ velocity in xz')

        plt.ioff()
        plt.show()

    return bitvectors_smoothed


def detect_walking_keyframes(motion, features, skeleton, verbose=False):
    """ FOR INTERNAL USE ONLY! Use detect_keyframes with motion_type='walking'
    Detect all Keyframes for the given Feature(s) in the given Walking Motion

    Parameters
    ----------
    motion : numpy.ndarray
        The frames of the walking motion
    feature : list of str
        The Food features
    skeleton: animation_data.skeleton.Skeleton
        The skeleton to be used.
    verbose: bool
        Wether to print / plot debug information or not

    Returns
    -------
    A dictionary containing a list for each feature.
    Each list contains all [Startframe, Endframe] Pairs for this feature.
    """
    bitvectors = calc_bitvector_walking(motion, features, skeleton, verbose)

    keyframes = {feature: [] for feature in features}

    last = 0
    highes = 0
    highest_feature = None

    for i in xrange(1, len(bitvectors)):
        for feature in features:
            if bitvectors[i][feature] == 0 and bitvectors[i-1][feature] == 1:
                keyframes[feature].append([last, i])
                last = i
                if highes < last:
                    highes = last
                    highest_feature = feature

    f = None
    for feature in features:
        if feature != highest_feature:
            f = feature
            break
    keyframes[f].append([highes, len(bitvectors)-1])

    for feature in features:
        keyframes[feature].sort()

    print keyframes
    return keyframes


def detect_keyframes(motion, features, skeleton=None,
                     motion_type='walking', verbose=False):
    """ Detect all Keyframes for the given Feature(s) in the given Motion

    Parameters
    ----------
    motion : numpy.ndarray
        The frames of the walking motion
    feature : list of str
        The features corresponding to the searched Keyframes.
        if the motion_type is 'walking', than this should be eather the left
        foot, the right foot or both.
    skeleton: animation_data.skeleton.Skeleton
        The skeleton to be used. If None, the default Interact skeleton
        will be loaded
    motion_type: string
        The motion type of the given frames. Currently, only 'walking'
        is supported
    verbose: bool
        Wether to print / plot debug information or not


    Returns
    -------
    A list containing all Keyframes.
    """
    if motion_type == 'walk' or motion_type == 'walking':
        return detect_walking_keyframes(motion, features, skeleton, verbose)

    raise ValueError('The motiontype "%s" is not supported yet' % motion_type)


def splitt_motion(motion, keyframes,
                  skeleton_file='Skeleton.bvh'):
    """ Splitt a Motion by the given Keyframes

    Parameters
    ----------
    motion : AnimationData.SkeletonAnimationData
        The motion as AnimationData.SkeletonAnimationData
    keyframes : dict of list of int
        A dictionary containing a list for each feature.
        Each list contains all Keyframes for this feature.

    Returns
    -------
    A dictionary containing a list for each feature
    Each list contains all new Motions as AnimationData.SkeletonAnimationData
    """
    motion_list = {feature: [] for feature in keyframes}
    frames = motion.getFramesData(weighted=0)

    # Calc number of steps for status update
    n = 0.0
    counter = 0.0
    for feature in keyframes:
        n += len(keyframes[feature])

    tmpmins = []
    tmpmax = []
    for feature in keyframes:
        tmpmins.append(min(keyframes[feature])[0])
        tmpmax.append(max(keyframes[feature])[1])
    firstframe = min(tmpmins)
    lastframe = max(tmpmax)

    for feature in keyframes:
        # save first step:
        if firstframe in keyframes[feature][0]:
            keyframe = keyframes[feature][0]
            subframes = np.ravel(frames[keyframe[0]:keyframe[1]])
            new_motion = AnimationData.SkeletonAnimationData()
            new_motion.buildFromBVHFile(skeleton_file)
            new_motion.fromVectorToMotionData(subframes, weighted=0)
            new_motion.name = 'begin_' + str(keyframe[0]) + '_' + str(keyframe[1]) \
                + '_' + feature + '_' + motion.name
            motion_list[feature].append(new_motion)
            keyframes[feature] = keyframes[feature][1:]

        # last step:
        if lastframe in keyframes[feature][-1]:
            keyframe = keyframes[feature][-1]
            subframes = np.ravel(frames[keyframe[0]:keyframe[1]])
            new_motion = AnimationData.SkeletonAnimationData()
            new_motion.buildFromBVHFile(skeleton_file)
            new_motion.fromVectorToMotionData(subframes, weighted=0)
            new_motion.name = 'end_' + str(keyframe[0]) + '_' + str(keyframe[1]) \
                + '_' + feature + '_' + motion.name
            motion_list[feature].append(new_motion)
            keyframes[feature] = keyframes[feature][:-1]

        for keyframe in keyframes[feature]:
            print counter / n * 100, '%'
            subframes = np.ravel(frames[keyframe[0]:keyframe[1]])
            new_motion = AnimationData.SkeletonAnimationData()
            new_motion.buildFromBVHFile(skeleton_file)
            new_motion.fromVectorToMotionData(subframes, weighted=0)
            new_motion.name = str(keyframe[0]) + '_' + str(keyframe[1]) \
                + '_' + feature + '_' + motion.name
            motion_list[feature].append(new_motion)

            counter += 1.0

    return motion_list


def filter_tpose(motion, features):
    """ TODO """
    bitvectors = calc_bitvector_walking(motion, features, threshold=0.1)

    converted = {feature: [] for feature in features}
    for i in xrange(len(bitvectors)):
        for feature in features:
            converted[feature].append(bitvectors[i][feature])

    first_falling_flank = np.inf
    last_rising_flank = 0

    for feature in features:
        vector = np.array(converted[feature])
        zeros_a = np.where(vector == 0)[0]
        if zeros_a[0] < first_falling_flank:
            first_falling_flank = zeros_a[0]
        if zeros_a[-1] > last_rising_flank:
            last_rising_flank = zeros_a[-1]

    frames = motion.getFramesData(weighted=0, usingQuaternion=False)
    # Add a few Frames as buffer
    subframes = np.ravel(frames[first_falling_flank:last_rising_flank + 11])
    new_motion = deepcopy(motion)
    new_motion.fromVectorToMotionData(subframes, weighted=0,
                                      usingQuaternion=False)
    return new_motion


def segmentation(source, segmentfolders, exclude, skeltonfile, features,
                 motion_type='walking', hastpose=True, tmppath=None,
                 verbose=False):
    """ TODO """
    if source[-1] != os.sep:
        source = source + os.sep

    if not tmppath:
        tmppath = source
    if tmppath[-1] != os.sep:
        tmppath = tmppath + os.sep

    if hastpose:
        for f in glob.glob(source + r'*.bvh'):
            new_motion = AnimationData.SkeletonAnimationData()
            new_motion.buildFromBVHFile(f)
            tmp = filter_tpose(new_motion, features)
            tmp.saveToFile(tmppath + tmp.name, usingQuaternion=False)

    for f in glob.glob(tmppath + '*.bvh'):
        new_motion = AnimationData.SkeletonAnimationData()
        new_motion.buildFromBVHFile(f)

        keyframes = detect_keyframes(new_motion, features,
                                     motion_type=motion_type,
                                     verbose=verbose)
        print keyframes
        splitted_motions = splitt_motion(new_motion, keyframes,
                                         skeleton_file=skeltonfile)

        if exclude[-1] != os.sep:
            exclude = exclude + os.sep
        if not os.path.exists(exclude):
            os.makedirs(exclude)

        for feature, path in zip(features, segmentfolders):
            print feature
            if path[-1] != os.sep:
                path = path + os.sep
            if not os.path.exists(path):
                os.makedirs(path)

            for splitted_motion in splitted_motions[feature]:
                if splitted_motion.name.split('_')[0] == 'begin' or \
                        splitted_motion.name.split('_')[0] == 'end':
                    splitted_motion.saveToFile(exclude + splitted_motion.name,
                                               usingQuaternion=False)
                else:
                    splitted_motion.saveToFile(path + splitted_motion.name,
                                               usingQuaternion=False)
        os.remove(f)


