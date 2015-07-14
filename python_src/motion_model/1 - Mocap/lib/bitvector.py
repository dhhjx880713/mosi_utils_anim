# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 08:35:41 2015

@author: mamauer
"""
from copy import deepcopy
import numpy as np
import matplotlib.pylab as plt


def dict_and(a, b):
    """
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
    """
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
    """ Smooth a bitvector

    Parameters
    ----------
    * bitvectors: list of dict
    \tA list of bitvectors. Each bitvector is a dict, having an entry for each
    feature
    * threshold: int
    \tA smoothing parameter

    Returns
    -------
    * vectors: list of dict
    \tA new list of dictionaries representing the smoothed bitvectors
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


def calc_bitvector_walking(motion, features, verbose=False, plot=False,
                           threshold=0.7):
    """
    Detect a bit vector for each frame in the motion

    Parameters
    ----------
    * motion: list of cartesian_frame
    \tThe motion data

    * features: list of string
    \tThe features

    Returns
    -------
    A list containing a bit vector for each frame. Each bit vector has one
    element for each feature, indicating wether this feature is on the ground
    or not.
    """
    if isinstance(features, basestring):
        features = [features]

    numframes = len(motion)
    velocity_bitvectors = [{feature: 0 for feature in features}
                           for i in xrange(int(numframes))]
    relativ_velos = {}

    for feature in features:
        relativ_velos[feature] = []
        n = numframes
        for i in xrange(n - 1):
            rel_velo = np.linalg.norm(
                np.subtract(motion[i][feature], motion[i+1][feature])
            )
            relativ_velos[feature].append(rel_velo)

            if rel_velo < threshold:
                velocity_bitvectors[i][feature] = 1
        velocity_bitvectors[n-1][feature] = \
            velocity_bitvectors[n-2][feature]

    bitvectors_smoothed = smooth_bitvectors(velocity_bitvectors,
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

    if verbose and plot:
        # Plots:
        plt.figure()
        for feature in features:
            tmp = [vector[feature] for vector in bitvectors_smoothed]
            plt.plot(tmp, label=feature)
        plt.legend()
        plt.xlabel('frameindex')
        plt.ylabel('bitvalue')

        plt.figure()
        for feature in features:
            tmp = [vector[feature] for vector in velocity_bitvectors]
            plt.plot(tmp, label=feature)
        plt.legend()
        plt.xlabel('frameindex')
        plt.ylabel('bitvalue xz unsmoothed')

        plt.figure()
        line_x = range(len(relativ_velos[features[0]]))
        line_y = [threshold] * len(line_x)
        plt.plot(line_x, line_y)
        for feature in features:
            plt.plot(relativ_velos[feature], label=feature)
        plt.legend()
        plt.xlabel('frameindex')
        plt.ylabel('relativ velocity in xz')

        plt.ioff()
        plt.show()

    bitvectors = bitvectors_smoothed

    return bitvectors
