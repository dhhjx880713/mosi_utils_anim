# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 10:40:41 2015

@author: mamauer, han
"""
import json
import os
from bitvector import calc_bitvector_walking
import numpy as np


def detect_walking_keyframes(motion, features, offset, verbose=False):
    """ FOR INTERNAL USE ONLY! Use detect_keyframes with motion_type='walking'
    Detect all Keyframes for the given Feature(s) in the given Walking Motion

    Parameters
    ----------
    * motion: list of cartesian_frame
    \tThe motion data
    * feature : list of str
    \tThe Food features
    * offset: int
    \tThe offset to be added on each keyframe

    Returns
    -------
    A dictionary containing a list for each feature.
    Each list contains all [Startframe, Endframe] Pairs for this feature.
    """
    bitvectors = calc_bitvector_walking(motion, features, verbose=verbose)

    keyframes = {feature: [] for feature in features}

    last = 0
    highes = 0
    highest_feature = None

    for i in xrange(1, len(bitvectors)):
        for feature in features:
            if bitvectors[i][feature] == 0 and bitvectors[i-1][feature] == 1:
                keyframes[feature].append([last+offset, i+offset])
                last = i
                if highes < last:
                    highes = last
                    highest_feature = feature

    f = None
    for feature in features:
        if feature != highest_feature:
            f = feature
            break
    keyframes[f].append([highes+offset, len(bitvectors)-1+offset])

    for feature in features:
        keyframes[feature].sort()

    return keyframes


def manual_keyframes(filename):
    # load keyframes
    manual_filename = 'keyframes.json'
    if manual_keyframes.keyframes is None:
        with open(manual_filename, 'r') as fp:
            manual_keyframes.keyframes = json.load(fp)

    try:
        return manual_keyframes.keyframes[filename]
    except KeyError:
        return None
manual_keyframes.keyframes = None


def detect_keyframes(motion, features, offset=0,
                     motion_type='walking', verbose=False):
    """ Detect all Keyframes for the given Feature(s) in the given Motion

    Parameters
    ----------
    * motion: list of cartesian_frame
    \tThe motion data
    feature : list of str
    \tThe features corresponding to the searched Keyframes. if the motion_type
    is 'walking', than this should be eather the left foot,
    the right foot or both.
    *offset: int
    \tAn offset which is added to each keyframe

    Returns
    -------
    A list containing all Keyframes.
    """
    if motion_type.startswith('walk') or motion_type.startswith('walking') \
            or motion_type.startswith('carry'):
        return detect_walking_keyframes(motion, features, offset, verbose)

    raise ValueError('The motiontype "%s" is not supported yet' % motion_type)


def filter_tpose(motion, features, verbose=False):
    """ Filter the T-Pose of the given motion based on the given features.

    Parameters
    ----------
    * motion: list of cartesian_frame
    \tThe motion data

    * features: list of string
    \tThe features

    * verbose: boolean
    \tIf True, additional debug outputs and plots are created.

    Returnes
    --------
    * start_frame: int
    \tThe start of the actual motion after the T-Pose

    * end_frame: int
    \tThe end of the actual motion bevor the T-Pose

    """
    bitvectors = calc_bitvector_walking(motion, features, verbose=verbose)
    n = len(bitvectors)
    converted = {feature: [] for feature in features}
    for i in xrange(n):
        for feature in features:
            converted[feature].append(bitvectors[i][feature])

    # Helperfunction to calculate the number of values in a row.
    def gen_same_in_row(data):
        last = data[0]
        count = 1
        for i in xrange(1, len(data)):
            if data[i] == last:
                count += 1
            else:
                yield count, i
                count = 1
                last = data[i]

    # calculate possible frames. Possible start frames are frames, where the
    # first falling or rising flank after more than <step_threshold> frames
    # occurs. Possible end frames are frames where the last falling or rising
    # flank for more than <step_threshold> frames occure.
    step_threshold = 150
    possible_start_frames = []
    possible_end_frames = []
    for feature in features:
        vector = np.array(converted[feature])
        last_i = -1
        for count, i in gen_same_in_row(vector):
            if count > step_threshold:
                possible_start_frames.append(i)

                if (last_i != -1):
                    possible_end_frames.append(last_i)
            last_i = i
        possible_end_frames.append(last_i)
    possible_start_frames.sort()
    possible_end_frames.sort()

    # calculate the difference between the possible frames
    diff_start = np.diff(possible_start_frames)
    diff_end = np.diff(possible_end_frames)

    # the acctual start frame is often the first startframe, same goes for
    # the end frame.
    start_frame = possible_start_frames[0]
    end_frame = possible_end_frames[-1]

    # however, in some files, a sidestep to pick up the object for carring is
    # performed. This can be filtered here:
    for i in xrange(len(diff_start)):
        if diff_start[i] > step_threshold and \
                possible_start_frames[i+1] < n * 0.7:
            start_frame = possible_start_frames[i+1]
            break
    for i in xrange(len(diff_end)-1, 0, -1):
        if diff_end[i] > step_threshold and \
                possible_end_frames[i] > start_frame:
            end_frame = possible_end_frames[i]
            break

    return start_frame, end_frame


def text_to_json(textfile, jsonfile=None):
    """ Converts a textfile with manual keyframe labels to a json
    string or file """
    with open(textfile, 'rb') as f:
        data = {}
        current_motion = None
        for l in f:
            l = l.rstrip()
            if '.bvh' in l: # file name
                current_motion = l
                data[current_motion] = []
            elif current_motion is not None and l != '' and l != '\n':  # data
                try:
                    line_split = l.split(' ')
    #                print line_split
                    tmp = {'elementary_action': line_split[0],
                           'motion_primitive': line_split[1],
                           'frames': [int(line_split[2]), int(line_split[3])]}
                    data[current_motion].append(tmp)
                except ValueError:
                    raise ValueError("Couldn't process line: %s" % l)                    
        f.close()
    if jsonfile is not None:
        with open(jsonfile, 'w+') as fp:
            json.dump(data, fp)
            fp.close()
    return data


def main():
    fileprefix = 'place_newfiles'
    textfile = os.sep.join(('..', 'manual_keyframes', fileprefix + '.txt'))
    jsonfile = os.sep.join(('..', 'manual_keyframes', fileprefix + '.json'))
    text_to_json(textfile, jsonfile=jsonfile)


if __name__ == '__main__':
    main()
