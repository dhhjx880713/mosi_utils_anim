# -*- coding: utf-8 -*-
"""
Created on Wed May 07 13:44:05 2014

@author: MAMAUER
"""
from copy import deepcopy
import os
from lib.bvh_io import load_animations_as_cartesian_frames
from lib.bvh_io import load_animations_as_euler_frames
from lib.bvh_io import save_animations_from_euler_frames
from lib.keyframes import detect_keyframes, filter_tpose, manual_keyframes

ROOT_DIR = os.sep.join([".."] * 3)


#############################################################
# Input / Output dirs
#############################################################

def get_input_data_folder(take, subfolder=''):
    """Returns folder path as string without trailing os.sep

    Parameters
    ----------

     * elementary_action: String
    \tName of the takes to be processed, e.g. 'walk' or 'Take_walk'

    """

    data_dir_name = "data"
    mocap_dir_name = "1 - MoCap"
    retargeting_dir_name = "2 - Rocketbox retargeting"

    if take.startswith('take_'):
        take[0] = 'T'

    if not take.startswith('Take_'):
        take = 'Take_' + take

    if subfolder != '':
        input_dir = os.sep.join([ROOT_DIR,
                                 data_dir_name,
                                 mocap_dir_name,
                                 retargeting_dir_name,
                                 take,
                                 subfolder])

    else:
        input_dir = os.sep.join([ROOT_DIR,
                                 data_dir_name,
                                 mocap_dir_name,
                                 retargeting_dir_name,
                                 take])

    return input_dir


def get_output_folder(elementary_action):
    """Returns folder path as string without trailing os.sep

    Parameters
    ----------

     * elementary_action: String
    \tElementary action of the motion primitive
    """
    data_dir_name = "data"
    mocap_dir_name = "1 - MoCap"
    cutting_dir_name = "3 - Cutting"

    if not elementary_action.startswith('elementary_action_'):
        elementary_action = 'elementary_action_' + elementary_action

    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              mocap_dir_name,
                              cutting_dir_name,
                              elementary_action])
    return output_dir


def clean_path(path):
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])

        if ':' not in path:
            cwd = [directory for directory in os.getcwd().split(os.sep)]
        else:
            cwd = []

        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd)-relative_levels] +
            [directory for directory in path.split(os.sep)
                if directory != ''][relative_levels:])
    return path

#############################################################
# splitt functions
#############################################################


def __splitt_motion_automatic_keyframes(motion, keyframes):
    """ Splitt a Motion by the given Keyframes

    Parameters
    ----------
    * motion: list of cartesian_frame
    \tThe motion data
    * keyframes : dict of list of int
    \tA dictionary containing a list for each feature.
    Each list contains all Keyframes for this feature.

    Returns
    -------
    A dictionary containing a list for each feature
    Each list contains all new Motions as list of cartesian_frame
    """

    tmpmins = []
    tmpmax = []
    for feature in keyframes:
        tmpmins.append(min(keyframes[feature])[0])
        tmpmax.append(max(keyframes[feature])[1])
    firstframe = min(tmpmins)
    lastframe = max(tmpmax)

    motion_list = {}

    for feature in keyframes:
        featstring = ''
        if feature == 'Bip01_L_Toe0':
            featstring = 'rightStance'
        elif feature == 'Bip01_R_Toe0':
            featstring = 'leftStance'
        motion_list[featstring] = []
        # save first step:
        if firstframe in keyframes[feature][0]:
            feature_begin = 'begin' + \
                featstring[0].capitalize() + featstring[1:]
            keyframe = keyframes[feature][0]
            subframes = motion[keyframe[0]:keyframe[1]]

            motion_list[feature_begin] = [(subframes, keyframe), ]
            keyframes[feature] = keyframes[feature][1:]

        # last step:
        if lastframe in keyframes[feature][-1]:
            feature_end = 'end' + featstring[0].capitalize() + featstring[1:]
            keyframe = keyframes[feature][-1]
            subframes = motion[keyframe[0]:keyframe[1]]

            motion_list[feature_end] = [(subframes, keyframe), ]
            keyframes[feature] = keyframes[feature][:-1]

        for keyframe in keyframes[feature]:
            subframes = motion[keyframe[0]:keyframe[1]]
            motion_list[featstring].append((subframes, keyframe))

    return motion_list


def __splitt_motion_manual_keyframes(motion, keyframes):
    """ splitt a motion based on a set of manual keyframes

    Parameters
    ----------
    motion: list of lists
    \tThe frames of the motion
    keyframes: list of dicts
    \tThe keyframes. Each keyframe is a dict with a value for "primitive" and
    one for "frames"

    Return
    ------
    motion_list: dict
    \tA dictionarie containing the submotions for each primitive
    """
    motion_list = {}
    for keyframe in keyframes:
        primitive = keyframe['primitive']
        sub_keyframe = keyframe["frames"]
        submotion = motion[sub_keyframe[0]:sub_keyframe[1]]
        try:
            motion_list[primitive].append((submotion, sub_keyframe))
        except KeyError:
            motion_list[primitive] = [(submotion, sub_keyframe), ]
    return motion_list


def align_toes(motion, toe_indexes):
    """ @brief Sets the Toes to a fixed value.

    This is a dirty hack to exclude shaking of the toe values

    @param motion a np.array of shape (refnumberframes, refnumberchannels)
    @param toe_indexes The channel number of the Toes (without EndNodes)

    @return The adjusted motion as np.array of shape
    (numberframes, numberchannels)
    """
    internalmo = deepcopy(motion)
    for index in toe_indexes:
        tmp = index * 3 + 3
        internalmo[:, tmp] = 90
        internalmo[:, tmp+2] = 75

    return internalmo


#############################################################
# Main module function
#############################################################


def segmentation(input_folder, output_folder, skeltonfile, features,
                 motion_type='walk', hastpose=True, verbose=False,
                 mask='*.bvh'):
    """ Segment all bvh files in the input_folder.

    Parameters
    ----------
    * input_folder: string
    \tThe folder with the bvh files.

    * output_folder: string
    \tThe folder where to save the segments. Will create subfolders according
    to the motion primitives

    * skeletonfile: string
    \tA file with the skeletonstructure saved

    * features: list of strings
    \tA list with the node names of the features to be used to segment all
    files

    * motion_type: string
    \tA string to determine which processing function to be used

    * hastpose: boolean
    \tIf True, the file starts and ends with a T-Pose

    * verbose: boolean
    \tIf True, additional debug outputs and plots are created.

    * mask: string
    \tMask pattern for file search, passed to gen_file_path(...)
    """
    # check folders
    if input_folder[-1] != os.sep:
        input_folder = input_folder + os.sep
    if output_folder[-1] != os.sep:
        output_folder = output_folder + os.sep

    if verbose:
        print "Looking for files in '%s'" % input_folder

    data = load_animations_as_cartesian_frames(input_folder, mask=mask)
    if verbose:
        print "Loaded %s files." % len(data)

    # detect t-pose
    start_end_frames = {}
    if hastpose:
        for f, frames in data.iteritems():
            result = filter_tpose(frames, features, verbose=verbose)
            if verbose:
                print "Motionfile '%s' starts with %d and ends with %d" % \
                    (f, result[0], result[1])
            start_end_frames[f] = result
    else:
        for f, frames in data.iteritems():
            start_end_frames[f] = (10, -21)     # Dirty fix...
            end = -1

    eulerdata = load_animations_as_euler_frames(input_folder, mask=mask)
    for f in data:
        euler_frames = eulerdata[f]
        cat_frames = data[f]
        start, end = start_end_frames[f]
        start -= 10
        end += 20
        print "Processing file %s from Frame %d to %d" % (f, start, end)
        frames_without_tpose = cat_frames[start:end]

        # check for manual keyframes:
        keyframes = manual_keyframes(f)
        if keyframes is not None:
            print "Manual Keyframes for: %s" % f
            splitted_motions = __splitt_motion_manual_keyframes(
                euler_frames, keyframes)

        else:
            keyframes = detect_keyframes(frames_without_tpose, features,
                                         motion_type=motion_type, offset=start,
                                         verbose=verbose)
            splitted_motions = __splitt_motion_automatic_keyframes(
                euler_frames, keyframes)
        # save it
        filename = f.split(os.sep)[-1].split('.')[0]
        for feature in splitted_motions:
            for splitted_motion, keyframe in splitted_motions[feature]:
                filepath = os.sep.join((output_folder, feature,
                                        filename + '_' + feature + '_' +
                                        str(keyframe[0]) + '_' +
                                        str(keyframe[1])))
                save_animations_from_euler_frames(splitted_motion, filepath)


#############################################################
# example function to demonstrate module usage
#############################################################

def main():
    action = 'place'
    filter_tpose = False

    input_folder = get_input_data_folder(action + '_newfiles')

    output_folder = get_output_folder(action)

    skeleton_file = os.sep.join((ROOT_DIR, '1 - MoCap', 'skeleton.bvh'))

    features = ['Bip01_L_Toe0', 'Bip01_R_Toe0']
    mask = '*.bvh'
    segmentation(input_folder, output_folder, skeleton_file,
                 features, motion_type=action, verbose=True, mask=mask,
                 hastpose=filter_tpose)
    return


if __name__ == '__main__':
    main()
