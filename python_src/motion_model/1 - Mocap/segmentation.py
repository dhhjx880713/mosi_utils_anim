# -*- coding: utf-8 -*-
"""
Created on Wed May 07 13:44:05 2014

@author: MAMAUER, Han
"""
from copy import deepcopy
import os
from lib.bvh_io import load_animations_as_cartesian_frames
from lib.bvh_io import load_animations_as_euler_frames
from lib.bvh_io import save_animations_from_euler_frames
from lib.keyframes import detect_keyframes, filter_tpose, manual_keyframes,\
    text_to_json
from lib.bvh import BVHReader, BVHWriter
from lib.motion_editing import transform_euler_frames
import glob
import json
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

        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd) - relative_levels] +
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
        internalmo[:, tmp + 2] = 75

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


def manual_motion_segmentation(folder_path, annotation_file):
    """Cut motion in folder into motion clips based on animation file

    Parameters
    ----------
    *folder_path: string
    \tFolder path of files

    *animation_file: str
    \tJson file contains key frame annotation
    """
    if not folder_path.endswith(os.sep):
        folder_path += os.sep
    # load BVH file
#    files = glob.glob(folder_path + '*bvh')
    with open(annotation_file, 'rb') as infile:
        annotation = json.load(infile)
        infile.close()
#    print len(annotation.keys())
#    filename = annotation.keys()[0]
#    reader = BVHReader(folder_path + filename)
    joints = ['Bip01_L_Toe0', 'Bip01_R_Toe0']
#    idx1 = reader.node_names.keys().index(joints[0]) * 3 + 3
#
#    idx2 = reader.node_names.keys().index(joints[1]) * 3 + 3
#
#    start_frame = annotation[filename][0]['frames'][0]
#    end_frame = annotation[filename][0]['frames'][1]
#    new_frames = reader.keyframes[start_frame: end_frame]
#    print len(new_frames)
#    for frame in new_frames:
#        frame[idx1:idx1+3] = [90.0,-1.00000000713e-06,75.0	]
#        frame[idx2:idx2+3] = [90.0,-1.00000000713e-06,75.0	]
#    outfile = filename
#    BVHWriter(outfile,reader,new_frames,frame_time=0.013889,\
#                                        is_quaternion = False)
    for item in annotation.keys():
        try:
            filename = folder_path + item
            reader = BVHReader(filename)
            idxL = reader.node_names.keys().index(joints[0]) * 3 + 3
            idxR = reader.node_names.keys().index(joints[1]) * 3 + 3
        except IOError:
            print "cannot find file " + item
        tmp = item.split('_')
        for segment in annotation[item]:
            elementary_action = segment['elementary_action']
            primitive_type = segment['motion_primitive']
            start_frame = segment['frames'][0]
            end_frame = segment['frames'][1]
            new_frames = reader.keyframes[start_frame:end_frame]
            # fix toe pose
            for frame in new_frames:
                frame[idxL:idxL + 3] = [90.0, -1.00000000713e-06, 75.0	]
                frame[idxR:idxR + 3] = [90.0, -1.00000000713e-06, 75.0	]
            output_folder = get_output_folder(elementary_action)
            save_folder_dir = output_folder + os.sep + primitive_type
            if not os.path.exists(save_folder_dir):
                os.mkdir(save_folder_dir)
            if tmp[0] == elementary_action:
                output_filename = save_folder_dir + os.sep + '%s_%s_%d_%d.bvh' % (item[:-4],
                                                                                  primitive_type,
                                                                                  start_frame,
                                                                                  end_frame)
            else:
                output_filename = save_folder_dir + os.sep + '%s_%s_%d_%d_from_%s.bvh' % (item[:-4],
                                                                                          primitive_type,
                                                                                          start_frame,
                                                                                          end_frame,
                                                                                          tmp[0])
#            print output_filename
            BVHWriter(output_filename, reader, new_frames,
                      frame_time=0.013889, is_quaternion=False)


def cut_motion(infile, start_frame, end_frame, save_filename):
    reader = BVHReader(infile)
    joints = ['Bip01_L_Toe0', 'Bip01_R_Toe0']
    idxL = reader.node_names.keys().index(joints[0]) * 3 + 3
    idxR = reader.node_names.keys().index(joints[1]) * 3 + 3
    new_frames = reader.keyframes[start_frame: end_frame]
    for frame in new_frames:
        frame[idxL:idxL + 3] = [90.0, -1.00000000713e-06, 75.0	]
        frame[idxR:idxR + 3] = [90.0, -1.00000000713e-06, 75.0	]
    save_filename = clean_path(save_filename)
    BVHWriter(save_filename, reader, new_frames,
              frame_time=0.013889, is_quaternion=False)
#############################################################
# example function to demonstrate module usage
#############################################################


def get_save_folder_for_cutted_files(elementary_action, motion_primitive):
    """return the path of the folder to save segmented files, if the path 
       does not exist, then create the folder
    """
    elementary_action_folder = get_output_folder(elementary_action)
    if not elementary_action_folder.endswith(os.sep):
        elementary_action_folder += os.sep
    if not os.path.exists(elementary_action_folder):
        os.mkdir(elementary_action_folder)
    primitive_folder = elementary_action_folder + motion_primitive + os.sep
    if not os.path.exists(primitive_folder):
        os.mkdir(primitive_folder)
    return primitive_folder


def manual_segment_one_motion_primitive(manual_label_file,
                                        file_folder,
                                        elementary_action,
                                        primitive_type):
    """segment a specific motion primitive from 
    """
    data = text_to_json(manual_label_file)
    save_folder = get_save_folder_for_cutted_files(elementary_action,
                                                   primitive_type)
    if not file_folder.endswith(os.sep):
        file_folder += os.sep
    # search the target motion primitive
    for key, item in data.iteritems():
        for primitive_data in item:
            if primitive_data['elementary_action'] == elementary_action and \
               primitive_data['motion_primitive'] == primitive_type:
                print "find motion primitive " + elementary_action + '_' + \
                      primitive_type + ' in ' + key
                # cut the file based on frame range
                # find the file in the file folder
                filename = file_folder + key
                if not os.path.isfile(filename):
                    raise IOError('cannot find ' + key + ' in ' + file_folder)
                start_frame = primitive_data['frames'][0]
                end_frame = primitive_data['frames'][1]
                segments = key[:-4].split('_')
                segments[0] = elementary_action
                outfilename = '_'.join(
                    segments) + '_%s_%d_%d.bvh' % (primitive_type, start_frame, end_frame)
                save_filename = save_folder + outfilename
                cut_motion(filename, start_frame, end_frame, save_filename)


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


def apply_rotation_to_folder():
    file_folder = r'C:\Users\du\MG++\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_turnwalk'
    if not file_folder.endswith(os.sep):
        file_folder += os.sep
    save_folder = file_folder  + os.sep
    bvh_files = glob.glob(file_folder + '*.bvh')
    reader = BVHReader(bvh_files[0])
    joints = ['Bip01_L_Toe0', 'Bip01_R_Toe0']
    idxL = reader.node_names.keys().index(joints[0]) * 3 + 3
    idxR = reader.node_names.keys().index(joints[1]) * 3 + 3
    for item in bvh_files:
        bvhreader = BVHReader(item)
        filename = os.path.split(bvhreader.filename)[-1]
        rotation_angles = [-90, 0, 0]
        translation = [0, 0, 0]
        transformed_frames = transform_euler_frames(bvhreader.keyframes,
                                                    rotation_angles,
                                                    translation)
        for frame in transformed_frames:
            frame[idxL:idxL + 3] = [90.0, -1.00000000713e-06, 75.0	]
            frame[idxR:idxR + 3] = [90.0, -1.00000000713e-06, 75.0	]
        save_filename = save_folder + filename
        BVHWriter(save_filename, reader, transformed_frames,
                  frame_time=0.013889, is_quaternion=False)
#    bvhreader = BVHReader(bvh_files[0])
#    rotation_angle = [-90, 0, 0]
#    translation = [0, 0, 0]
#    transformed_frames = transform_euler_frames(bvhreader.keyframes,
#                                                rotation_angle,
#                                                translation)
#    save_filename = 'rotated_motion.bvh'
#    BVHWriter(save_filename, bvhreader, transformed_frames, frame_time=0.013889, is_quaternion = False)


if __name__ == '__main__':
    #    main()
    #    folder_path = r'C:\Users\hadu01\MG++\repo\data\1 - MoCap\2 - Rocketbox retargeting\Place_one_hand'
    #    annotation_file = 'manual_keyframes\place_oneHand.json'
    #    manual_motion_segmentation(folder_path, annotation_file)
    #    infile = r'C:\Users\hadu01\MG++\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_pick\pick_038_4.bvh'
    #    start_frame = 723
    #    end_frame = 872
    #    primitive_name = 'turningRight'
    #    save_folder = r'C:\Users\hadu01\MG++\repo\data\1 - MoCap\3 - Cutting\elementary_action_carryRight\turningRight\\'
    #    filename = os.path.split(infile)[-1]
    #    new_filename = filename[:-4] + '_' + primitive_name + '_' + str(start_frame) + '_' + str(end_frame) + '.bvh'
    #    cut_motion(infile, start_frame, end_frame, save_folder + new_filename)
    #    infile = r'C:\Users\hadu01\MG++\workspace\MotionGraphs++\mocap data\pick\pickRight\turningRight\orientationAlignment\carry_018_4_turningRight_640_804.bvh'
    #    start_frame = 42
    #    end_frame = 163
    #    save_path = infile
    #    cut_motion(infile, start_frame, end_frame, save_path)

    manual_label_file = r'C:\Users\du\MG++\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_sidestep\key_frame_annotation.txt'
    file_folder = r'C:\Users\du\MG++\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_sidestep\rotated_files'
    elementary_action = 'walk'
    primitive_type = 'sidestepRight'
    manual_segment_one_motion_primitive(manual_label_file,
                                        file_folder,
                                        elementary_action,
                                        primitive_type)
    data = text_to_json(r'C:\Users\du\MG++\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_sidecarry\key_frame_annotation.txt')
#    apply_rotation_to_folder()
