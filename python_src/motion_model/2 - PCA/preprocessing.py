# -*- coding: utf-8 -*-
'''
Created on Nov 18, 2014

@author: hadu01, Martin Manns, Erik Herrmann
'''

import os
import glob
import json
from lib.bvh import BVHReader
from lib.quaternion_frame import QuaternionFrame
import numpy as np
from cgkit.bvh import BVHReader as BVHR
ROOT_DIR = os.sep.join([".."] * 6)


#def get_quaternion_frames(bvhFile):
#    """Returns an animation read from a BVH file as a list of pose parameters, 
#	   where the rotation is represented using quaternions.
#
#    Parameters
#    ----------
#
#     * bvhFile: String
#    \tPath to a bvh file
#
#    """
#
#    bvh_reader = BVHReader(bvhFile)
#    frames = []
#    number_of_frames = len(bvh_reader.keyframes)
#    for i in xrange(number_of_frames):
#        quat_frame = QuaternionFrame(bvh_reader, i,False)
#        root_translation = [t+o for t, o in zip(bvh_reader.keyframes[i][0:3],
#												bvh_reader.root.offset)]
#        frame_values = [root_translation,]+quat_frame.values()
#        frames.append(frame_values)
#    return frames
    
def get_quaternion_frames(bvhFile,filter_values = True):
    """Returns an animation read from a BVH file as a list of pose parameters, 
	   where the rotation is represented using quaternions.
       
       To enable the calculation of a mean of different animation samples later on, a filter 
       can be activated to align the rotation direction of the quaternions
       to be close to a fixed reference quaternion ([1,1,1,1] has been experimentally selected). 
       In order to handle an error that can occur using this approach,
       the frames are additionally smoothed by comparison to the previous frame
       We assume based on experiments that the singularity does not happen in the first 
       frame of an animation.
       
    Parameters
    ----------

     * bvhFile: String
    \tPath to a bvh file
     * filter_values: Bool
    \tActivates or deactivates filtering
    
    """

    bvh_reader = BVHReader(bvhFile)

    frames = []
    number_of_frames = len(bvh_reader.keyframes)
    last_quat_frame = None
    if filter_values:
        for i in xrange(number_of_frames):
            quat_frame = QuaternionFrame(bvh_reader, i,True)
            if last_quat_frame != None:
                for joint_name in bvh_reader.node_names.keys():
                     if joint_name in quat_frame.keys():
                    
                        #http://physicsforgames.blogspot.de/2010/02/quaternions.html
                        #get dot product to see if they are far away from each other
                        dot = quat_frame[joint_name][0]*last_quat_frame[joint_name][0] + \
                            quat_frame[joint_name][1]*last_quat_frame[joint_name][1] + \
                            quat_frame[joint_name][2]*last_quat_frame[joint_name][2]  + \
                            quat_frame[joint_name][3]*last_quat_frame[joint_name][3]
                    
                        #if they are far away then flip the sign
                        if dot < 0:
                            quat_frame[joint_name] = [-v for v in quat_frame[joint_name]]              
            last_quat_frame = quat_frame
            root_translation = bvh_reader.keyframes[i][0:3] 
            # rescale root position by a saclar
#            root_translation = [i/50 for i in root_translation]
            frame_values = [root_translation,]+quat_frame.values()
            frames.append(frame_values)
            
    else:
        for i in xrange(number_of_frames):
            quat_frame = QuaternionFrame(bvh_reader, i,False)
            root_translation = bvh_reader.keyframes[i][0:3]
            # rescale root position by a saclar
#            root_translation = [i*50 for i in root_translation]
            frame_values = [root_translation,]+quat_frame.values()
            frames.append(frame_values)
            
    return frames


def scale_rootchannels(frames, turn_on=True):
    """ Scale all root channels in the given frames.
    It scales the root channel by taking its absolut maximum 
    (max_x, max_y, max_z) and devide all values by the maximum, 
    scaling all positions between -1 and 1    
    
    Parameters
    ----------
    * frames: dict
    \tThe frames for all files.
    
    Returns
    -------
    * scaled_data: dict
    \tA new dict with the same keys as in frames but scaled root positions.
    * rootmax: list
    \tX, Y and Z maximas of all datas.
    """
    if turn_on:
        scaled_data = {}
        
        max_x = 0
        max_y = 0
        max_z = 0
        
        for key, value in frames.iteritems():
            tmp = np.array(value)
            
            # Bit confusing conversion needed here, since of numpys view system
            rootchannels = tmp[:, 0].tolist()                                              
            rootchannels = np.array(rootchannels)
            
            max_x_i = np.max(np.abs(rootchannels[:, 0]))
            max_y_i = np.max(np.abs(rootchannels[:, 1]))
            max_z_i = np.max(np.abs(rootchannels[:, 2]))
           
            if max_x < max_x_i:
                max_x = max_x_i
                
            if max_y < max_y_i:
                max_y = max_y_i
                
            if max_z < max_z_i:
                max_z = max_z_i
                
        for key, value in frames.iteritems():
            tmp = np.array(value)        
            # Bit confusing conversion needed here, since of numpys view system
            rootchannels = tmp[:, 0].tolist()                                              
            rootchannels = np.array(rootchannels)        
            
            rootchannels[:, 0] /= max_x
            rootchannels[:, 1] /= max_y
            rootchannels[:, 2] /= max_z
            
            scaled_data[key] = value
            for frame in xrange(len(tmp)):
                scaled_data[key][frame][0] = tuple(rootchannels[frame].tolist())
    else:
        scaled_data = frames
        max_x = 1
        max_y = 1
        max_z = 1

    return scaled_data, [max_x, max_y, max_z]
    
    
def load_animations_using_quaternions(input_dir):
    """
    Returns a list of animations read from BVH files as a dictionary of lists
    of CartesianFrames.
    Each CartesianFrames instance contains a pose in form of joint positions in
    cartesian space.

    Parameters
    ----------

     * dir: String
    \tPath of input folder, in which the input files reside

   """

    data = {}
    if not input_dir.endswith(os.sep):
        input_dir += os.sep
    temporal_parameter_file = input_dir + 'timewarping.json'
    if len(temporal_parameter_file) > 116:
        temporal_parameter_file = clean_path(temporal_parameter_file)
    with open(temporal_parameter_file, 'rb') as infile:
        temporal_parameters = json.load(infile)
        infile.close()
    file_order = sorted(temporal_parameters.keys())
    for item in file_order:
        file_path = clean_path(input_dir + item)
        data[item] = get_quaternion_frames(file_path, filter_values = True)
#    for item in gen_file_paths(input_dir):
#        if len(item) > 116:
#            item = clean_path(item)
#        filename = os.path.split(item)[-1]
#        
#        data[filename] = get_quaternion_frames(item, filter_values = True)
#     data = np.array(data)
#    print data
    return data


def gen_file_paths(dir, mask='*.bvh'):
    """Generator of input file paths

    Parameters
    ----------

     * dir: String
    \tPath of input folder, in which the input files reside
     * mask: String, defaults to '*.bvh'
    \tMask pattern for file search

    """

    if not dir.endswith(os.sep):
        dir += os.sep

    for filepath in glob.glob(dir + mask):
        yield filepath


def get_input_data_folder(elementary_action, motion_primitive):
    """Returns folder path as string without trailing os.sep

    Parameters
    ----------

     * elementary_action: String
    \tElementary action of the motion primitive
     * motion_primitive: String
    \tMotion primitive for which the folder shall be returned

    """

    data_dir_name = "data"
    mocap_dir_name = "1 - MoCap"
    alignment_dir_name = "4 - Alignment"
    elementary_action_folder = "elementary_action_" + elementary_action

    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             mocap_dir_name,
                             alignment_dir_name,
                             elementary_action_folder,
                             motion_primitive])

    return input_dir


def get_output_folder():
    """
    Return folder path to store result without trailing os.sep
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "1 - preprocessing"
    action = 'experiments'
    test_feature = "2 - FPCA with quaternion joint angles"
    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              PCA_dir_name,
                              type_parameter,
                              step,
                              action,
                              test_feature])
    return output_dir


def clean_path(path):
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])
        cwd = [directory for directory in os.getcwd().split(os.sep)] if ':' not in path else []
        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd)-relative_levels] + [directory for directory in path.split(os.sep) if directory != ''][relative_levels:])
    return path

def main():
    elementary_action = 'carryBoth'
    motion_primitive = 'turningRightStance'
    input_dir = get_input_data_folder(elementary_action,
                                      motion_primitive)
    output_dir = get_output_folder()
#    print input_dir
#    print output_dir
    if len(input_dir) > 116:  # avoid a too long path
        input_dir = clean_path(input_dir)    
#    if len(output_dir) > 116:  # avoid a too long path
#        output_dir = clean_path(output_dir)
#    print path

    data = load_animations_using_quaternions(input_dir)


    data, rootmax = scale_rootchannels(data, turn_on=True)
    filename = output_dir + os.sep + '%s_%s_featureVector.json' % (elementary_action,
                                                                   motion_primitive)
    if len(filename) > 116:
        filename = clean_path(filename)
    with open(filename, 'wb') as outfile:
        json.dump(data, outfile)
        outfile.close()
        
    filename = output_dir + os.sep + '%s_%s_maxVector.json' % (elementary_action,
                                                               motion_primitive)
    filename = clean_path(filename)                                                           
    with open(filename, 'wb') as outfile:
        json.dump(rootmax, outfile)
        outfile.close()    

def test():
    """Extract motion feature vector from one bvh file
    """
    testfile = r'C:\Users\hadu01\MG++\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\endLeftStance\walk_002_2_endleftStance_520_555.bvh'
    framevalues = get_quaternion_frames(testfile, filter_values = True)
    print framevalues[0]

if __name__ == '__main__':
    main()
#    test()
