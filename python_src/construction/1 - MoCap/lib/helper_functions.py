# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 12:56:39 2015

@author: Han Du, Martin Manns, Erik Herrmann
"""


import os
import glob
import json
import collections
from datetime import datetime
from motion_editing import transform_euler_frames, \
                           transform_quaternion_frames
from bvh2 import BVHWriter, BVHReader
import numpy as np
from quaternion_frame import QuaternionFrame

ROOT_DIR = os.sep.join([".."] * 2)



def write_to_logfile(path,time_string,data):
    """ Appends json data to a text file.
        Creates the file if it does not exist.
        TODO use logging library instead
    """
    data_string = json.dumps(data,indent=4)
    line = time_string + ": \n" + data_string + "\n-------\n\n"
    if not os.path.isfile(path):
        file_handle = open(path,"wb")
        file_handle.write(line)
        file_handle.close()
    else:
        with open(path,"a") as file_handle:
            file_handle.write(line)

    
def write_to_logfile2(path,time_string,data_string):
    """ Appends json data to a text file.
        Creates the file if it does not exist.
        TODO use logging library instead
    """

    line = time_string + ": \n" + data_string + "\n-------\n\n"
    if not os.path.isfile(path):
        file_handle = open(path,"wb")
        file_handle.write(line)
        file_handle.close()
    else:
        with open(path,"a") as file_handle:
            file_handle.write(line)

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.
    source: http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
    '''
    z = x.copy()
    z.update(y)
    return z



def load_json_file(filename, use_ordered_dict = False):
    """ Load a dictionary from a file

    Parameters
    ----------
    * filename: string
    \tThe path to the saved json file.
    * use_ordered_dict: bool
    \tIf set to True dicts are read as OrderedDicts.
    """
    tmp = None
    with open(filename, 'rb') as infile:
        if use_ordered_dict:
            tmp = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(infile.read())
        else:
            tmp = json.load(infile)
        infile.close()
    return tmp

def write_to_json_file(filename,serilzable, indent=4):
      with open(filename, 'wb') as outfile:
          tmp = json.dumps(serilzable, indent=4)
          outfile.write(tmp)
          outfile.close()

def gen_file_paths(dir, mask='*mm.json'):
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



def get_morphable_model_directory(morphable_model_type = "motion_primitives_quaternion_PCA95"):
    """
    Return folder path to store result without trailing os.sep
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    mm_dir = os.sep.join([ROOT_DIR,
                          data_dir_name,
                          process_step_dir_name,
                          morphable_model_type])


    return mm_dir

def get_motion_primitive_directory(elementary_action):
    """Return motion primitive file path
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    morphable_model_type = "motion_primitives_quaternion_PCA95"
    mm_path = os.sep.join([ROOT_DIR,
                           data_dir_name,
                           process_step_dir_name,
                           morphable_model_type,
                           'elementary_action_' + elementary_action
                           ])
    return mm_path

def get_motion_primitive_path(elementary_action,
                              motion_primitive):
    """Return motion primitive file
    """ 
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    morphable_model_type = "motion_primitives_quaternion_PCA95"
    mm_path = os.sep.join([ROOT_DIR,
                           data_dir_name,
                           process_step_dir_name,
                           morphable_model_type,
                           'elementary_action_' + elementary_action,
                           '_'.join([elementary_action,
                                     motion_primitive,
                                     'quaternion',
                                     'mm.json'])
                           ])
    return mm_path               

def get_transition_model_directory():
    data_dir_name = "data"
    process_step_dir_name = "4 - Transition model"
    transition_dir = os.sep.join([ROOT_DIR,
                          data_dir_name,
                          process_step_dir_name,"output"])
    return transition_dir


def clean_path(path):
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])
        cwd = [directory for directory in os.getcwd().split(os.sep)] if ':' not in path else []
        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd)-relative_levels] + [directory for directory in path.split(os.sep) if directory != ''][relative_levels:])
    return path


def export_euler_frames_to_bvh( output_dir,bvh_reader,euler_frames,prefix = "",start_pose = None,time_stamp = True):
    """ Exports a list of euler frames to a bvh file after transforming the frames
    to the start pose.

    Parameters
    ---------
    * output_dir : string
        directory without trailing os.sep
    * bvh_reader : BVHRreader
        contains joint hiearchy information
    * euler_frames : np.ndarray
        Represents the motion
    * start_pose : dict
        Contains entry position and orientation each as a list with three components

    """
    if start_pose != None:
        euler_frames = transform_euler_frames(euler_frames,start_pose["orientation"],start_pose["position"])
    if time_stamp:
        filepath = output_dir + os.sep+prefix+"_" + unicode(datetime.now().strftime("%d%m%y_%H%M%S"))+".bvh"
    elif prefix!= "":
        filepath =  output_dir + os.sep+prefix+".bvh"
    else:
         filepath =  output_dir + os.sep+"output"+".bvh"
    print filepath
    BVHWriter(filepath,bvh_reader, euler_frames,bvh_reader.frame_time,is_quaternion=False)

def export_quat_frames_to_bvh(output_dir,bvh_reader,quat_frames,prefix = "",
                              start_pose = None,time_stamp = True):
    """ Exports a list of quat frames to a bvh file after transforming the 
    frames to the start pose.

    Parameters
    ---------
    * output_dir : string
        directory without trailing os.sep
    * bvh_reader : BVHRreader
        contains joint hiearchy information
    * quat_frames : np.ndarray
        Represents the motion
    * start_pose : dict
        Contains entry position and orientation each as a list with three components

    """
    if start_pose != None:
        quat_frames = transform_quaternion_frames(quat_frames,
                                                  start_pose["orientation"],
                                                  start_pose["position"])
    if time_stamp:
        filepath = output_dir + os.sep+prefix+"_" + unicode(datetime.now().strftime("%d%m%y_%H%M%S"))+".bvh"
    elif prefix!= "":
        filepath =  output_dir + os.sep+prefix+".bvh"
    else:
         filepath =  output_dir + os.sep+"output"+".bvh"
#    print filepath
#    print "#####################################"
#    print type(quat_frames)
#    print quat_frames.shape
    BVHWriter(filepath,bvh_reader, quat_frames,bvh_reader.frame_time,
              is_quaternion=True)


def trajectory_len(frames):
    """Compute the trajectory length of a list of frames
    
    Parameters
    ----------
    *frames: list
    \tFrames can be euler frames or quaternion frame
    
    Return
    ------
    *trajectory_len: float
    \tLength of trajectory

    """
    assert type(frames) is list, "frames should be a list object"
    tra_len = 0
    for i in xrange(len(frames)-1):
        tra_len += np.sqrt((frames[i+1][2] - frames[i][2])**2 + (frames[i+1][0] - frames[i][0])**2)
    return tra_len    

def point_distance(point1, point2):
    """Compute Euclidean distance of two points
       if the point contains None value, then it is regards as 0
    Parameters
    ----------
    * point1: list
    * point1: list
    """
    assert len(point1) == len(point2)
    dist = 0
    for i in xrange(len(point1)):
        if point1[i] is not None and point2[i] is not None:
            dist += (point1[i] - point2[i]) ** 2
    return np.sqrt(dist)

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
    number_of_frames = len(bvh_reader.frames)
#    print 'number of frames is:' + str(number_of_frames)
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
#            for joint_name in quat_frame.keys():
#                tmp = [abs(quat_frame[joint_name][0]), 
#                       abs(quat_frame[joint_name][1]),
#                       abs(quat_frame[joint_name][2]),
#                       abs(quat_frame[joint_name][3])]
#                max_index = max(xrange(len(tmp)), key = tmp.__getitem__)
#                if quat_frame[joint_name][max_index] < 0:
#                    quat_frame[joint_name] = [-v for v in quat_frame[joint_name]]
            last_quat_frame = quat_frame
            root_translation = bvh_reader.frames[i][0:3] 
            # rescale root position by a saclar
#            root_translation = [i/50 for i in root_translation]
            frame_values = [root_translation,]+quat_frame.values()
            frame_values = convert_to_sequence(frame_values)
            frames.append(frame_values)
            
    else:
        for i in xrange(number_of_frames):
            quat_frame = QuaternionFrame(bvh_reader, i,True)
            root_translation = bvh_reader.frames[i][0:3]
            # rescale root position by a saclar
#            root_translation = [i*50 for i in root_translation]
            frame_values = [root_translation,]+quat_frame.values()
            frame_values = convert_to_sequence(frame_values)
            frames.append(frame_values)
            
    return frames 

def convert_to_sequence(quaternion_frame):
    n_joints = len(quaternion_frame) - 1
    new_frame = np.array(quaternion_frame[0])
    for i in xrange(n_joints):
        new_frame = np.concatenate((new_frame, np.asarray(quaternion_frame[i+1])))
    return new_frame
                  

def main():
    point1 = [0, None, 0]
    point2 = [1,1,None]
    print point_distance(point1, point2)

if __name__ == "__main__":
    main()