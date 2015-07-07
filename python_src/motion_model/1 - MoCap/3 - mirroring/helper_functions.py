# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 20:45:31 2015

@author: herrmann
"""
from math import radians,degrees
from cgkit.cgtypes import quat
from lib.quaternion_frame import *
from lib.bvh import *


def euler_to_quaternion(euler_angles,rotation_order = ['X','Y','Z']):

    current_axis = 0
    quaternion= 0
    while current_axis < len(rotation_order):
        rad = radians(euler_angles[current_axis])
        if rotation_order[current_axis] == 'X':
         
            temp_quaternion = quat()
            temp_quaternion.fromAngleAxis(rad, vec3(1.0,0.0,0.0))
            if quaternion != 0:
                quaternion = quaternion*temp_quaternion
            else:
                quaternion = temp_quaternion
        elif rotation_order[current_axis] == 'Y':
            temp_quaternion = quat()
            temp_quaternion.fromAngleAxis(rad, vec3(0.0,1.0,0.0))
            if quaternion != 0:
                quaternion = quaternion*temp_quaternion 
            else:
                quaternion = temp_quaternion
        elif rotation_order[current_axis] == 'Z':
            temp_quaternion = quat()
            temp_quaternion.fromAngleAxis(rad, vec3(0.0,0.0,1.0))
            if quaternion != 0:
                quaternion = quaternion*temp_quaternion 
            else:
                quaternion = temp_quaternion 
        current_axis +=1
    return quaternion


def quaternion_to_euler(q,rotation_order = \
                                    ['Xrotation','Yrotation','Zrotation']):
    q = quat(q)
    return _matrix_to_euler(q.toMat3(),rotation_order)
    
def _matrix_to_euler(matrix,rotation_channel_order):
    """ Wrapper around the matrix to euler angles conversion implemented in 
        cgkit. The channel order gives the rotation order around
        the X,Y and Z axis. For each rotation order a different method is 
        provided by cgkit.
        TODO: Use faster code by Ken Shoemake in Graphic Gems 4, p.222
        http://thehuwaldtfamily.org/jtrl/math/Shoemake,%20Euler%20Angle%20Conversion,%20Graphic%27s%20Gems%20IV.pdf
        https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf
    """
    if rotation_channel_order[0] =='Xrotation':
          if rotation_channel_order[1] =='Yrotation':
              euler = matrix.toEulerXYZ()
          elif rotation_channel_order[1] =='Zrotation':
              euler = matrix.toEulerXZY()
    elif rotation_channel_order[0] =='Yrotation':
        if rotation_channel_order[1] =='Xrotation':
             euler = matrix.toEulerYXZ()
        elif rotation_channel_order[1] =='Zrotation':
             euler = matrix.toEulerYZX()   
    elif rotation_channel_order[0] =='Zrotation': 
        if rotation_channel_order[1] =='Xrotation':
            euler = matrix.toEulerZXY()    
        elif rotation_channel_order[1] =='Yrotation': 
            euler = matrix.toEulerZYX()  
    return [degrees(e) for e in euler] 
    

   
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
            frame_values = [root_translation,]+quat_frame.values()
            frames.append(frame_values)
            
    else:
        for i in xrange(number_of_frames):
            quat_frame = QuaternionFrame(bvh_reader, i,False)
            root_translation = bvh_reader.keyframes[i][0:3]
            frame_values = [root_translation,]+quat_frame.values()
            frames.append(frame_values)
            
    return frames

def get_frame_vectors_from_quat_animations(data):
    motion_vector =[]
    for frame in data:
         frame_vector = []
         for t in frame:
              frame_vector+=list(t)
         motion_vector.append(frame_vector)
    return motion_vector