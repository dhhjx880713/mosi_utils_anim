# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:50:23 2015

@author: @author: Erik Herrmann, Martin Manns
"""
from math import degrees
from copy import deepcopy
import numpy as np
from cgkit.cgtypes import quat
import fk3
from math import radians,degrees
from itertools import izip

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



def convert_quaternion_to_euler(frames):
    """
    frames must be a list
    """

    #print len(frames[0])
    temp = deepcopy(frames)
    new_frames =[]
    for frame in temp:
        new_frame = frame[:3]
        i = 3
        while i < len(frame):
            q = quat(frame[i:i+4])
            new_frame += quaternion_to_euler(q)
            i+=4
        new_frames.append(new_frame)
    return np.array(new_frames)

def euler_substraction(theta1, theta2):
    '''
    @brief: compute the angular distance from theta1 to theta2, positive value is anti-clockwise, negative is clockwise
    @param theta1, theta2: angles in degree
    '''
    theta1 = theta1%360
    theta2 = theta2%360
    if theta1 > 180:
        theta1 = theta1 - 360
    elif theta1 < -180:
        theta1 = theta1 + 360

    if theta2 > 180:
        theta2 = theta2 - 360
    elif theta2 < - 180:
        theta2 = theta2 + 360

    theta = theta1 - theta2
    if theta > 180:
        theta = theta - 360
    elif theta < -180:
        theta = theta + 360
    return theta

def euler_addition(theta1, theta2):
    '''
    @param theta1: angle to be rotated
    @param theta2: rotation angle
    '''
    theta1 = theta1%360
    theta2 = theta2%360
    if theta1 > 180:
        theta1 = theta1 - 360
    elif theta1 < -180:
        theta1 = theta1 + 360

    if theta2 > 180:
        theta2 = theta2 - 360
    elif theta2 < - 180:
        theta2 = theta2 + 360

    theta = theta1 + theta2
    if theta > 180:
        theta = theta - 360
    elif theta < -180:
        theta = theta + 360
    return theta






fk_funcs = [
    fk3.one_joint_fk,
    fk3.two_joints_fk,
    fk3.three_joints_fk,
    fk3.four_joints_fk,
    fk3.five_joints_fk,
    fk3.six_joints_fk,
    fk3.seven_joints_fk,
    fk3.eight_joints_fk,
]


def get_cartesian_coordinates( bvh_reader, node_name, euler_frame):
    """Returns cartesian coordinates for one node at one frame

    Parameters
    ----------

    * node_name: String
    \tName of node
     * bvh_reader: BVHReader
    \tBVH data structure read from a file
    * frame_number: Integer
    \tAnimation frame number that gets extracted

    """
    #print euler_frame,node_name

    if bvh_reader.node_names[node_name]._is_root:
        root_frame_position = euler_frame[:3]
        root_node_offset = bvh_reader.node_names[node_name].offset

        return [t + o for t, o in
                izip(root_frame_position, root_node_offset)]

    else:
        # Names are generated bottom to up --> reverse
        chain_names = list(bvh_reader.gen_all_parents(node_name))
        chain_names.reverse()
        chain_names += [node_name]  # Node is not in its parent list


        eul_angles = []
        for nodename in chain_names:
            index = bvh_reader.node_names.keys().index(nodename)*3 + 3
            eul_angles.append(euler_frame[index:index+3])

        #print chain_names, bvh_reader.node_names.keys().index("RightShoulder")*3 + 3,len(euler_frame)
        rad_angles = (map(radians, eul_angle) for eul_angle in eul_angles)

        thx, thy, thz = map(list, zip(*rad_angles))

        offsets = [bvh_reader.node_names[nodename].offset
                   for nodename in chain_names]

        # Add root offset to frame offset list
        root_position = euler_frame[:3]
        offsets[0] = [r + o for r, o in izip(root_position, offsets[0])]

        ax, ay, az = map(list, izip(*offsets))

        # f_idx identifies the kinematic forward transform function
        # This does not lead to a negative index because the root is
        # handled separately

        f_idx = len(ax) - 2
        #print "f",f_idx
        if len(ax)-2 < len(fk_funcs):
            return fk_funcs[f_idx](ax, ay, az, thx, thy, thz)
        else:
            return [0,0,0]

def get_cartesian_coordinates2( bvh_reader, node_name, euler_frame,node_name_map):
    """Returns cartesian coordinates for one node at one frame. Modified to
     handle frames with omitted values for joints starting with "Bip"

    Parameters
    ----------

    * node_name: String
    \tName of node
     * bvh_reader: BVHReader
    \tBVH data structure read from a file
    * frame_number: Integer
    \tAnimation frame number that gets extracted
    * node_name_map: dict
    \tA map from node name to index in the euler frame

    """
    #print len(euler_frame),node_name

    if bvh_reader.node_names[node_name]._is_root:
        root_frame_position = euler_frame[:3]
        root_node_offset = bvh_reader.node_names[node_name].offset

        return [t + o for t, o in
                izip(root_frame_position, root_node_offset)]

    else:
        # Names are generated bottom to up --> reverse
        chain_names = list(bvh_reader.gen_all_parents(node_name))
        chain_names.reverse()
        chain_names += [node_name]  # Node is not in its parent list


        eul_angles = []
        for nodename in chain_names:
            index = node_name_map[nodename]*3 +3
            eul_angles.append(euler_frame[index:index+3])

        #print chain_names, bvh_reader.node_names.keys().index("RightShoulder")*3 + 3,len(euler_frame)
        rad_angles = (map(radians, eul_angle) for eul_angle in eul_angles)

        thx, thy, thz = map(list, zip(*rad_angles))

        offsets = [bvh_reader.node_names[nodename].offset
                   for nodename in chain_names]

        # Add root offset to frame offset list
        root_position = euler_frame[:3]
        offsets[0] = [r + o for r, o in izip(root_position, offsets[0])]

        ax, ay, az = map(list, izip(*offsets))

        # f_idx identifies the kinematic forward transform function
        # This does not lead to a negative index because the root is
        # handled separately

        f_idx = len(ax) - 2
        #print "f",f_idx
        if len(ax)-2 < len(fk_funcs):
            return fk_funcs[f_idx](ax, ay, az, thx, thy, thz)
        else:
            return [0,0,0]

def convert_euler_frame_to_cartesian_frame(bvh_reader,euler_frame,node_name_map = None):
    """
    converts euler frames to cartesian frames by calling get_cartesian_coordinates for each joint
    """
    cartesian_frame = []
    for node_name in bvh_reader.node_names:
        #ignore Bip joints
        # end sites are already ignored by the BVH Reader
        if  not node_name.startswith("Bip"): #not bvh_reader.node_names[node_name].isEndSite() and
            if node_name_map:
                cartesian_frame.append(get_cartesian_coordinates2(bvh_reader,node_name,euler_frame,node_name_map))
            else:
                cartesian_frame.append(get_cartesian_coordinates(bvh_reader,node_name,euler_frame))
    return cartesian_frame

def convert_euler_frames_to_cartesian_frames(bvh_reader,euler_frames,node_name_map = None):
    """
    converts to euler frames to cartesian frames
    """

    cartesian_frames = []
    for euler_frame in euler_frames:
        cartesian_frames.append(convert_euler_frame_to_cartesian_frame(bvh_reader,euler_frame,node_name_map) )
    return np.array(cartesian_frames)