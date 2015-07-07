# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 20:41:42 2015

@author: herrmann
"""
import os
import sys
from libtest import params, pytest_generate_tests
from itertools import izip

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(1, TESTPATH)
sys.path.insert(1, TESTPATH + (os.sep + os.pardir))


import fk3
from lib.quaternion_frame import *
from lib.bvh import *
from mirror_animation import *

    
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

def convert_euler_frame_to_cartesian_frame(bvh_reader,euler_frame):
    """
    converts euler frames to cartesian frames by calling get_cartesian_coordinates for each joint
    """
    cartesian_frame = []
    for node_name in bvh_reader.node_names:
        #ignore Bip joints
        # end sites are already ignored by the BVH Reader
        if  not node_name.startswith("Bip"): #not bvh_reader.node_names[node_name].isEndSite() and
            cartesian_frame.append(get_cartesian_coordinates(bvh_reader,node_name,euler_frame))
    return cartesian_frame
  
            
def convert_quaternion_frame_to_euler(node_names,quat_frame):
    """
    converts quaternion frames to euler frames
    """
    euler_frame =[]
    euler_frame.append(quat_frame[:3])
    index = 3
    for node_name in node_names.keys():
        #if  not node_name.startswith("Bip"): #not bvh_reader.node_names[node_name].isEndSite() and
            #index = node_names.keys().index(node_name)*4 +3
            #Bip joints are needed to keep the order of the node names dict valid
            #end sites are already ignored in the bvh reader
            q = quat(quat_frame[index:index+4])
            euler = quaternion_to_euler(q)
            euler_frame.append(euler)
            index+=4

    return euler_frame
    
  
  
def convert_quaternion_frames_to_cartesian_frames(bvh_reader,quat_frames):
    """
    converts to euler and then to cartesian frames
    """
    cartesian_frames = []
    for frame in quat_frames:
        euler_frame = np.array(convert_quaternion_frame_to_euler(bvh_reader.node_names,frame)).flatten().tolist()
        cartesian_frames.append(convert_euler_frame_to_cartesian_frame(bvh_reader,euler_frame) )
    return np.array(cartesian_frames)
    
def mirror_cartesian_frames_along(cartesian_frames,axis,node_names,mirror_map):
    """
    flips the sign on the given axis. axis is a map e.g.{0:True,1:False,2:False}
    """
    m_cartesian_frames=[]
    for frame in cartesian_frames:
        mirrored_frame= []
        
        axis_indices =[i for i in axis.keys() if axis[i] ==True]
        #print "axis",axis_indices
        
        #change coordinate system
        i = 0
        while i < len(frame):
            coordinate = [frame[i][0],frame[i][1],frame[i][2]]
            for ai in axis_indices:
                coordinate[ai] =-frame[i][ai]
                print i

            mirrored_frame.append(coordinate)
            i+=1
            
        #mirror joints
        temp = mirrored_frame[:]
        for node_name in node_names:
            if node_name in mirror_map:
                index1 = node_names.index(node_name)
                index2 =node_names.index(mirror_map[node_name])
                mirrored_frame[index1] = temp[index2]
        m_cartesian_frames.append(mirrored_frame)
    return np.array(m_cartesian_frames)
        
    
def check_correctness_of_mirroring(bvh_reader,quat_frames,m_quat_frames,mirror_map):
    """
    checks whether the mirroring was done correctly using Cartesian coordinates
    """
    #convert frames from joint space into Cartesian space using forward kinematics
    
    node_names_reduced =[node_name for node_name in bvh_reader.node_names.keys() if not node_name.startswith("Bip")]
    cartesian_frames = convert_quaternion_frames_to_cartesian_frames(bvh_reader,quat_frames)
    m_cartesian_frames = convert_quaternion_frames_to_cartesian_frames(bvh_reader,m_quat_frames)

    axis = {0:True,1:False,2:False}
    #remove offset of root
    offset =  list(bvh_reader.root.offset)
    frame_index = 0
    while frame_index < len(cartesian_frames):
        joint_index = 0
        while joint_index < len(cartesian_frames[frame_index]):#Note Cartesian frames do not include endsites or joints starting with Bip
            cartesian_frames[frame_index][joint_index] = [v-o for v, o in izip(cartesian_frames[frame_index][joint_index],offset)]
            m_cartesian_frames[frame_index][joint_index] = [v-o for v, o in izip(m_cartesian_frames[frame_index][joint_index],offset)]
            joint_index+=1
        frame_index+=1
        
    #mirror Cartesian data    
    cartesian_frames2 =mirror_cartesian_frames_along(m_cartesian_frames,axis,node_names_reduced,mirror_map)
    print cartesian_frames[0][4],cartesian_frames2[0][4]#,m_cartesian_frames[0][10]

    #if verbose :
    
    #Check each channel of each joint for each frame if the mirrored data in joint parameter space is equal to the mirrored data in 
    #Cartesian space. In order to calculate the difference the poses where converted into Cartesian space before
    eps = 0.0001
    frame_index=0
    while frame_index < len(cartesian_frames):
        joint_index = 0
        while joint_index < len(cartesian_frames[frame_index]):
            d =0
            for c in xrange(len(cartesian_frames[frame_index][joint_index])):
                d += cartesian_frames[frame_index][joint_index][c] - cartesian_frames2[frame_index][joint_index][c]#np.round()
            
            if   d < eps: #np.round(cartesian_frames[frame_index][joint_index][0])== np.round(cartesian_frames2[frame_index][joint_index][0]):
                print "equal",frame_index,joint_index,node_names_reduced[joint_index]
                #print cartesian_frames[frame_index][joint_index][0], cartesian_frames2[frame_index][joint_index][0]
            else:
                print "unequal",frame_index,joint_index,node_names_reduced[joint_index]
                print d, np.round(cartesian_frames[frame_index][joint_index][1]), np.round(cartesian_frames2[frame_index][joint_index][1])
                #print cartesian_frames[frame_index][joint_index][0], cartesian_frames2[frame_index][joint_index][0]
                return False
            joint_index+=1
        frame_index+= 1

    #i = 11
    return True#np.all(np.round(cartesian_frames[i]) == np.round(cartesian_frames2[i]) )
    

def test_correctness_of_mirroring():
    """
    Loads an animation from a  BVH file, mirrors it and converts it to Cartesian Frames.
    These Cartesian frames are then compared Cartesian that were extracted from the unmirrored animation.
    """
    in_file_name = "test.bvh"
    mirror_map ={ "LeftShoulder":"RightShoulder",
                    "LeftArm":"RightArm",
                   "LeftForeArm":"RightForeArm",
                   "LeftHand": "RightHand",
                   "LeftUpLeg":"RightUpLeg",
                   "LeftLeg": "RightLeg",
                   "LeftFoot":"RightFoot"
    }    
    for k in mirror_map.keys():
        mirror_map[mirror_map[k]] = k
    
    bvh_reader = BVHReader(in_file_name)
    frames = get_quaternion_frames(in_file_name)
    frames = get_frame_vectors_from_quat_animations(frames)
    new_frames = mirror_animation(bvh_reader.node_names,frames,mirror_map)
    assert check_correctness_of_mirroring(bvh_reader,frames,new_frames,mirror_map)
    
    
if __name__ == "__main__":
    test_correctness_of_mirroring()