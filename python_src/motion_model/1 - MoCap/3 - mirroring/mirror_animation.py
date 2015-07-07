# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:56:45 2015

@author: herrmann
"""
import copy
from lib.bvh import *
from lib.quaternion_frame import *
from helper_functions import *
from cgkit.cgtypes import quat,mat4



def flip_coordinate_system(q):
    matrix = np.array(q.toMat4().toList()).reshape(4,4)
    print "before",matrix
    #l = 4
    #r = 3
    temp = matrix[0]
    matrix[0]=matrix[1]
    matrix[1] = temp
    print "after",matrix
    q = quat().fromMat(mat4(matrix.flatten().tolist() ) )
    return q
    
def flip_coordinate_system2(q):
    euler = quaternion_to_euler(q)#3,4,5,6
        #print euler
    euler[0] = euler[0] 
    euler[1] = -euler[1]
    euler[2] = euler[2]
    print euler
    q =euler_to_quaternion(euler)
    return q
    
def flip_coordinate_system3(q):
    """
    http://www.ch.ic.ac.uk/harrison/Teaching/L4_Symmetry.pdf
    http://gamedev.stackexchange.com/questions/27003/flip-rotation-matrix
    https://www.khanacademy.org/math/linear-algebra/alternate_bases/change_of_basis/v/lin-alg-alternate-basis-tranformation-matrix-example
    """
    conversion_matrix = mat4([-1, 0, 0,  0,
                          0, 1, 0, 0,
                          0, 0,-1, 0,
                          0, 0,  0, 1])
                         
    # as far as i understand it is assumed that we are already in a different coordinate system 
    # so we first flip the coordinate system then do the original transformation 
    #then flip coordinate system again to go back to the flipped coordinate system
    # this results in the flipped transformation
    new_transformation = conversion_matrix *q.toMat4()  *conversion_matrix# ##
    q.fromMat(new_transformation)
    return q
    
    
def flip_coordinate_system4(q):
    """
    http://www.ch.ic.ac.uk/harrison/Teaching/L4_Symmetry.pdf
    http://gamedev.stackexchange.com/questions/27003/flip-rotation-matrix
     https://www.khanacademy.org/math/linear-algebra/alternate_bases/change_of_basis/v/lin-alg-alternate-basis-tranformation-matrix-example
    """
    conversion_matrix = mat4([1, 0, 0,  0,
                          0, -1, 0, 0,
                          0, 0,1, 0,
                          0, 0,  0, 1])
                         
    # as far as i understand it is assumed that we are already in a different coordinate system 
    # so we first flip the coordinate system then do the original transformation 
    #then flip coordinate system again to go back to the flipped coordinate system
    # this results in the flipped transformation
    new_transformation = conversion_matrix *q.toMat4()  *conversion_matrix# ##
    q.fromMat(new_transformation)
    return q
    
    
def flip_coordinate_system5(q1):
    """
    http://www.gamedev.sk/mirroring-animations
    http://www.gamedev.net/topic/599824-mirroring-a-quaternion-against-the-yz-plane/
    """
    #q.x = -q.x
    #q.w = -q.w 
    q2 = copy.copy(q1)
    q2.w  = q1.w
    q2.x  = q1.x
    q2.y = -q1.y
    q2.z = -q1.z
    #temporary fix
    #rotate by 180 degree around z
    euler = quaternion_to_euler(q2)#3,4,5,6
    euler[2] = euler[2]-180
    q2 =euler_to_quaternion(euler)
    return q2
    
def mirror_animation(node_names,frames,mirror_map):
    """
    http://www.gamedev.sk/mirroring-animations
    http://stackoverflow.com/questions/1263072/changing-a-matrix-from-right-handed-to-left-handed-coordinate-system
    """
    new_frames = []
    temp = frames[:]
    for frame in temp:
        new_frame = frame[:]
        #handle root seperately
        
        new_frame[:3] =[-new_frame[0],new_frame[1],new_frame[2]]
        q = quat(new_frame[3:7])
        q = flip_coordinate_system5(q)
        new_frame[3:7]  = [q.w,q.x,q.y,q.z]

        # bring rotation into different coordinate system
        i = 3
        for node_name in node_names.keys() :
        
            if  not node_names[node_name].isRoot():
                q = quat(new_frame[i:i+4])
                q = flip_coordinate_system3(q)
                new_frame[i:i+4]  = [q.w,q.x,q.y,q.z]

            i+=4
            
        #mirror joints
        if mirror_map != None:   
            temp = new_frame[:]
            for node_name in node_names.keys():
                if not node_name.startswith("Bip") and node_name in mirror_map.keys():
                    index1 = node_names.keys().index(node_name)*4+3
                    index2 = node_names.keys().index(mirror_map[node_name])*4+3
                    #print "mirror",node_name,mirror_map[node_name],index1,index2
                    new_frame[index1:index1+4] = temp[index2:index2+4]
            new_frames.append(new_frame)
            #print "new frame"
    #print frames[0][:3],new_frames[0][:3]
    return new_frames
    
    
if __name__ =="__main__":
    
    
    in_file_name = "test.bvh"
    out_file_name = "mirrored.bvh"
    
    mirror_map ={ "LeftShoulder":"RightShoulder",#
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
    #print frames
    new_frames = mirror_animation(bvh_reader.node_names,frames,mirror_map)
    BVHWriter(out_file_name,bvh_reader,new_frames,frame_time= 0.013889,\
                                        is_quaternion = True)