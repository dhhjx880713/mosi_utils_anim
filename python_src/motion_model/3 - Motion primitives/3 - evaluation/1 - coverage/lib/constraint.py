# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:32:58 2015

@author: mamauer,FARUPP, erhe01

Provides all funktionality to check how good a constraint is met
 by a motion primitive sample s
"""

import numpy as np
from motion_editing import convert_quaternion_to_euler,\
                            transform_euler_frames,\
                             get_cartesian_coordinates,\
                            get_cartesian_coordinates2,\
                            get_orientation_vec, \
                            pose_orientation,\
                            get_joint_weights, \
                            convert_euler_frame_to_cartesian_frame, \
                            align_point_clouds_2D, \
                            transform_point_cloud, \
                            calculate_point_cloud_distance, \
                            transform_quaternion_frames
from custom_transformations import vector_distance,\
                                   get_aligning_transformation2,\
                                   transform_point,\
                                   create_transformation
from transformations import rotation_matrix


def constraint_distance(constraint, target_position=None,
                        target_orientation=None):
    """returns the euclidean distance of the target_position and the position
    of the constraint and the frobenius distance of the rotations

    Parameters
    ----------
    * constraint : dict
    \tThe constraint, contains position and orientation
    * target_position : list
    \tThe position of the target
    * target_orientation : list
    \tThe orientation of the target

    Returns
    -------
    * position_distance : float
    \tThe euclidean distance
    * rotation_distance : float
    \tThe frobenius difference of the rotation matrices
    """
    position_distance = 0
    rotation_distance = 0

    if "position" in constraint.keys() and target_position is not None:
        position_distance = vector_distance(constraint["position"],
                                            target_position)
    if "orientation" in constraint.keys() and target_orientation is not None:
        # get orientation distance based on unique representation:
        # rotation matrix, orientation order is X, Y, Z

        rotation_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert len(constraint["orientation"]) == 3,\
            ('the length of rotation vector should be 3')
        rotmat_constraint = np.eye(4)
        rotmat_target = np.eye(4)
        for i in xrange(len(constraint["orientation"])):
            if constraint["orientation"][i] is not None:
                tmp_constraint = rotation_matrix(np.deg2rad(constraint
                                                 ["orientation"][i]),
                                                 rotation_axes[i])
                rotmat_constraint = np.dot(tmp_constraint, rotmat_constraint)
                tmp_target = rotation_matrix(np.deg2rad(target_orientation[i]),
                                             rotation_axes[i])
                rotmat_target = np.dot(tmp_target, rotmat_target)
        rotation_distance = vector_distance(np.ravel(rotmat_constraint),
                                            np.ravel(rotmat_target))

    return position_distance, rotation_distance


def check_constraint_one_frame(frame, constraint, bvh_reader,
                               node_name_map=None,
                               precision={"pos":1,"rot":1},
                               verbose=False):
    """ checks whether the constraint is fullfiled for a given frame

    Parameters
    ----------
    * frame : numpy.ndarry
    \tThe frame to check already transformed to global coordinate system
    * constraint : tuple
    \tThe constraint as (joint, [pos_x, pos_y, pos_z],
    \t[rot_x, rot_y, rot_z]) where unconstrained variables
    \tare set to None
    * bvh_reader: BVHReader
    \tUsed for hierarchy information
    * node_name_map : dict
    \t Maps node name to index in frame vector (ignoring "Bip" joints)
    * pos_precision : float
    \tThe precision of the position in the sample to be rated as
    \t"constraint fulfilled"
    * rot_precision : float
    \tThe precision of the rotation
    Returns
    -------
    * success : bool
    \tWhether the constraint is fulfilled or not
    * distance : float
    \t distance to constraint
    """
    if "joint" in constraint.keys():
        node_name = constraint["joint"]
        if node_name_map is not None:
            target_position = get_cartesian_coordinates2(bvh_reader,
                                                         node_name, frame,
                                                         node_name_map)
    #        parameter_index =node_name_map[node_name]*3 +3
        else:
            target_position = get_cartesian_coordinates(bvh_reader, node_name,
                                                        frame)                                                
    #        parameter_index =bvh_reader.node_names[node_name]*3 +3

    #    target_orientation = None#frame[parameter_index:parameter_index+3]
    #    #print "target", target_orientation,constraint["orientation"]
        pos_distance, rot_distance = constraint_distance(constraint,
                                                         target_position=
                                                         target_position,
                                                         target_orientation=
                                                         None)
        if pos_distance <= precision["pos"]  and rot_distance <= precision["rot"]:
            success = True
        else:
            success = False

        #return sum of distances for the selection of the best sample
        return success, pos_distance + rot_distance
    else:
        print "missing joint name in constraint definition"
        return False,np.inf



def find_aligned_frames(mm, s, prev_frames, start_pose,
                        bvh_reader, node_name_map):
    """Find aligning transformation for the sample"""

    transformation = None
    frames = convert_quaternion_to_euler(mm.back_project(s,use_time_parameters=True).get_motion_vector().tolist())
    if prev_frames != None:
        transformation = get_aligning_transformation2(frames,prev_frames,bvh_reader,node_name_map)
    elif start_pose != None:
        transformation = start_pose
    if transformation != None and "orientation" in transformation.keys() and "position" in transformation.keys():
        frames = transform_euler_frames(frames,transformation["orientation"],transformation["position"])
    return frames

def find_aligned_quaternion_frames(mm, s, prev_frames, start_pose, bvh_read,
                                   node_name_map):
    """Align quaternion frames from low dimensional vector s based on
    previous frames
       
    Parameters
    ----------
    * mm: motion primitive
    * s: numpy array
    \tLow dimensional vector for motion sample from motion primitive
    * prev_frames: list
    \tA list of quaternion frames
    * start_pose: dict
    \tA dictionary contains staring position and orientation
    
    Returns:
    quaternion_frames
    """
    # get quaternion frames of input motion s
    quat_frames = mm.back_project(s, use_time_parameters=True).get_motion_vector()
    # find alignment transformation: rotation and translation
    # covert the first frame of quat_frames and the last frame of pre_frames 
    # into euler frames, then compute the transformation based on point cloud
    # alignment
    if prev_frames != None:
        last_frame_euler = np.ravel(convert_quaternion_to_euler([prev_frames[-1]]))
        first_frame_euler = np.ravel(convert_quaternion_to_euler([quat_frames[0]]))
        point_cloud_a = convert_euler_frame_to_cartesian_frame(bvh_read,
                                                               last_frame_euler,
                                                               node_name_map)
        point_cloud_b = convert_euler_frame_to_cartesian_frame(bvh_read,
                                                               first_frame_euler,
                                                               node_name_map)
        weights = get_joint_weights(bvh_read, node_name_map)
        theta, offset_x, offset_z = align_point_clouds_2D(point_cloud_a,
                                                          point_cloud_b,
                                                          weights)
#        rotation_angle = [0, theta, 0]
#        translation = [offset_x, 0, offset_z]
        transformation = {"orientation": [0, np.rad2deg(theta), 0],
                          "position": np.array([offset_x, 0, offset_z])}                                                      
    elif start_pose != None:
        transformation = start_pose

    quat_frames = transform_quaternion_frames(quat_frames,
                                              transformation["orientation"],
                                              transformation["position"])   
    return quat_frames                                                    


def check_constraint(mm,
                     s,
                     constraint,
                     bvh_reader,
                     range_start=None,
                     range_stop=None,
                     node_name_map=None,
                     prev_frames=None,
                     start_pose=None,
                     precision={"pos":1,"rot":1,"smooth":1},
                     firstFrame=None,
                     lastFrame=None,
                     verbose=False):
    """ Main function of the modul. Check whether a sample fullfiles the
    constraint with the given precision or not. 
    Note only one type of constraint is allowed at once.

    Parameters
    ----------
    * mm: MotionPrimitive
        Instance of a MotionPrimitive class for backprojection
    * s: numpy.ndarray
        The s vector
    * constraint : tuple
        The constraint as (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
        where unconstrained variables are set to None
    * range_start : int
    \tThe index of the frame where the range where the constraint should be
    \tchecked starts. This range is respected iff firstFrame=lastFrame=None
    * range_stop : int
    \tThe index where the range ends
    * bvh_reader: BVHReader
    \tUsed for hierarchy information
    * node_name_map : dict
    \t Optional: Maps node name to index in frame vector ignoring "Bip" joints
    * transformation : dict
      Contains position as cartesian coordinates and orientation
      as euler angles in degrees
     * prev_frames : list
     \tA list of quternion frames
    Returns
    -------
    * ok : list
        list of frames, which have been checked and
        where the constrained is fullfilled
        together with the distance to the constraint calculated using l2 norm ignoring None
    * failed: list
        list of frames, which have been checked and
        where the constrained is not fullfilled
        together with the distance to the constraint calculated using l2 norm ignoring None
    """

    start_stop_dict = {
        (None, None): (None, None),
        (True, None): (None, 1),
        (False, None): (1, None),
        (None, True): (-1, None),
        (None, False): (None, -1),
        (True, False): (None, 1),
        (False, True): (-1, None),
        (False, False): (1, -1)
    }

#    frames = find_aligned_frames(mm, s, prev_frames, start_pose,
#                                 bvh_reader, node_name_map)
    quat_frames = find_aligned_quaternion_frames(mm, s, prev_frames,
                                                 start_pose, bvh_reader,
                                                 node_name_map)
#    n_frames = len(frames)
    n_frames = len(quat_frames)                                             
    #handle the different types of constraints

    if "dir_vector" in constraint.keys() and lastFrame: # orientation constraint on last frame
#        print "direction constraint is called"
        # get motion orientation
#        motion_dir = get_orientation_vec(frames)
        motion_dir = pose_orientation(quat_frames[-1])
#        last_transformation = create_transformation(frames[-1][3:6],[0, 0, 0])
#        motion_dir = transform_point(last_transformation,[0,0,1])
#        motion_dir = np.array([motion_dir[0], motion_dir[2]])
#        motion_dir = motion_dir/np.linalg.norm(motion_dir)
        target_dir = np.array([constraint["dir_vector"][0], constraint["dir_vector"][2]])
        target_dir = target_dir/np.linalg.norm(target_dir)
        
        r_distance = abs(target_dir[0] - motion_dir[0]) + \
                     abs(target_dir[1] - motion_dir[1])
        if r_distance < precision["rot"]:
            success = True
        else:
            success = False
        r_distance = r_distance * 10
        # to check the last frame pass rotation and trajectory constraint or not
        # put higher weights for orientation constraint
        if success:
            return [(n_frames-1, r_distance)], []
        else:
            return [], [(n_frames - 1,r_distance)]

    elif "frame_constraint" in  constraint.keys():
#        print "frame constraint is called"
        #print "check frame constraint"
        # get point cloud of first frame
        first_frame_euler = np.ravel(convert_quaternion_to_euler([quat_frames[0]]))
        point_cloud = convert_euler_frame_to_cartesian_frame(bvh_reader,
                                                             first_frame_euler,
                                                             node_name_map)
#        point_cloud = convert_euler_frame_to_cartesian_frame(bvh_reader,
#                                                             frames[0],
#                                                             node_name_map)
        constraint_point_cloud = []
        for joint in node_name_map.keys():
            constraint_point_cloud.append(constraint["frame_constraint"][joint])
        weights = get_joint_weights(bvh_reader,node_name_map)

        theta, offset_x, offset_z = align_point_clouds_2D(constraint_point_cloud,
                                                          point_cloud,
                                                          weights)
        t_point_cloud = transform_point_cloud(point_cloud, theta, offset_x, offset_z)

        error = calculate_point_cloud_distance(constraint_point_cloud,t_point_cloud)
        if error < precision["smooth"]:
            success = True
        else:
            success = False
        if success:
            return [(0, error)], []
        else:
            return [], [(0,error)]

    elif "position" or "orientation" in constraint.keys():
#        print "position constraint is called"
        #check specific frames
        last_frame_euler = np.ravel(convert_quaternion_to_euler([quat_frames[-1]]))
        if firstFrame and lastFrame:  # special case because of and
#            first,f_distance = check_constraint_one_frame(frames[0], constraint, bvh_reader,node_name_map,
#                                            precision,verbose)
#
#            last,l_distance = check_constraint_one_frame(frames[-1], constraint,  bvh_reader,node_name_map,
#                                           precision,verbose)
            first,f_distance = check_constraint_one_frame(first_frame_euler, 
                                                          constraint, 
                                                          bvh_reader,
                                                          node_name_map,
                                                          precision,
                                                          verbose)

            last,l_distance = check_constraint_one_frame(last_frame_euler,
                                                         constraint,  
                                                         bvh_reader,
                                                         node_name_map,
                                                         precision,
                                                         verbose)        
            if first+last:
                return [(0,f_distance), (n_frames - 1,l_distance )],[]
            else:
                return [],[(0,f_distance), (n_frames - 1,l_distance )]

        start, stop = start_stop_dict[(firstFrame, lastFrame)]



        heuristic_range = 0.10 * n_frames

#        filtered_frames = frames[-heuristic_range:]
        filtered_frames = quat_frames[-heuristic_range:]
        filtered_frames_euler = convert_quaternion_to_euler(filtered_frames)
        filtered_frame_nos = range(n_frames)

        ok = []
        failed = []
        for frame_no, frame in zip(filtered_frame_nos, filtered_frames_euler):
            success ,distance = check_constraint_one_frame(frame, constraint,  bvh_reader,node_name_map,
                                          precision,verbose)
            if success:
                ok.append((frame_no,distance))
            else:
                failed.append( (frame_no,distance))

        return ok,failed
    else:
        print "no constraint type specified"
        return [],[(0,10000)]



#def test():
#    constraints = {"position": [20, 30, 40], "orientation": [None, None, None]}
#    target_position = [25, 30, 45]
#    target_orientation = [None, None, None]
#    dist = constraint_distance(constraints, target_position, target_orientation)
#    print dist
#
#if __name__ == '__main__':
#    test()

