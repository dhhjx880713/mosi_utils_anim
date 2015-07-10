# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 15:00:01 2015

@author: erhe01
"""
import numpy as np
from math import sqrt
from motion_editing import euler_to_quaternion
from cgkit.cgtypes import quat
from scipy.optimize import minimize
from motion_editing import euler_substraction
from external.transformations import euler_matrix,  euler_from_matrix
from motion_editing import convert_quaternion_to_euler, \
                                transform_euler_frames,\
                                find_aligning_transformation

def cgkit_mat_to_numpy4x4(matrix):
    """ Converts to cgkit matrix to a numpy matrix  """
    return np.array(matrix.toList(rowmajor=True), np.float32).reshape(4,4)


def create_rotation_matrix(euler_angles, \
            rotation_order = ['Xrotation','Yrotation','Zrotation']):
    """ Creates a 4x4 homogenous transformation matrix containing a rotation
        in form of a numpy array

    Parameters
    ----------
    * euler_angles: list
    \tAngles in degrees
    * rotation_order : list
    \t must contain 'Xrotation','Yrotation','Zrotation' in some order

    Returns
    -------
    * matrix : np.ndarray
    \t 4x4 rotation matrix
    """
    q = euler_to_quaternion(euler_angles,rotation_order)
    return cgkit_mat_to_numpy4x4(quat(q).toMat4())

def create_local_rotation(euler_angles,\
            rotation_order = ['Xrotation','Yrotation','Zrotation']):
    """Calculate the local rotation matrix using the three point method
    http://www.biomechanics.psu.edu/tutorials/Kin_tutor09a.html
    """
    #euler_angles = [-e for e in euler_angles]
    rotation_matrix = create_rotation_matrix(euler_angles,rotation_order)
    x = [1,0,0]
    y = [0,1,0]
    z = [0,0,1]
    new_x = transform_point(rotation_matrix,x)
    new_y = transform_point(rotation_matrix,y)
    new_z = transform_point(rotation_matrix,z)
    local_mat = np.zeros((4,4))
    local_mat[:,0] = new_x + [0]
    local_mat[:,1] = new_y + [0]
    local_mat[:,2] = new_z + [0]
    #local_mat[:,3] = translation + [1]
    return local_mat

def generate_rotation_order_format_string(rotation_order):
    """ Converts the custom rotation order format list to a string used by
        transformations.py
    """
    xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    if 'x' in rotation_order[0].lower():
        if 'y' in rotation_order[1].lower():
            rot_order = 'xyz'
        else:
            rot_order = 'xzy'
    elif 'y' in rotation_order[0].lower():
        if 'x' in rotation_order[1].lower():
            rot_order = 'yxz'
        else:
            rot_order = 'yzx'
    elif 'z' in rotation_order[0].lower():
        if 'x' in rotation_order[1].lower():
            rot_order = 'zxy'
        else:
            rot_order = 'zyx'
    else:
        raise ValueError('unknown rotation order')
    return rot_order

def rotation_euler_frames(euler_frames, transform_matrix, \
                         rotation_order=['Xrotation','Yrotation','Zrotation']):
    """Rotate euler frames by a 4*4 transformation matrix
    """
    rot_order = generate_rotation_order_format_string(rotation_order)
    for frame in euler_frames:
        trans = np.asarray(frame[:3] + [1])
        x_rad, y_rad, z_rad = np.deg2rad(frame[3:6])
        rotmat = euler_matrix(x_rad, y_rad, z_rad, axes='s'+rot_order)
        rotation = np.dot(transform_matrix, rotmat)
        new_trans = np.dot(transform_matrix, trans.T)
        frame[0] = new_trans[0]
        frame[1] = new_trans[1]
        frame[2] = new_trans[2]
        euler_angles = euler_from_matrix(rotation, 'r'+rot_order)
        euler_angles = np.rad2deg(euler_angles)
        frame[3] = euler_angles[0]
        frame[4] = euler_angles[1]
        frame[5] = euler_angles[2]
    return euler_frames


def create_transformation(euler_angles, translation, \
            rotation_order = ['Xrotation','Yrotation','Zrotation']):
    """ Creates a 4x4 homogenous transformation matrix as numpy array

    Parameters
    ----------
    * euler_angles: list
    \tAngles in degrees
    * translation: list
    \tCartesian coordinates
    * rotation_order : list
    \t must contain 'Xrotation','Yrotation','Zrotation' in some order

    Returns
    -------
    * matrix : np.ndarray
    \tHomogenous 4x4 transformation matrix
    """
    matrix = create_rotation_matrix(euler_angles,rotation_order)
    matrix[:,3] = translation+[1]
    return matrix



def transform_point(transformation_matrix,point):
    """ Transforms a 3d point represented as a list by a numpy transformation

    Parameters
    ----------
    *transformation_matrix: np.ndarray
    \tHomogenous 4x4 transformation matrix
    *point: list
    \tCartesian coordinates

    Returns
    -------
    * point: list
    \tThe transformed point as a list
    """
    #temp = np.array(point+[1])
    #print "temp",temp,transformation_matrix
    assert type(point) == list
    return np.dot(transformation_matrix,np.array(point+[1,]))[:3].tolist()


def transform_euler_angles(transformation_matrix,angles,\
                         rotation_order=['Xrotation','Yrotation','Zrotation']):
    """ Transforms a 3d point represented as a list by a numpy transformation

    Parameters
    ----------
    *transformation_matrix: np.ndarray
    \tHomogenous 4x4 transformation matrix
    *point: np.ndarray
    \tEuler angles

    Returns
    -------
    * point: list
    \tThe transformed point as a list
    """
    rot_order = generate_rotation_order_format_string(rotation_order)
    x_rad, y_rad, z_rad = np.deg2rad(angles)
    rotmat = euler_matrix(x_rad, y_rad, z_rad, axes='s'+rot_order)
    rotation = np.dot(transformation_matrix, rotmat)
    euler_angles = euler_from_matrix(rotation, 'r'+rot_order)
    euler_angles = np.rad2deg(euler_angles)

    return angles


def to_local_coordinate_system(coordinate_system_transformation,point):
    """ Brings a 3d point represented as a list into a local coordinate system
        represented by a numpy transformation

    Parameters
    ----------
    *transformation_matrix: np.ndarray
    \tGlobal transformation of the coordinate system
    *point: list
    \tCartesian coordinates

    Returns
    -------
    * point: list
    \tThe transformed point as a list
    """

    inv_coordinate_system = np.linalg.inv(coordinate_system_transformation)

    return transform_point(inv_coordinate_system, point)

def get_delta_root_transformation(euler_frames_a,euler_frames_b, \
            rotation_order = ['Xrotation','Yrotation','Zrotation']):
    """ Extracts the transformation of the root at the end of the motion

    Parameters
    ----------
    *euler_frames_a: np.ndarray
    \tA list of pose parameters using euler frames as rotation representation
    *euler_frames_b: np.ndarray
    \tA list of pose parameters using euler frames as rotation representation
    * rotation_order : list
    \t must contain 'Xrotation','Yrotation','Zrotation' in some order

    Returns
    -------
    * transformation : np.ndarray
    \tHomogenous 4x4 transformation matrix aligning the start of b to the end of a
    """
    translation = (euler_frames_a[-1][:3] -  euler_frames_b[0][:3] ).tolist()
    rotation = [euler_substraction(a,b) for a,b in zip(euler_frames_a[-1][3:6],  euler_frames_b[0][3:6])]
    #rotation = (euler_frames_b[0][:3] - euler_frames_a[-1][3:6]).tolist()
    #rotation = (euler_frames_a[-1][3:6] -  euler_frames_b[0][3:6] ).tolist()
    transformation = create_transformation(rotation,translation,rotation_order)
    return transformation


def get_end_root_transformation(euler_frames, \
            rotation_order = ['Xrotation','Yrotation','Zrotation']):
    """ Extracts the transformation of the root at the end of the motion

    Parameters
    ----------
    *euler_frames_a: np.ndarray
    \tA list of pose parameters using euler frames as rotation representation
    *euler_frames_b: np.ndarray
    \tA list of pose parameters using euler frames as rotation representation
    * rotation_order : list
    \t must contain 'Xrotation','Yrotation','Zrotation' in some order

    Returns
    -------
    * transformation : np.ndarray
    \tHomogenous 4x4 transformation matrix at the end of euler_frames
    """
    translation = (euler_frames[-1][:3] ).tolist()
    rotation = (euler_frames[-1][3:6] ).tolist()
    transformation = create_transformation(rotation,translation,rotation_order)
    return transformation

def get_end_root_position(euler_frames, \
            rotation_order = ['Xrotation','Yrotation','Zrotation']):
    """ Extracts the transformation of the root at the end of the motion

    Parameters
    ----------
    *euler_frames: np.ndarray
    \tA list of pose parameters using euler frames as rotation representation


    Returns
    -------
    * position : list
    \tThe position of the root at the end of the motion
    """
    p = [0,0,0]
    return transform_point(get_end_root_transformation(euler_frames,rotation_order),p)


def transform_point2(rotation,translation,point):
    """ Creates a transformation matrix and then calls transform_point
    """
    t = create_transformation(rotation,translation)
    #print "matrix",t, point
    point = transform_point(t,point)
    return point


def vector_distance(a,b):
    """Returns the distance ignoring entries with None
    """
    d_sum = 0
    #print a,b
    for i in xrange(len(a)):
        if a[i] != None and b[i]!=None:
            d_sum += (a[i]-b[i])**2
    return sqrt(d_sum)

def get_best_initial_orientation(frames,goal,max_iterations = 100):
    """ Runs an optimization to find the angle that reduces distance of
    the last frame to the goal

    Returns
    -------
    *  theta : float
       angle in degrees
    """

    def error_func(x,goal,last_point):
        #print point,last_point
        last_point = transform_point2([0,x[0],0],[0,0,0],last_point)
        return vector_distance(goal,last_point)

    last_point = frames[-1][:3].tolist()
    data = (goal,last_point)
    x0 =[0,]
    result = minimize(error_func,x0,data,method= "nelder-mead")
    return result.x[0]

def get_best_initial_transformation(frames,point,max_iterations=100):
    theta = get_best_initial_orientation(frames,point,max_iterations)
    print theta
    return create_transformation([0,theta,0],[0,0,0])

def get_aligning_transformation2(sample_frames,prev_frames, bvh_reader,node_name_map):
    """ Wrapper around find_aligning_transformation
    Returns
    -------
    * transformation : dict
      Contains position as cartesian coordinates and orientation
      as euler angles in degrees
    """

    #bring motions to same y-coordinate to make 2d alignment possible
#    offset_y = prev_frames[-1][1] - sample_frames[0][1]
#    sample_frames = transform_euler_frames(sample_frames, [0,0,0],[0,offset_y,0])
    #find aligning 2d transformation
    theta, offset_x, offset_z = find_aligning_transformation(bvh_reader,prev_frames,sample_frames, node_name_map)
    transformation = {"orientation": [0,np.degrees(theta),0], "position": [offset_x,0,offset_z]}
    return transformation

def get_aligning_transformation(graph_node,bvh_reader,prev_frames, node_name_map):
    """ Wrapper around find_aligning_transformation
    Returns
    -------
    * transformation : dict
      Contains position as cartesian coordinates and orientation
      as euler angles in degrees
    """

    sample_parameters = np.ravel(graph_node.mp.gmm.sample())
    sample_frames =convert_quaternion_to_euler( graph_node.mp.back_project(sample_parameters).get_motion_vector().tolist() )
    #bring motions to same y-coordinate to make 2d alignment possible
    offset_y = prev_frames[-1][1] - sample_frames[0][1]
    sample_frames = transform_euler_frames(sample_frames, [0,0,0],[0,offset_y,0])
    #find aligning 2d transformation
    theta, offset_x, offset_z = find_aligning_transformation(bvh_reader,prev_frames,sample_frames, node_name_map)
    transformation = {"orientation": [0,np.degrees(theta),0], "position": [offset_x,offset_y,offset_z]}
    return transformation


def main():
    p = [5,0,0]
#    translation = [0,0,0]
#    rotation = [0,180,0]
#    p = transform_point2(rotation,translation,p)
#    print p
#
    print p
    translation = [0,0,0]
    rotation = [0,90,0]
    p = transform_point2(rotation,translation,p)
    print p
    transform = create_transformation(rotation,translation)
    transform = np.linalg.inv(transform)
    #rotation = create_local_rotation(rotation)
    p = transform_point(transform,p)
    print p
    return

if __name__ == "__main__":

    main()