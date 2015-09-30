# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:10:21 2014

@author: erhe01

"""
import collections
import numpy as np
from ..external.transformations import euler_matrix, quaternion_from_matrix


class QuaternionFrame(collections.OrderedDict):

    """OrderedDict that contains data for a quaternion frame"""

    def __init__(self, bvh_reader, frame_vector, filter_values=True, ignore_bip_joints=True):
        """Reads an animation frame from a BVH file and fills the list class
           with quaternions of the skeleton nodes

        Parameters
        ----------

         * bvh_reader: BVHReader
        \t Contains skeleton information
        * frame_vector: np.ndarray
        \t animation keyframe frame represented by Euler angles
        * filter_values: Bool
        \t enforce a unique rotation representation

        """
        quaternions = \
            self._get_all_nodes_quat_repr(bvh_reader, frame_vector, filter_values, ignore_bip_joints=ignore_bip_joints)
        collections.OrderedDict.__init__(self, quaternions)

    @classmethod
    def _get_quaternion_from_euler(cls, euler_angles, rotation_order,
                                   filter_values):
        """Convert euler angles to quaternion vector [qw, qx, qy, qz]

        Parameters
        ----------
        * euler_angles: list of floats
        \tA list of ordered euler angles in degress
        * rotation_order: Iteratable
        \t a list that specifies the rotation axis corresponding to the values in euler_angles
        * filter_values: Bool
        \t enforce a unique rotation representation

        """
        # convert euler angles into rotation matrix, then convert rotation matrix
        # into quaternion
        assert len(
            euler_angles) == 3, ('The length of euler angles should be 3!')
        # convert euler angle from degree into radians
        euler_angles = np.deg2rad(euler_angles)
        if rotation_order[0] == 'Xrotation':
            if rotation_order[1] == 'Yrotation':
                rotmat = euler_matrix(euler_angles[0],
                                      euler_angles[1],
                                      euler_angles[2],
                                      axes='rxyz')
            elif rotation_order[1] == 'Zrotation':
                rotmat = euler_matrix(euler_angles[0],
                                      euler_angles[1],
                                      euler_angles[2],
                                      axes='rxzy')
        elif rotation_order[0] == 'Yrotation':
            if rotation_order[1] == 'Xrotation':
                rotmat = euler_matrix(euler_angles[0],
                                      euler_angles[1],
                                      euler_angles[2],
                                      axes='ryxz')
            elif rotation_order[1] == 'Zrotation':
                rotmat = euler_matrix(euler_angles[0],
                                      euler_angles[1],
                                      euler_angles[2],
                                      axes='ryzx')
        elif rotation_order[0] == 'Zrotation':
            if rotation_order[1] == 'Xrotation':
                rotmat = euler_matrix(euler_angles[0],
                                      euler_angles[1],
                                      euler_angles[2],
                                      axes='rzxy')
            elif rotation_order[1] == 'Yrotation':
                rotmat = euler_matrix(euler_angles[0],
                                      euler_angles[1],
                                      euler_angles[2],
                                      axes='rzyx')
        else:
            raise ValueError('Unknown rotation order')
        # convert rotation matrix R into quaternion vector (qw, qx, qy, qz)
        quat = quaternion_from_matrix(rotmat)
        # filter the quaternion see
        # http://physicsforgames.blogspot.de/2010/02/quaternions.html
        if filter_values:
            dot = np.sum(quat)
            if dot < 0:
                quat = -quat
        return quat[0], quat[1], quat[2], quat[3]

    def _get_quaternion_representation(self, bvh_reader, node_name,
                                       frame_vector, filter_values=True):
        """Returns the rotation for one node at one frame of an animation as
           a quaternion

        Parameters
        ----------

        * node_name: String
        \tName of node
        * bvh_reader: BVHReader
        \t BVH data structure read from a file
        * frame_vector: np.ndarray
        \t animation keyframe frame represented by Euler angles
        * filter_values: Bool
        \t enforce a unique rotation representation


        """
        x_idx = bvh_reader.node_channels.index((node_name, 'Xrotation'))
        y_idx = bvh_reader.node_channels.index((node_name, 'Yrotation'))
        z_idx = bvh_reader.node_channels.index((node_name, 'Zrotation'))
        assert y_idx - x_idx == 1 and z_idx - y_idx == 1
        euler_angles_x = frame_vector[x_idx]
        euler_angles_y = frame_vector[y_idx]
        euler_angles_z = frame_vector[z_idx]
        euler_angles = [euler_angles_x, euler_angles_y, euler_angles_z]
        #if node_name.startswith("Bip"):
        #    euler_angles = [0, 0, 0]     # Set Fingers to zero

        rotation_order = (
            'Xrotation',
            'Yrotation',
            'Zrotation')  # hard coded for now
        return QuaternionFrame._get_quaternion_from_euler(
            euler_angles,
            rotation_order,
            filter_values)

    def _get_all_nodes_quat_repr(self, bvh_reader, frame_vector, filter_values, ignore_bip_joints=True):
        """Returns dictionary of all quaternions for all nodes except leave nodes
           Note: bvh_reader.node_names may not include EndSites

        Parameters
        ----------

         * bvh_reader: BVHReader
        \t BVH data structure read from a file
        * frame_vector: np.ndarray
        \t animation keyframe frame represented by Euler angles
        * filter_values: Bool
        \t enforce a unique rotation representation

        """

        for node_name in bvh_reader.node_names:
            # simple fix for ignoring finger joints.
            if (not ignore_bip_joints or not node_name.startswith("Bip")) and 'EndSite' not in node_name:
                yield node_name, self._get_quaternion_representation(bvh_reader,
                                                                     node_name,
                                                                     frame_vector,
                                                                     filter_values)


