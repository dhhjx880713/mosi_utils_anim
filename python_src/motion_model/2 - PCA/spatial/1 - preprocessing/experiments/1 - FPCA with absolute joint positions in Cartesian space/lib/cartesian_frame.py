# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 13:09:01 2014

@author: mamanns, erhe01

"""
import collections
from math import radians
from itertools import izip

import fk3


class CartesianFrame(collections.OrderedDict):
    """OrderedDict that contains data for a cartesian frame

       On instantiation, CartesianFrame reads an animation frame from a
       BVH file and fills itself with cartesian coordinates of the skeleton
       nodes

       Parameters
       ----------

        * bvh_reader: BVHReader
       \tBVH data structure read from a file
        * frame_number: Integer
       \tAnimation frame number that gets extracted

        """

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

    def __init__(self, bvh_reader, frame_number):
        cartesian_coordinates = \
            self._get_all_nodes_cartesian_coordinates(bvh_reader, frame_number)
        collections.OrderedDict.__init__(self, cartesian_coordinates)

    def _get_cartesian_coordinates(self, bvh_reader, node_name, frame_number):
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

        if bvh_reader.node_names[node_name]._is_root:
            root_frame_position = bvh_reader.get_root_positions(frame_number)
            root_node_offset = bvh_reader.node_names[node_name].offset

            return [t + o for t, o in
                    izip(root_frame_position, root_node_offset)]

        else:
            # Names are generated bottom to up --> reverse
            chain_names = list(bvh_reader.gen_all_parents(node_name))
            chain_names.reverse()
            chain_names += [node_name]  # Node is not in its parent list

            eul_angles = (bvh_reader.get_angles(nodename, frame_number)
                          for nodename in chain_names)

            rad_angles = (map(radians, eul_angle) for eul_angle in eul_angles)

            thx, thy, thz = map(list, zip(*rad_angles))

            offsets = [bvh_reader.node_names[nodename].offset
                       for nodename in chain_names]

            # Add root offset to frame offset list
            root_position = bvh_reader.get_root_positions(frame_number)
            offsets[0] = [r + o for r, o in izip(root_position, offsets[0])]

            ax, ay, az = map(list, izip(*offsets))

            # f_idx identifies the kinematic forward transform function
            # This does not lead to a negative index because the root is
            # handled separately

            f_idx = len(ax) - 2

            return self.fk_funcs[f_idx](ax, ay, az, thx, thy, thz)

    def _get_all_nodes_cartesian_coordinates(self, bvh_reader, frame_number):
        """Returns list of all cartesian coordinates of all nodes

        Parameters
        ----------

         * bvh_reader: BVHReader
        \t BVH data structure read from a file
        * frame_number: Integer
        \t animation frame number that gets extracted
        """
        for node_name in bvh_reader.node_names:
            if not node_name.startswith("Bip"):
                yield node_name, self._get_cartesian_coordinates(bvh_reader,
                                                                 node_name,
                                                                 frame_number)


def main():
    from bvh import BVHReader
    filepath = "test/walk_001_1_rightStance_86_128.bvh"
    bvh_reader = BVHReader(filepath)
    frame = CartesianFrame(bvh_reader, 0)
    print frame.values()


if __name__ == '__main__':
    main()
