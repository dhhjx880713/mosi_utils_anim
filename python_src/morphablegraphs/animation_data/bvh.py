#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

BVH
===

Biovision file format classes for reading and writing.
BVH Reader by Martin Manns
BVH Writer by Erik Herrmann

"""

from collections import OrderedDict
import numpy as np
from ..external.transformations import quaternion_matrix,\
                                       euler_from_matrix
import os


class BVHReader(object):

    """Biovision file format class

    Parameters
    ----------
     * infile: string
    \t path to BVH file that is loaded initially

    """

    def __init__(self, infilename=""):

        self.node_names = OrderedDict()
        self.node_channels = []
        self.parent_dict = {}
        self.frame_time = None
        self.frames = None
        self.root = ""  # needed for the bvh writer
        if infilename != "":
            infile = open(infilename, "rb")
            self.read(infile)
        self.filename = os.path.split(infilename)[-1]
        infile.close()

    def _read_skeleton(self, infile):
        """Reads the skeleton part of a BVH file"""

        parents = []
        level = 0
        name = None

        for line in infile:
            if "{" in line:
                parents.append(name)
                level += 1

            if "}" in line:
                level -= 1
                parents.pop(-1)
                if level == 0:
                    break

            line_split = line.strip().split()

            if line_split:
                if line_split[0] == "ROOT":
                    name = line_split[1]
                    self.root = name
                    self.node_names[name] = {
                        "children": [], "level": level, "channels": []}

                elif line_split[0] == "JOINT":
                    name = line_split[1]
                    self.node_names[name] = {
                        "children": [], "level": level, "channels": []}
                    self.node_names[parents[-1]]["children"].append(name)

                elif line_split[0] == "CHANNELS":
                    for channel in line_split[2:]:
                        self.node_channels.append((name, channel))
                        self.node_names[name]["channels"].append(channel)

                elif line_split == ["End", "Site"]:
                    name += "_" + "".join(line_split)
                    self.node_names[name] = {"level": level}
                    # also the end sites need to be adde as children
                    self.node_names[parents[-1]]["children"].append(name)

                elif line_split[0] == "OFFSET" and name in self.node_names.keys():
                    offset = [float(x) for x in line_split[1:]]
                    self.node_names[name]["offset"] = offset

    def _read_frametime(self, infile):
        """Reads the frametime part of a BVH file"""

        for line in infile:
            if line.startswith("Frame Time:"):
                self.frame_time = float(line.split(":")[-1].strip())
                break

    def _read_frames(self, infile):
        """Reads the frames part of a BVH file"""

        frames = []
        for line in infile:

            line_split = line.strip().split()
            frames.append(map(float, line_split))

        self.frames = np.array(frames)

    def read(self, infile):
        """Reads BVH file infile

        Parameters
        ----------
         * infile: Filelike object, optional
        \tBVH file

        """

        for line in infile:
            if line.startswith("HIERARCHY"):
                break

        self._read_skeleton(infile)

        for line in infile:
            if line.startswith("MOTION"):
                break

        self._read_frametime(infile)
        self._read_frames(infile)

    def get_angles(self, *node_channels):
        """Returns numpy array of angles in all frames for specified channels

        Parameters
        ----------
         * node_channels: 2-tuples of strings
        \tEach tuple contains joint name and channel name
        \te.g. ("hip", "Xposition")

        """

        indices = [self.node_channels.index(nc) for nc in node_channels]
        return self.frames[:, indices]


class BVHWriter(object):

    """ Saves an input motion defined either as an array of euler or quaternion 
    frame vectors as a BVH file.

    Parameters
    ----------
    * filename: String or None
        Name of the created bvh file. Can be None.
    * skeleton: Skeleton
        Skeleton structure needed to copy the hierarchy
    * frame_data: np.ndarray
        array of motion vectors, either with euler or quaternion as 
        rotation parameters
    * frame_time: float
        time in seconds for the display of each keyframe
    * is_quaternion: Boolean
        Defines wether the frame_data is quaternion data or euler data
    """

    def __init__(self, filename, skeleton, frame_data, frame_time, is_quaternion=False):
        self.skeleton = skeleton
        self.frame_data = frame_data
        self.frame_time = frame_time
        self.is_quaternion = is_quaternion
        if filename is not None:
            self.write(filename)

    def write(self, filename):
        """ Write the hierarchy string and the frame parameter string to file
        """
        bvh_string = self.generate_bvh_string()
        if filename[-4:] == '.bvh':
            filename = filename
        else:
            filename = filename + '.bvh'
        fp = open(filename, 'wb')
        fp.write(bvh_string)
        fp.close()

    def generate_bvh_string(self):
        bvh_string = self._generate_hierarchy_string(
            self.skeleton.root, self.skeleton.node_names) + "\n"
        bvh_string += self._generate_frame_parameter_string(
            self.frame_data, self.skeleton.node_names, self.frame_time, self.is_quaternion)
        return bvh_string

    def _generate_hierarchy_string(self, root, node_names):
        """ Initiates the recursive generation of the skeleton structure string
            by calling _generate_joint_string with the root joint
        """
        hierarchy_string = "HIERARCHY\n"
        hierarchy_string += self._generate_joint_string(root, node_names, 0)
        return hierarchy_string

    def _generate_joint_string(self, joint, node_names, joint_level):
        """ Recursive traversing of the joint hierarchy to create a 
            skeleton structure string in the BVH format
        """
        joint_string = ""
        temp_level = 0
        tab_string = ""
        while temp_level < joint_level:
            tab_string += "\t"
            temp_level += 1

        # determine joint type
        if joint_level == 0:
            joint_string += tab_string + "ROOT " + joint + "\n"
        else:
            if joint != "End Site" and "channels" in node_names[joint].keys():
                joint_string += tab_string + "JOINT " + joint + "\n"
            else:
                joint_string += tab_string + "End Site" + "\n"

        # open bracket add offset
        joint_string += tab_string + "{" + "\n"
        joint_string += tab_string + "\t"  + "OFFSET " + "\t " + \
            str(node_names[joint]["offset"][0]) + "\t " + str(node_names[joint]["offset"][1]) \
            + "\t " + str(node_names[joint]["offset"][2]) + "\n"

        if joint != "End Site" and "channels" in node_names[joint].keys():
            # channel information
            channels = node_names[joint]["channels"]
            joint_string += tab_string + "\t" + \
                "CHANNELS " + str(len(channels)) + " "
            for tok in channels:
                joint_string += tok + " "
            joint_string += "\n"

            joint_level += 1
            # recursive call for all children
            for child in node_names[joint]["children"]:
                joint_string += self._generate_joint_string(child, node_names,
                                                            joint_level)

        # close the bracket
        joint_string += tab_string + "}" + "\n"
        return joint_string

    def _generate_frame_parameter_string(self, frame_data, node_names, frame_time, is_quaternion=False):
        """ Converts the joint parameters for a list of frames into the BVH file representation. 
            Note: for the toe joints of the rocketbox skeleton a hard set value is used
            * frame_data: array of motion vectors, either as euler or quaternion
            * node_names: OrderedDict containing the nodes of the skeleton accessible by their name
            * frame_time: time in seconds for the display of each keyframe
            * is_quaternion: defines wether the frame_data is quaternion data 
                            or euler data
        """

        # convert to euler frames if necessary
        if not is_quaternion:
            skip_joints = True
            if len(frame_data[0]) == len([n for n in node_names if "children" in node_names[n].keys()]) * 3 + 3:
                skip_joints = False
            if not skip_joints:
                euler_frames = frame_data
            else:
                euler_frames = []
                for frame in frame_data:
                    euler_frame = frame[:3]
                    joint_idx = 0
                    # go through the node names to
                    for node_name in node_names:
                                                # to append specific data
                        # ignore end sites completely
                        if "children" in node_names[node_name].keys():
                            if not node_name.startswith("Bip") or not skip_joints:
                                if node_name in ["Bip01_R_Toe0", "Bip01_L_Toe0"]:
                                    # special fix for unused toe parameters
                                    euler_frame = np.concatenate(
                                        (euler_frame, ([90.0, -1.00000000713e-06, 75.0	])), axis=0)
                                else:
                                    # print node_name
                                    # get start index in the frame vector
                                    i = joint_idx * 3 + 3
                                    if node_names[node_name]["level"] == 0:
                                        channels = node_names[
                                            node_name]["channels"][3:]
                                    else:
                                        channels = node_names[
                                            node_name]["channels"]

                                    euler_frame = np.concatenate(
                                        (euler_frame, frame[i:i + 3]), axis=0)
                                joint_idx += 1
                            else:
                                if node_name in ["Bip01_R_Toe0", "Bip01_L_Toe0"]:
                                    # special fix for unused toe parameters
                                    euler_frame = np.concatenate(
                                        (euler_frame, ([90.0, -1.00000000713e-06, 75.0	])), axis=0)
                                else:
                                    euler_frame = np.concatenate(
                                        (euler_frame, ([0, 0, 0])), axis=0)  # set rotation to 0

                    euler_frames.append(euler_frame)
        else:
            # check whether or not "Bip" frames should be ignored
            skip_joints = True
            if len(frame_data[0]) == len([n for n in node_names if "children" in node_names[n].keys()]) * 4 + 3:
                skip_joints = False
            euler_frames = []
            for frame in frame_data:
                euler_frame = frame[:3]     # copy root
                joint_idx = 0
                for node_name in node_names:  # go through the node names to
                                              # to append specific data
                    # ignore end sites completely
                    if "children" in node_names[node_name].keys():
                        if not node_name.startswith("Bip") or not skip_joints:
                            if node_name in ["Bip01_R_Toe0", "Bip01_L_Toe0"]:
                                # special fix for unused toe parameters
                                euler_frame = np.concatenate(
                                    (euler_frame, ([90.0, -1.00000000713e-06, 75.0	])), axis=0)
                            else:
                                # print node_name
                                # get start index in the frame vector
                                i = joint_idx * 4 + 3
                                if node_names[node_name]["level"] == 0:
                                    channels = node_names[
                                        node_name]["channels"][3:]
                                else:
                                    channels = node_names[
                                        node_name]["channels"]

                                euler_frame = np.concatenate(
                                    (euler_frame, self._quaternion_to_euler(frame[i:i + 4], channels)), axis=0)
                            joint_idx += 1
                        else:
                            if node_name in ["Bip01_R_Toe0", "Bip01_L_Toe0"]:
                                # special fix for unused toe parameters
                                euler_frame = np.concatenate(
                                    (euler_frame, ([90.0, -1.00000000713e-06, 75.0	])), axis=0)
                            else:
                                euler_frame = np.concatenate(
                                    (euler_frame, ([0, 0, 0])), axis=0)  # set rotation to 0

                euler_frames.append(euler_frame)

        # create frame string
        frame_parameter_string = "MOTION\n"
        frame_parameter_string += "Frames: " + str(len(euler_frames)) + "\n"
        frame_parameter_string += "Frame Time: " + str(frame_time) + "\n"
        for frame in euler_frames:
            frame_parameter_string += ' '.join([str(f) for f in frame])
            frame_parameter_string += '\n'

        return frame_parameter_string

    def _quaternion_to_euler(self, q, rotation_order=['Xrotation',
                                                      'Yrotation',
                                                      'Zrotation']):
        """
        Parameters
        ----------
        * q: list of floats
        \tQuaternion vector with form: [qw, qx, qy, qz]
    
        Return
        ------
        * euler_angles: list
        \tEuler angles in degree with specified order
        """
        q = np.asarray(q)
        q = q / np.linalg.norm(q)
        Rq = quaternion_matrix(q)
        if rotation_order[0] == 'Xrotation':
            if rotation_order[1] == 'Yrotation':
                euler_angles = np.rad2deg(euler_from_matrix(Rq, 'rxyz'))
            elif rotation_order[1] == 'Zrotation':
                euler_angles = np.rad2deg(euler_from_matrix(Rq, 'rxzy'))
        elif rotation_order[0] == 'Yrotation':
            if rotation_order[1] == 'Xrotation':
                euler_angles = np.rad2deg(euler_from_matrix(Rq, 'ryxz'))
            elif rotation_order[1] == 'Zrotation':
                euler_angles = np.rad2deg(euler_from_matrix(Rq, 'ryzx'))
        elif rotation_order[0] == 'Zrotation':
            if rotation_order[1] == 'Xrotation':
                euler_angles = np.rad2deg(euler_from_matrix(Rq, 'rzxy'))
            elif rotation_order[1] == 'Yrotation':
                euler_angles = np.rad2deg(euler_from_matrix(Rq, 'rzyx'))
        return euler_angles.tolist()


