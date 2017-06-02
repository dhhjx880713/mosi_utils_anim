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

EULER_LEN = 3
QUAT_LEN = 4
TRANSLATION_LEN = 3
TOE_NODES = ["Bip01_R_Toe0", "Bip01_L_Toe0"]

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
            lines = infile.readlines()
            self.process_lines(lines)
            infile.close()
        self.filename = os.path.split(infilename)[-1]

    @classmethod
    def init_from_string(cls, skeleton_string):
        bvh_reader = cls(infilename="")
        lines = skeleton_string.split("\n")
        bvh_reader.process_lines(lines)
        return bvh_reader

    def _read_skeleton(self, lines, line_index=0, n_lines=-1):
        """Reads the skeleton part of a BVH file"""
        line_index = line_index
        parents = []
        level = 0
        name = None
        if n_lines == -1:
            n_lines = len(lines)

        while line_index < n_lines:
            if lines[line_index].startswith("MOTION"):
                break

            else:
                #print lines[line_index]
                if "{" in lines[line_index]:
                    parents.append(name)
                    level += 1

                if "}" in lines[line_index]:
                    level -= 1
                    parents.pop(-1)
                    if level == 0:
                        break

                line_split = lines[line_index].strip().split()

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
                line_index += 1
        return line_index

    def _read_frametime(self, lines, line_index):
        """Reads the frametime part of a BVH file"""

        if lines[line_index].startswith("Frame Time:"):
            self.frame_time = float(lines[line_index].split(":")[-1].strip())
        else:
            self.frame_time = 0.013889  # TODO use constant

    def _read_frames(self, lines, line_index, n_lines=-1):
        """Reads the frames part of a BVH file"""
        line_index = line_index
        if n_lines == -1:
            n_lines = len(lines)
        frames = []
        while line_index < n_lines:
            #print lines[line_index]
            line_split = lines[line_index].strip().split()
            frames.append(np.array(map(float, line_split)))
            line_index += 1

        self.frames = np.array(frames)
        return line_index

    def process_lines(self, lines):
        """Reads BVH file infile

        Parameters
        ----------
         * infile: Filelike object, optional
        \tBVH file

        """
        line_index = 0
        n_lines = len(lines)
        while line_index < n_lines:
            if lines[line_index].startswith("HIERARCHY"):
                line_index = self._read_skeleton(lines, line_index, n_lines)
            if lines[line_index].startswith("MOTION"):
                self._read_frametime(lines, line_index+2)
                line_index = self._read_frames(lines, line_index+3, n_lines)
            else:
                line_index += 1


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

    def get_animated_joints(self):
        """Returns an ordered list of joints which have animation channels"""
        for name, node in self.node_names.iteritems():
            if "channels" in node.keys() and len(node["channels"]) > 0:
                yield name


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
        outfile = open(filename, 'wb')
        outfile.write(bvh_string)
        outfile.close()

    def generate_bvh_string(self):
        bvh_string = self._generate_hierarchy_string(self.skeleton) + "\n"
        if self.is_quaternion:
            #euler_frames = self.convert_quaternion_to_euler_frames_skipping_fixed_joints(self.frame_data, self.is_quaternion)
            euler_frames = self.convert_quaternion_to_euler_frames(self.skeleton, self.frame_data)

        else:
            euler_frames = self.frame_data
        bvh_string += self._generate_bvh_frame_string(euler_frames, self.frame_time)
        return bvh_string

    def _generate_hierarchy_string(self, skeleton):
        """ Initiates the recursive generation of the skeleton structure string
            by calling _generate_joint_string with the root joint
        """
        hierarchy_string = "HIERARCHY\n"
        hierarchy_string += self._generate_joint_string(skeleton.root, skeleton, 0)
        return hierarchy_string

    def _generate_joint_string(self, joint, skeleton, joint_level):
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
            if len(skeleton.nodes[joint].children) > 0:
                joint_string += tab_string + "JOINT " + joint + "\n"
            else:
                joint_string += tab_string + "End Site" + "\n"

        # open bracket add offset
        joint_string += tab_string + "{" + "\n"
        offset = skeleton.nodes[joint].offset
        joint_string += tab_string + "\t" + "OFFSET " + "\t " + \
            str(offset[0]) + "\t " + str(offset[1]) + "\t " + str(offset[2]) + "\n"

        if len(skeleton.nodes[joint].children) > 0:
            # channel information
            channels = skeleton.nodes[joint].channels
            joint_string += tab_string + "\t" + \
                "CHANNELS " + str(len(channels)) + " "
            for tok in channels:
                joint_string += tok + " "
            joint_string += "\n"

            joint_level += 1
            # recursive call for all children
            for child in skeleton.nodes[joint].children:
                joint_string += self._generate_joint_string(child.node_name, skeleton, joint_level)

        # close the bracket
        joint_string += tab_string + "}" + "\n"
        return joint_string

    def convert_quaternion_to_euler_frames(self, skeleton, quat_frames):
        """ Converts the joint rotations from quaternion to euler rotations
            * quat_frames: array of motion vectors with rotations represented as quaternion

        """
        joint_names = self.skeleton.get_joint_names()
        n_frames = len(quat_frames)
        n_joints = len(joint_names)
        n_params = n_joints*QUAT_LEN +TRANSLATION_LEN
        euler_frames = np.zeros((n_frames, n_params))
        for frame_idx, quat_frame in enumerate(quat_frames):
            euler_frames[frame_idx,:TRANSLATION_LEN] = quat_frame[:TRANSLATION_LEN]
            d_offset = TRANSLATION_LEN
            s_offset = TRANSLATION_LEN
            for joint_name in joint_names:
                rotation_order = skeleton.nodes[joint_name].channels[-3:]
                #print d_offset,s_offset, idx
                euler_frames[frame_idx,d_offset:d_offset+EULER_LEN] = BVHWriter._quaternion_to_euler(quat_frame[s_offset:s_offset+QUAT_LEN], rotation_order)
                d_offset += EULER_LEN
                s_offset += QUAT_LEN
        return euler_frames

    def _generate_bvh_frame_string(self,euler_frames, frame_time):
        """
            Converts a list of euler frames into the BVH file representation.
            * frame_time: time in seconds for the display of each keyframe
        """

        # create frame string
        frame_parameter_string = "MOTION\n"
        frame_parameter_string += "Frames: " + str(len(euler_frames)) + "\n"
        frame_parameter_string += "Frame Time: " + str(frame_time) + "\n"
        for frame in euler_frames:
            frame_parameter_string += ' '.join([str(f) for f in frame])
            frame_parameter_string += '\n'

        return frame_parameter_string

    def convert_quaternion_to_euler_frames_skipping_fixed_joints(self, frame_data, is_quaternion=False):
            """ Converts the joint rotations from quaternion to euler rotations
                Note: for the toe joints of the rocketbox skeleton a hard set value is used
                * frame_data: array of motion vectors, either as euler or quaternion
                * node_names: OrderedDict containing the nodes of the skeleton accessible by their name
                * is_quaternion: defines wether the frame_data is quaternion data
                                or euler data
            """

            skip_joints = not self.skeleton.is_motion_vector_complete(frame_data, is_quaternion)
            # print "skip joints", skip_joints
            # convert to euler frames if necessary
            if not is_quaternion:
                if not skip_joints:
                    euler_frames = frame_data
                else:
                    euler_frames = []
                    for frame in frame_data:
                        euler_frame = self._get_euler_frame_from_partial_euler_frame(frame, skip_joints)
                        euler_frames.append(euler_frame)
            else:
                # check whether or not "Bip" frames should be ignored
                euler_frames = []
                for frame in frame_data:
                    if skip_joints:
                        euler_frame = self._get_euler_frame_from_partial_quaternion_frame(frame)
                    else:
                        euler_frame = self._get_euler_frame_from_quaternion_frame(frame)
                    # print len(euler_frame), euler_frame
                    euler_frames.append(euler_frame)
            return euler_frames

    def _get_euler_frame_from_partial_euler_frame(self, frame, skip_joints):
        euler_frame = frame[:3]
        joint_idx = 0
        for node_name in node_names:
            if "children" in node_names[node_name].keys():# ignore end sites
                if not node_name.startswith("Bip") or not skip_joints:
                    if node_name in TOE_NODES:
                        # special fix for unused toe parameters
                        euler_frame = np.concatenate((euler_frame,([0.0, -90.0, 0.0])),axis=0)
                    else:
                        i = joint_idx * EULER_LEN + TRANSLATION_LEN
                        euler_frame = np.concatenate((euler_frame, frame[i:i + EULER_LEN]), axis=0)
                    joint_idx += 1
                else:
                    if node_name in TOE_NODES:
                        # special fix for unused toe parameters
                        euler_frame = np.concatenate((euler_frame,([0.0, -90.0, 0.0])),axis=0)
                    else:
                        euler_frame = np.concatenate((euler_frame,([0, 0, 0])),axis=0)  # set rotation to 0
            return euler_frame

    def _get_euler_frame_from_partial_quaternion_frame(self, frame):
        euler_frame = frame[:3]     # copy root
        joint_idx = 0
        for node_name in self.skeleton.nodes.keys():
            if len(self.skeleton.nodes[node_name].channels) > 0:# ignore end sites completely
                if not node_name.startswith("Bip"):
                    i = joint_idx * QUAT_LEN + TRANSLATION_LEN
                    if node_name == self.skeleton.root:
                        channels = self.skeleton.nodes[node_name].channels[TRANSLATION_LEN:]
                    else:
                        channels = self.skeleton.nodes[node_name].channels
                    euler = BVHWriter._quaternion_to_euler(frame[i:i + QUAT_LEN], channels)
                    euler_frame = np.concatenate((euler_frame, euler), axis=0)
                    joint_idx += 1
                else:
                    if node_name in TOE_NODES:
                        # special fix for unused toe parameters
                        euler_frame = np.concatenate((euler_frame,([0.0, -90.0, 0.0])),axis=0)
                    else:
                        euler_frame = np.concatenate((euler_frame,([0, 0, 0])),axis=0)  # set rotation to 0
        return euler_frame

    def _get_euler_frame_from_quaternion_frame(self, frame):
        euler_frame = frame[:3]  # copy root
        joint_idx = 0
        for node_name in self.skeleton.nodes.keys():
            if len(self.skeleton.nodes[node_name].channels) > 0:  # ignore end sites completely
                if node_name in TOE_NODES:
                    # special fix for unused toe parameters
                    euler_frame = np.concatenate((euler_frame, ([0.0, -90.0, 0.0])), axis=0)
                else:
                    i = joint_idx * QUAT_LEN + TRANSLATION_LEN
                    if node_name == self.skeleton.root:
                        channels = self.skeleton.nodes[node_name].channels[TRANSLATION_LEN:]
                    else:
                        channels = self.skeleton.nodes[node_name].channels
                    euler = BVHWriter._quaternion_to_euler(frame[i:i + QUAT_LEN], channels)
                    euler_frame = np.concatenate((euler_frame, euler), axis=0)
                joint_idx += 1
        return euler_frame

    @classmethod
    def _quaternion_to_euler(cls, quat, rotation_order=['Xrotation','Yrotation','Zrotation']):
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

        rotation_order = [v.lower() for v in rotation_order]
        quat = np.asarray(quat)
        quat = quat / np.linalg.norm(quat)
        rotmat_quat = quaternion_matrix(quat)
        if rotation_order[0] == 'xrotation':
            if rotation_order[1] == 'yrotation':
                euler_angles = np.rad2deg(euler_from_matrix(rotmat_quat, 'rxyz'))
            elif rotation_order[1] == 'zrotation':
                euler_angles = np.rad2deg(euler_from_matrix(rotmat_quat, 'rxzy'))
        elif rotation_order[0] == 'yrotation':
            if rotation_order[1] == 'xrotation':
                euler_angles = np.rad2deg(euler_from_matrix(rotmat_quat, 'ryxz'))
            elif rotation_order[1] == 'zrotation':
                euler_angles = np.rad2deg(euler_from_matrix(rotmat_quat, 'ryzx'))
        elif rotation_order[0] == 'zrotation':
            if rotation_order[1] == 'xrotation':
                euler_angles = np.rad2deg(euler_from_matrix(rotmat_quat, 'rzxy'))
            elif rotation_order[1] == 'yrotation':
                euler_angles = np.rad2deg(euler_from_matrix(rotmat_quat, 'rzyx'))
        return euler_angles


