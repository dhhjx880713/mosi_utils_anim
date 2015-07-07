#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Martin Manns

"""

BVH
===

Biovision file format class.
Currently, only reading bvh files is supported.

"""

import numpy


class BVH(object):
    """Biovision file format class

    Parameters
    ----------
     * infile: Filelike object, optional
    \tBVH file that is loaded initially

    """

    def __init__(self, infile=None):

        self.skeleton = {}
        self.node_channels = []
        self.frame_time = None
        self.frames = None

        if infile is not None:
            self.read(infile)

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
                    self.skeleton[name] = {"children": []}

                elif line_split[0] == "JOINT":
                    name = line_split[1]
                    self.skeleton[name] = {"children": []}
                    self.skeleton[parents[-1]]["children"].append(name)

                elif line_split[0] == "CHANNELS":
                    for channel in line_split[2:]:
                        self.node_channels.append((name, channel))

                elif line_split == ["End", "Site"]:
                    name += "_" + "".join(line_split)
                    self.skeleton[name] = {}

                elif line_split[0] == "OFFSET":
                    offset = [float(x) for x in line_split[1:]]
                    self.skeleton[name]["offset"] = offset

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

        self.frames = numpy.array(frames)

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
