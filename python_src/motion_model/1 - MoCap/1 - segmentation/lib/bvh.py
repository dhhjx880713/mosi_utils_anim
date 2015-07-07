# -*- coding: utf-8 -*-

"""
bvh.py
======

@author: mamanns

"""
import os
from collections import OrderedDict
from cgkit.bvh import BVHReader as BVHR
from math import atan2, pi, degrees, asin


class BVHReader(BVHR):
    """Reads BVH files and provides additionally:

     * gen_all_parents: Generator of all parents of a node (bottom up)
     * get_root_positions: Root joint positions (without joint offset)
     * get_angles: Root angles

    """

    def __init__(self, filename):
        BVHR.__init__(self, filename)

        self.keyframes = []  # Used later in onFrame

        self.angle_cache = {}

        self.root = None

        self.read()

        self.node_names = self._get_node_names()
        self.parent_dict = self._get_parent_dict()

    def _get_node_names(self):
        """Returns an OrderedDict from node name to node object.

        This method ignores end sites

        """

        def get_node_names_nodes(node):
            """Returns list of  2-tuples (nodename, node)"""

            nnlist = []

            # End nodes need to be ignored otherwise the ordered dictionary can
            # not be used to get the index for the keyframe values.
            # Nodes that start with Bip need to be read for that purpose and
            # must be later ignored.

            if not node.isEndSite():
                nnlist.append((node.name, node))

            for childnode in node.children:
                nnlist += get_node_names_nodes(childnode)

            return nnlist

        return OrderedDict(get_node_names_nodes(self.root))

    def _get_parent_dict(self):
        """Returns a dict of node names to their parent node's name"""

        parent_dict = {}

        for node_name in self.node_names:
            for child_node in self.node_names[node_name].children:
                parent_dict[child_node.name] = node_name

        return parent_dict

    def intToken(self):
        """Return the next token which must be an int.

        Subclassed because of a bug in cgkit

        """

        tok = self.token()
        try:
            return int(float(tok))

        except ValueError:
            msg = "Syntax error in line {}: Integer expected, got '{}' instead"
            msg.format(self.linenr, tok)
            raise SyntaxError(msg)

    def onHierarchy(self, root):
        """Called after the joint hierarchy was read. Must be subclassed"""

        self.root = root  # Save root for later use

    def onFrame(self, values):
        """Called for each frame in the file. Must be subclassed"""
        self.keyframes.append(values)

    def gen_all_parents(self, node_name):
        """Generator of all parents' node names of node with node_name"""

        while node_name in self.parent_dict:
            parent_name = self.parent_dict[node_name]
            yield parent_name
            node_name = parent_name

    def get_root_positions(self, frame_number=None):
        """Returns a list of lists of x, y, z root coordinate tuples

        Joint offset of sekeleton is not taken into account.
         Returns list of root coordinates if frame_number is given

        """

        if frame_number is None:
            return [frame[:3] for frame in self.keyframes]

        else:
            return self.keyframes[frame_number][:3]

    def get_angles(self, node_name, frame_number=None):
        """Returns a list of lists of angles for node with node_name

        Returns list of angles if frame_number is given

        Parameters
        ----------
         * node_name: String
        \tName of node for which angles are returned
         * frame_number: Integer, defaults to None
        \tNumber of the frame for which anglea are returned if not None

        """

        if frame_number is None:
            idx = self.node_names.keys().index(node_name) * 3 + 3
            return [frame[idx:idx+3] for frame in self.keyframes]

        else:
            if (node_name, frame_number) in self.angle_cache:
                return self.angle_cache[(node_name, frame_number)]

            idx = self.node_names.keys().index(node_name) * 3 + 3
            angles = self.keyframes[frame_number][idx:idx+3]
            self.angle_cache[(node_name, frame_number)] = angles
            return angles


class BVHWriter(object):
    """Write BVH files

    @author: mamauer
    """

    def __init__(self, data, filename, is_quaternion=False):
        if filename[-4:] == '.bvh':
            self.filename = filename
        else:
            self.filename = filename + '.bvh'
        self.data = data

        if not is_quaternion:
            self.frames = data
            self.write()
        else:
            self.frames = []
            for frame in data:
                euler_frame = frame[:3]     # copy root
                for i in xrange(3, len(frame), 4):
                    eulers = self.quaternionToEuler(frame[i:i+4])
                    for e in eulers:
                        euler_frame.append(e)
                self.frames.append(euler_frame)
            self.write()

    #COPIED FROM http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
    # q1 can be non-normalised quaternion #
    def quaternionToEuler(self, q1):
        w, x, y, z = q1
        sqw = w*w
        sqx = x*x
        sqy = y*y
        sqz = z*z
        # if normalised is one, otherwise is correction factor
        unit = sqx + sqy + sqz + sqw
        test = x*y + z*w
        if test > 0.499*unit:   # singularity at north pole
            heading = 2 * atan2(x, w)
            attitude = pi/2
            bank = 0
        elif test < -0.499*unit:    # singularity at south pole
            heading = -2 * atan2(x, w)
            attitude = -pi/2
            bank = 0

        else:
            heading = atan2(2*y*w-2*x*z, sqx - sqy - sqz + sqw)
            attitude = asin(2*test/unit)
            bank = atan2(2*x*w-2*y*z, -sqx + sqy - sqz + sqw)
        # http://www.euclideanspace.com/maths/standards/index.htm
        return [degrees(bank), degrees(heading), degrees(attitude)]

    def get_skeletonstring(self):
        path = os.path.realpath(__file__).split(os.sep)[:-1] + \
            ['skeletonstring.txt']
        f = os.sep.join(path)
        with open(f, 'r') as fp:
            ret_value = fp.read()
        return ret_value

    def write(self):
        """ Write all frames to file """
        fp = open(self.filename, 'w+')
        fp.write(self.get_skeletonstring())
        fp.write("MOTION\n")
        fp.write("Frames: %d\n" % len(self.frames))
        fp.write("Frame Time: 0.013889\n")
        for frame in self.frames:
            fp.write(' '.join([str(i) for i in frame]))
            fp.write('\n')
        fp.close()