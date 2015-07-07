#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
test_clipboard
==============

Unit tests for clipboard.py

"""

import os
import sys

from numpy import isclose

FILEPATH = os.path.realpath(__file__)
TESTPATH = os.sep.join(FILEPATH.split(os.sep)[:-1])

sys.path.insert(0, TESTPATH)
sys.path.insert(0, TESTPATH + (os.sep + os.pardir))
sys.path.insert(0, TESTPATH + (os.sep + os.pardir) * 2)

TESTFILE = TESTPATH + "/data/bvh/81_12.bvh"

import bvh_reader


def params(funcarglist):
    """Test function parameter decorator

    Provides arguments based on the dict funcarglist.

    """

    def wrapper(function):
        function.funcarglist = funcarglist
        return function
    return wrapper


def pytest_generate_tests(metafunc):
    """Enables params to work in py.test environment"""

    for funcargs in getattr(metafunc.function, 'funcarglist', ()):
        metafunc.addcall(funcargs=funcargs)


class TestBVH(object):
    """Unit test class for BVH"""

    def setup_class(self):
        """Class setup method"""
        
        with open(TESTFILE) as infile:
            self.bvh = bvh_reader.BVH(infile)

    def test_read_skeleton(self):
        """Unit test for _read_skeleton"""

        assert "leftEye" in self.bvh.skeleton["head"]["children"]
        assert isclose(self.bvh.skeleton["rFoot_EndSite"]["offset"][2], 12.103)
        assert isclose(self.bvh.skeleton["neck"]["offset"][0], 0)
        assert self.bvh.node_channels[0] == ("hip", "Xposition")

    def test_read_frametime(self):
        """Unit test for _read_frametime"""

        assert isclose(self.bvh.frame_time, 0.00833333)

    def test_read_frames(self):
        """Unit test for _read_frames"""

        assert len(self.bvh.frames) == 503
        assert isclose(self.bvh.frames[0][2], 12.2403)

    param_get_angles = [
        {
            'node_channels': [("hip", "Xposition")],
            'frame': 0,
            'resindex': 0,
            'res': 50.3196,
        },
        {
            'node_channels': [("hip", "Xposition")],
            'frame': 1,
            'resindex': 0,
            'res': 53.9146,
        },
        {
            'node_channels': [("hip", "Xposition"), ("hip", "Yposition")],
            'frame': 0,
            'resindex': 0,
            'res': 50.3196,
        },
        {
            'node_channels': [("hip", "Xposition"), ("hip", "Yposition")],
            'frame': 0,
            'resindex': 1,
            'res': 84.8928,
        },
        {
            'node_channels': [("hip", "Xposition"), ("hip", "Yposition")],
            'frame': 5,
            'resindex': 1,
            'res': 83.3711,
        },
    ]

    @params(param_get_angles)
    def test_get_angles(self, node_channels, frame, resindex, res):
        """Unit test for get_angles"""

        assert isclose(self.bvh.get_angles(*node_channels)[frame][resindex],
                       res)
