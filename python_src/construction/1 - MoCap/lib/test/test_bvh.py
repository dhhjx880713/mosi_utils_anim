"""
test_bvh.py
Unit test for bvh.py
======

Author: Han

"""

import os
import sys

from libtest import params, pytest_generate_tests

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(1, TESTPATH)
sys.path.insert(1, TESTPATH + (os.sep + os.pardir))

from bvh import BVHReader


class TestBVHReader(object):
    """BVHReader test class"""

    def setup_method(self, method):
        testfile = TESTPATH + "/walk_001_1_rightStance_86_128.bvh"
        self.bvh_reader = BVHReader(testfile)

    def test_onHierarchy(self):
        '''unit test for onHierarchy()'''

        self.bvh_reader.onHierarchy(self.bvh_reader.root)

        assert self.bvh_reader.root.name == "Hips"

    param_onHierarchy = [{'res': [-1.1578, 90.0, 4.68537]}]

    @params(param_onHierarchy)
    def test_get_root_positions(self, res):
        """Unit test for get_root_positions"""

        pos = self.bvh_reader.get_root_positions()[0]

        for ax in xrange(3):
            assert round(pos[ax], 3) == round(res[ax], 3)

    param_get_node_names = [{'res': 'Hips'}, {'res': 'LeftShoulder'}]

    @params(param_get_node_names)
    def test_get_node_names(self, res):
        """Unit test for get_node_name"""
        node_names = self.bvh_reader._get_node_names().keys()
        assert res in node_names

    param_get_parent_dict = [{'jointname': 'LeftFoot', 'res': 'LeftLeg'},
                             {'jointname': 'LeftShoulder', 'res': 'Neck'}]

    @params(param_get_parent_dict)
    def test_get_parent_dict(self, jointname, res):
        """Unit test for _get_parent_dict"""
        parent_dict = self.bvh_reader._get_parent_dict()
        assert parent_dict[jointname] == res

    param_intToken = [{'testToken': '45.0', 'res': 45}]

    @params(param_intToken)
    def test_intToken(self, testToken, res):
        """Unit test for intToken"""
        self.bvh_reader.tokenlist = [testToken]
        assert self.bvh_reader.intToken() == res

    param_onFrame = [{'frameValue': [1, 2, 3, 4, 5, 6, 7],
                      'res': [1, 2, 3, 4, 5, 6, 7]}]

    @params(param_onFrame)
    def test_onFrame(self, frameValue, res):
        """Unit test for onFrame"""
        self.bvh_reader.onFrame(frameValue)
        assert self.bvh_reader.keyframes[-1] == res
        del(self.bvh_reader.keyframes[-1])

    param_gen_all_parents = [{'node_name': 'Hips',
                              'res': []},
                             {'node_name': 'LeftHand',
                              'res': ['LeftForeArm', 'LeftArm', 'LeftShoulder',
                                      'Neck', 'Spine_1', 'Spine', 'Hips']}]

    @params(param_gen_all_parents)
    def test_gen_all_parents(self, node_name, res):
        """Unit test for gen_all_parents """
        assert list(self.bvh_reader.gen_all_parents(node_name)) == res

    param_get_angles = [{'node_name': 'RightShoulder',
                         'res': [-166.268, -11.727, -107.406]}]

    @params(param_get_angles)
    def test_get_angles(self, node_name, res):
        """Unit test for get_angles"""
        angles = self.bvh_reader.get_angles(node_name)[0]
        for i in xrange(len(res)):
            assert round(angles[i], 3) == round(res[i], 3)
