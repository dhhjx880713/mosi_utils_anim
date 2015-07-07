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
from bvh import BVHWriter
import numpy as np

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

class TestBVHWriter(object):
    """BVHWriter test class"""

    def setup_method(self, method):
        import re
        testfile = TESTPATH + "/walk_001_1_rightStance_86_128.bvh"
        f = open(testfile)
        self.file_content = f.readlines()
        f.close()
        self.hierarchy_string = ""
        self.frame_parameter_string = ""
        frame_parameters = False
        i = 0
        while i < len(self.file_content):
            line = self.file_content[i]
            if re.match("(.*)MOTION(.*)",line):
                frame_parameters = True
            if not frame_parameters:
                self.hierarchy_string+= line
            else:
                self.frame_parameter_string+=line
            i+=1
        self.reader = BVHReader(testfile)
        self.frames = np.array(self.reader.keyframes)
        
   
        
    def test_generate_hierarchy_string(self):
        """Unit test for _generate_hierarchy_string"""
        
        writer = BVHWriter(None,None,None,None)
        assert writer._generate_hierarchy_string(self.reader.root) == \
                                                    self.hierarchy_string
        
                                                    
    param_get_angles = [{'angles':[0.707107, 0.707107, 0, 0],
                        "expected": [90, 0, 0]},
                        {'angles':[0.589646, 0.390278, 0.589646, -0.390278],
                        "expected": [90.0, 23.0, -90.0]}
                            ]
    @params(param_get_angles)
    def test_quaternion_to_euler(self,angles,expected):
        """Unit test for_quaternion_to_euler"""
        
        writer = BVHWriter(None,None,None,None)
                                    
        e = writer._quaternion_to_euler(angles)
        for i in xrange(len(expected)):
            assert round(e[i], 3) == round(expected[i], 3)
   