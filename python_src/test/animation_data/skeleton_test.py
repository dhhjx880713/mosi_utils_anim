__author__ = 'du'
import os
import sys
import collections
ROOT_DIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
sys.path.append(ROOT_DIR)
TEST_LIB_PATH = ROOT_DIR + 'test'
sys.path.append(TEST_LIB_PATH)
from animation_data.bvh import BVHReader
from animation_data.skeleton import Skeleton
from libtest import params, pytest_generate_tests
TEST_DATA_PATH = ROOT_DIR +  '../test_data/animation_data'

class TestSkeleton(object):

    def setup_class(self):
        test_file = TEST_DATA_PATH + os.sep + 'walk_001_1_rightStance_86_128.bvh'
        test_bvhreader = BVHReader(test_file)
        self.skeleton = Skeleton(test_bvhreader)

    param_get_parent_dict = [{'res': {'Head': 'Neck'}},
                             {'res': {'Bip01_L_Finger3': 'LeftHand'}}]

    @params(param_get_parent_dict)
    def test_get_parent_dict(self, res):
        parent_dic = self.skeleton._get_parent_dict()
        for key, value in res.iteritems():
            assert key in parent_dic.keys() and parent_dic[key] == value

    param_gen_all_parents = [{'node_name': 'Bip01_L_Finger0',
                              'res': ['LeftHand', 'LeftForeArm', 'LeftArm', 'LeftShoulder', 'Neck', 'Spine_1', 'Spine',
                                      'Hips']}]

    @params(param_gen_all_parents)
    def test_gen_all_parents(self, node_name, res):
        parents = []
        for joint in self.skeleton.gen_all_parents(node_name):
            parents.append(joint)
        assert  parents == res

    param_set_joint_weights = [{'res': [1.0, 0.36787944117144233, 0.1353352832366127, 0.049787068367863944,
                                        0.018315638888734179, 0.018315638888734179, 0.006737946999085467,
                                        0.0024787521766663585, 0.00091188196555451624, 0.018315638888734179,
                                        0.006737946999085467, 0.0024787521766663585, 0.00091188196555451624,
                                        0.36787944117144233, 0.1353352832366127, 0.049787068367863944,
                                        0.36787944117144233, 0.1353352832366127, 0.049787068367863944]}]

    @params(param_set_joint_weights)
    def test_set_joint_weights(self, res):
        for i in xrange(len(self.skeleton.joint_weights)):
            assert round(self.skeleton.joint_weights[i], 5) == round(res[i], 5)

    param_create_filtered_node_name_map = [{'res': collections.OrderedDict([('Hips', 0), ('Spine', 1), ('Spine_1', 2),
                                                                             ('Neck', 3), ('Head', 4), ('LeftShoulder', 5),
                                                                             ('LeftArm', 6), ('LeftForeArm', 7),
                                                                             ('LeftHand', 8), ('RightShoulder', 9),
                                                                             ('RightArm', 10), ('RightForeArm', 11),
                                                                             ('RightHand', 12), ('LeftUpLeg', 13),
                                                                             ('LeftLeg', 14), ('LeftFoot', 15),
                                                                             ('RightUpLeg', 16), ('RightLeg', 17),
                                                                             ('RightFoot', 18)])}]

    @params(param_create_filtered_node_name_map)
    def test__create_filtered_node_name_map(self, res):
        assert self.skeleton.node_name_map == res