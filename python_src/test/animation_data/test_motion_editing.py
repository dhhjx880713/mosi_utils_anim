__author__ = 'hadu01'

import os
from ...morphablegraphs.animation_data.bvh import BVHReader
from ...morphablegraphs.animation_data.skeleton import Skeleton
from ...morphablegraphs.animation_data.motion_editing import *
from ..libtest import params, pytest_generate_tests
import numpy as np
import copy
ROOT_DIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
TEST_DATA_PATH = ROOT_DIR + '../test_data/animation_data'
TEST_RESULT_PATH = ROOT_DIR + '../test_output/animation_data'


class TestMotionEditing(object):

    def setup_class(self):
        leftStance_file = TEST_DATA_PATH + os.sep + 'walk_001_1_leftStance_43_86.bvh'
        rightStance_file = TEST_DATA_PATH + os.sep + 'walk_001_1_rightStance_86_128.bvh'
        self.leftStance_bvhreader = BVHReader(leftStance_file)
        self.rightStance_bvhreader = BVHReader(rightStance_file)
        self.skeleton = Skeleton(self.leftStance_bvhreader)
        self.quat_frames_rightStance = convert_euler_frames_to_quaternion_frames(self.rightStance_bvhreader,
                                                                                 self.rightStance_bvhreader.frames)
        self.euler_frames_rightStance = convert_quaternion_frames_to_euler_frames(self.quat_frames_rightStance)
        self.quat_frames_leftStance = convert_euler_frames_to_quaternion_frames(self.leftStance_bvhreader,
                                                                                self.leftStance_bvhreader.frames)

    param_get_cartesian_coordinates_from_quaternion = [{'test_joint': 'LeftHand',
                                                        'res': [-23.602410903031142, 78.14988852117129, -2.39092472693668]}]

    @params(param_get_cartesian_coordinates_from_quaternion)
    def test_get_cartesian_coordinates_from_quaternion(self, test_joint, res):
        pos = get_cartesian_coordinates_from_quaternion(self.skeleton,
                                                        test_joint,
                                                        np.asarray(self.quat_frames_rightStance[0]))
        for i in xrange(len(pos)):
            assert round(pos[i], 5) == round(res[i], 5)

    param_get_cartesian_coordinates_from_euler_full_skeleton = [
        {
            'test_joint': 'LeftHand',
            'res': [
                -23.602410903031142,
                78.14988852117129,
                -2.39092472693668]}]

    @params(param_get_cartesian_coordinates_from_euler_full_skeleton)
    def test_get_cartesian_coordinates_from_euler_full_skeleton(self, test_joint, res):
        pos = get_cartesian_coordinates_from_euler_full_skeleton(self.leftStance_bvhreader,
                                                                 self.skeleton,
                                                                 test_joint,
                                                                 self.rightStance_bvhreader.frames[0])
        for i in xrange(len(pos)):
            assert round(pos[i], 5) == round(res[i], 5)

    param_get_cartesian_coordinates_from_euler = [{'test_joint': 'LeftHand',
                                                   'res': [-23.602410903031142, 78.14988852117129, -2.39092472693668]}]

    @params(param_get_cartesian_coordinates_from_euler)
    def test_get_cartesian_coordinates_from_euler(self, test_joint, res):
        pos = get_cartesian_coordinates_from_euler(self.skeleton,
                                                   test_joint,
                                                   self.euler_frames_rightStance[0])
        for i in xrange(len(pos)):
            assert round(pos[i], 5) == round(res[i], 5)

    param_euler_to_quaternion = [{'euler_angles': [0, 0, 60],
                                  'res': [0.86602540378443871, 0.0, 0.0, 0.5]}]

    @params(param_euler_to_quaternion)
    def test_euler_to_quatenrion(self, euler_angles, res):
        quat = euler_to_quaternion(euler_angles)
        for i in xrange(len(quat)):
            assert round(quat[i], 5) == round(res[i], 5)

    param_quaternion_to_euler = [{'quat': [0.86602540378443871, 0.0, 0.0, 0.5],
                                  'res': [0, 0, 60]}]
    @params(param_quaternion_to_euler)
    def test_quaternion_to_euler(self, quat, res):
        euler_angles = quaternion_to_euler(quat)
        for i in xrange(len(euler_angles)):
            assert round(euler_angles[i], 3) == round(res[i], 3)

    param_convert_euler_frames_to_quaternion_frames = [
        {'test_joint': 'RightHand'}]

    @params(param_convert_euler_frames_to_quaternion_frames)
    def test_convert_euler_frames_to_quaternion_frames(self, test_joint):
        expected_point = get_cartesian_coordinates_from_euler_full_skeleton(self.rightStance_bvhreader,
                                                                            self.skeleton,
                                                                            test_joint,
                                                                            self.rightStance_bvhreader.frames[0])
        res_point = get_cartesian_coordinates_from_quaternion(self.skeleton,
                                                              test_joint,
                                                              np.asarray(self.quat_frames_rightStance[0]))
        for i in xrange(len(expected_point)):
            assert round(res_point[i], 5) == round(expected_point[i], 5)

    param_convert_quaternion_frames_to_euler_frames = [{'test_joint': 'RightHand'}]

    @params(param_convert_quaternion_frames_to_euler_frames)
    def test_covnert_quaternion_frames_to_euler_frames(self, test_joint):
        expected_pos = get_cartesian_coordinates_from_quaternion(self.skeleton,
                                                                 test_joint,
                                                                 np.asarray(self.quat_frames_rightStance[0]))
        res_pos = get_cartesian_coordinates_from_euler(self.skeleton,
                                                       test_joint,
                                                       self.euler_frames_rightStance[0])
        for i in xrange(len(expected_pos)):
            assert round(res_pos[i], 5) == round(expected_pos[i], 5)

    param_euler_substraction = [{'angles': [30, 60],
                                 'res':-30}]

    @params(param_euler_substraction)
    def test_euler_substraction(self, angles, res):
        theta = euler_substraction(angles[0], angles[1])
        assert theta == res

    param_convert_euler_frame_to_cartesian_frame = [{'res': [[-1.15780000332, 90.0, 4.68537284741],
                                                             [-1.075011620146879, 105.31663141876564, 4.7243897688313785],
                                                             [-0.8253085123299746, 120.61976808868623, 5.341378281319706],
                                                             [-0.5462378947938223, 140.57824344585322, 4.566207144140793],
                                                             [-0.7093538348577836, 147.06823911374445, 2.1516590323212914],
                                                             [-7.6862387856948695, 134.37353991650141, 6.16165759695545],
                                                             [-20.796853946476467, 129.1952796769343, 5.584597318150094],
                                                             [-30.06074726937313, 102.27597596375247, 8.10634277308117],
                                                             [-23.60241090303116, 78.14988852117133, -2.3909247269366403],
                                                             [6.873954082072464, 134.66310561959193, 5.967250397970924],
                                                             [6.541708369456233, 147.88198475789756, 1.0492777789519985],
                                                             [5.868641635564988, 174.66099408632982, -8.913627048447927],
                                                             [5.230639631152401, 200.04559640371363, -18.35775459855264],
                                                             [-10.061255991446004, 78.02612547522367, 4.993760042865179],
                                                             [-9.845420430761052, 117.92898055363824, 5.063644603303839],
                                                             [-9.630854495904677, 157.5972394286102, 5.133117603869033],
                                                             [7.611974172257867, 77.9303801243421, 5.079810923448956],
                                                             [7.827809732942817, 117.83323520275668, 5.14969548388753],
                                                             [8.042377049088682, 157.50149406903287, 5.219169183542287]]}]

    @params(param_convert_euler_frame_to_cartesian_frame)
    def test_convert_euler_frame_to_cartesian_frame(self, res):

        cartesian_frame = convert_euler_frame_to_cartesian_frame(self.skeleton,
                                                                 self.rightStance_bvhreader.frames[0])
        assert cartesian_frame == res

    param_find_aligning_transformation = [{'res': {'theta': -0.107528789421,
                                                   'offset_x': -5.17677664791,
                                                   'offset_z': -56.4992063097}}]

    @params(param_find_aligning_transformation)
    def test_find_aligning_transformation(self, res):
        theta, offset_x, offset_z = find_aligning_transformation(self.skeleton,
                                                                 self.rightStance_bvhreader.frames,
                                                                 self.leftStance_bvhreader.frames)
        assert round(theta, 5) == round(res['theta'], 5) and \
               round(offset_x, 5) == round(res['offset_x'], 5) and \
               round(offset_z, 5) == round(res['offset_z'], 5)

    param_pose_orientation_euler = [{'res': [0.00488134, -0.99998809]}]

    @params(param_pose_orientation_euler)
    def test_pose_orientation_euler(self, res):
        orientation = pose_orientation_euler(self.rightStance_bvhreader.frames[0])
        for i in xrange(len(orientation)):
            assert round(res[i], 5) == round(orientation[i], 5)

    param_transform_euler_frames = [{'angles':[0, 60, 0],
                                    'offset': [100, 0, -100],
                                    'res':[86.12819299642089, 78.14988852117135, -80.75517493088452]}]

    @params(param_transform_euler_frames)
    def test_transform_euler_frames(self, angles, offset, res):
        frames = copy.deepcopy(self.rightStance_bvhreader.frames)
        transformed_frames = transform_euler_frames(frames,
                                                    angles,
                                                    offset)
        pos = get_cartesian_coordinates_from_euler_full_skeleton(self.rightStance_bvhreader,
                                                                 self.skeleton,
                                                                 'LeftHand',
                                                                 transformed_frames[0])
        for i in xrange(len(pos)):
            assert round(pos[i], 5) == round(res[i], 5)

    param_pose_orientation_quat = [{'res': [0.00488134, -0.99998809]}]

    @params(param_pose_orientation_quat)
    def test_pose_orientation_quat(self, res):
        orientation = pose_orientation_quat(self.quat_frames_rightStance[0])
        for i in xrange(len(orientation)):
            assert round(res[i], 5) == round(orientation[i], 5)

    param_extract_root_position = [{'res': [[-1.15780000332, 4.68537284741], [-1.32248545151, 2.8568899515499999],
                                            [-1.4592689838299999, 1.0244997242],
                                            [-1.5843856173199999, -0.80991457646200005],
                                            [-1.72196133696, -2.61564869486], [-1.88104476269, -4.3438243190800003],
                                            [-2.0597970608099998, -6.0307105841400004],
                                            [-2.2858608567999998, -7.6042393640399997],
                                            [-2.5293379749499998, -9.1572945242500001],
                                            [-2.7539419630799999, -10.6588000464],
                                            [-2.95641377377, -12.1175244551], [-2.95641377377, -12.1175244551],
                                            [-3.1255419821000001, -13.561962852900001],
                                            [-3.1255419821000001, -13.561962852900001],
                                            [-3.2967742374500002, -14.9910231899],
                                            [-3.2967742374500002, -14.9910231899],
                                            [-3.4654834407999999, -16.412164667799999],
                                            [-3.6291238365099998, -17.786423785699998],
                                            [-3.6291238365099998, -17.786423785699998],
                                            [-3.8087708842999999, -19.166269966000002],
                                            [-4.0010344199499999, -20.5247753419],
                                            [-4.1747319207300002, -21.8677750698],
                                            [-4.3566764645099996, -23.230543665300001],
                                            [-4.5205819679000001, -24.569811455300002],
                                            [-4.6677641953800002, -25.895449924699999],
                                            [-4.7933560660300003, -27.230664619399999],
                                            [-4.9054974425199998, -28.5712152507],
                                            [-5.0339967583299998, -29.850336857799999],
                                            [-5.0339967583299998, -29.850336857799999],
                                            [-5.14389480061, -31.147961922099999],
                                            [-5.2440370891499999, -32.452813727600002],
                                            [-5.2440370891499999, -32.452813727600002],
                                            [-5.3339874863599999, -33.7539428607],
                                            [-5.4076912721500001, -35.052451066000003],
                                            [-5.4076912721500001, -35.052451066000003],
                                            [-5.4752996346899998, -36.368166364899999],
                                            [-5.4752996346899998, -36.368166364899999],
                                            [-5.5231254467499999, -37.697362781800003],
                                            [-5.5471115866199998, -39.084197977199999],
                                            [-5.5724068612100002, -40.482049204699997],
                                            [-5.5889218168900001, -41.9280856305],
                                            [-5.5943798671899998, -43.410428766099997],
                                            [-5.5919102687400004, -44.951999854999997],
                                            [-5.5835626183800002, -46.528753379000001],
                                            [-5.5832036827699998, -48.108276801599999],
                                            [-5.5727347685700002, -49.742094557500003],
                                            [-5.53803242625, -51.4145109261], [-5.5094381934500003, -53.117831721100004],
                                            [-5.4671134753699997, -54.9015033437],
                                            [-5.3995719534799997, -56.743839424199997]]}]

    @params(param_extract_root_position)
    def test_extract_root_position(self, res):
        root_pos = extract_root_positions(self.quat_frames_rightStance)
        assert root_pos == res

    param_calculate_pose_distance = [{'res': 7.82180380808}]

    @params(param_calculate_pose_distance)
    def test_calculate_pose_distance(self, res):
        err = calculate_pose_distance(self.skeleton,
                                      self.leftStance_bvhreader.frames,
                                      self.rightStance_bvhreader.frames)
        assert round(err, 5) == round(res, 5)

    param_fast_quat_frames_alignment = [{'test_joint': 'Hips',
                                         'res': [-3.5570284326300001, 93.903405715550008, -53.764512292799999]}]

    @params(param_fast_quat_frames_alignment)
    def test_fast_quat_frames_alignment(self, test_joint, res):
        concatenated_frames = fast_quat_frames_alignment(self.quat_frames_leftStance,
                                                         self.quat_frames_rightStance)
        index = len(self.quat_frames_leftStance)
        pos = get_cartesian_coordinates_from_quaternion(self.skeleton,
                                                        test_joint,
                                                        concatenated_frames[index])
        assert pos == res

    param_point_to_euler_angle = [{'point': [-1, -1],
                                   'res': -135}]

    @params(param_point_to_euler_angle)
    def test_point_to_euler_angle(self, point, res):
        angle = point_to_euler_angle(point)
        assert res == angle

    param_get_rotation_angle = [{'points': [[-0.5, -np.sqrt(3)/2.0],
                                            [1, 1]],
                                 'res': 165}]

    @params(param_get_rotation_angle)
    def test_get_rotation_angle(self, points, res):
        angle = get_rotation_angle(points[0], points[1])
        assert angle == res