# encoding: UTF-8
import mgrd as mgrd
from ...utilities import load_json_file
from ...animation_data import BVHReader, convert_euler_frames_to_quaternion_frames
from ..fpca import FunctionalData
import numpy as np


class QuatSplineConstructor(object):

    def __init__(self, semantic_motion_primitive_file, skeleton_jsonfile):
        self.mm_data = load_json_file(semantic_motion_primitive_file)
        self.skeleton = mgrd.SkeletonJSONLoader(skeleton_jsonfile).load()
        self.motion_primitive = mgrd.MotionPrimitiveModel.load_from_json(self.skeleton,
                                                                         self.mm_data)

    def create_quat_spline_from_bvhfile(self, bvhfile, n_basis, degree=3):
        bvhreader = BVHReader(bvhfile)
        quat_frames = convert_euler_frames_to_quaternion_frames(bvhreader, bvhreader.frames)
        fd = FunctionalData()
        functional_coeffs = fd.convert_motion_to_functional_data(quat_frames, n_basis, degree)
        functional_coeffs = mgrd.asarray(functional_coeffs.tolist())
        knots = fd.knots
        sspm = mgrd.QuaternionSplineModel.load_from_json(self.skeleton, self.mm_data['sspm'])
        sspm.motion_primitive = self.motion_primitive
        coeffs_structure = mgrd.CoeffStructure(len(self.mm_data['sspm']['animated_joints']),
                                               mgrd.CoeffStructure.LEN_QUATERNION,
                                               mgrd.CoeffStructure.LEN_ROOT_POSITION)
        quat_spline = mgrd.QuatSpline(functional_coeffs,
                                      knots,
                                      coeffs_structure,
                                      degree,
                                      sspm)
        return quat_spline

    def create_quat_spline_from_functional_data(self, functional_datamat, knots, degree=3):
        functional_coeffs = mgrd.asarray(functional_datamat)
        knots = mgrd.asarray(knots)
        sspm = mgrd.QuaternionSplineModel.load_from_json(self.skeleton, self.mm_data['sspm'])
        sspm.motion_primitive = self.motion_primitive
        coeffs_structure = mgrd.CoeffStructure(len(self.mm_data['sspm']['animated_joints']),
                                               mgrd.CoeffStructure.LEN_QUATERNION,
                                               mgrd.CoeffStructure.LEN_ROOT_POSITION)
        quat_spline = mgrd.QuatSpline(functional_coeffs,
                                      knots,
                                      coeffs_structure,
                                      degree,
                                      sspm)
        return quat_spline


if __name__ == "__main__":
    from morphablegraphs.utilities import get_semantic_motion_primitive_path
    from ctypes import *
    import numpy as np
    elementary_action = 'pickBoth'
    motion_primitive = 'reach'
    datarepo = r'C:\repo'
    skeleton_file =r'../../../mgrd/data/skeleton.json'
    semantic_motion_primitive_file = get_semantic_motion_primitive_path(elementary_action,
                                                                        motion_primitive,
                                                                        datarepo)
    quat_spline_constructor = QuatSplineConstructor(semantic_motion_primitive_file,
                                                    skeleton_file)
    test_file = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_pickBoth\first\pickBoth_001_1_reach_502_604.bvh'
    quat_spline = quat_spline_constructor.create_quat_spline_from_bvhfile(test_file,
                                                                          7)
    cartesian_spline = quat_spline.to_cartesian()
    print((cartesian_spline.coeffs.shape))

