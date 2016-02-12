import numpy as np
from motion_primitive import MotionPrimitive as MGMotionPrimitive
from motion_spline import MotionSpline
from mgrd import MotionPrimitiveModel as MGRDMotionPrimitive
from mgrd import MixtureModel as MGRDMixtureModel
from mgrd import QuaternionSplineModel as MGRDQuaternionSplineModel
from ..utilities import load_json_file
from legacy_temporal_spline_model import LegacyTemporalSplineModel


class MixtureModelWrapper(object):
    def __init__(self, mgrd_mixture_model):
        self.mixture_model = mgrd_mixture_model

    def sample(self):
        return self.mixture_model.sample(1)[0]


class MotionPrimitiveModelWrapper(object):
    """ Class that wraps the MGRD MotionPrimitiveModel

    """
    def __init__(self):
        self.motion_primitive = None
        self.use_mgrd = False
        self.is_mgrd = True

    def _load_from_file(self, mgrd_skeleton, file_name):
        data = load_json_file(file_name)
        if data is not None:
            self._initialize_from_json(mgrd_skeleton, data)

    def _initialize_from_json(self, mgrd_skeleton, data):
        if "semantic_annotation" in data.keys():
            self.motion_primitive = MGRDMotionPrimitive.load_from_dict(mgrd_skeleton, data)
            self.use_mgrd = True
        else:
            mm = MGRDMixtureModel.load_from_json({'covars': data['gmm_covars'], 'means': data['gmm_means'],
                                                 'weights': data['gmm_weights'] })
            tspm = LegacyTemporalSplineModel(data)
            #animated_joints = []
            #for node in mgrd_skeleton.skeleton_nodes:
            #    if node.get_type != "end" and  node.get_type != "fixed-joint":
            #        animated_joints.append(node.name)
            animated_joints = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]
            #print np.asarray(data['eigen_vectors_spatial']).shape, np.asarray(data['mean_spatial_vector']).shape
            sspm = MGRDQuaternionSplineModel.load_from_json({
                                                        'eigen': np.asarray(data['eigen_vectors_spatial']).T,
                                                        'mean': np.asarray(data['mean_spatial_vector']),
                                                        'n_coeffs': data['n_basis_spatial'],
                                                        'n_dims': data['n_dim_spatial'],
                                                        'knots': np.asarray(data['b_spline_knots_spatial']),
                                                        'degree': 3,
                                                        'translation_maxima': np.asarray(data['translation_maxima']),
                                                        'animated_joints': animated_joints
                                                    })
            self.motion_primitive = MGRDMotionPrimitive(mgrd_skeleton, sspm, tspm, mm)
            #self.motion_primitive = MGMotionPrimitive(None)
            #self.motion_primitive._initialize_from_json(data)
            self.use_mgrd = False

    def sample(self, use_time=True):
        if self.is_mgrd:
            s_vec = self.sample_low_dimensional_vector()
            quat_spline = self.back_project(s_vec)
            if use_time:
                time_spline = self.motion_primitive.create_time_spline(s_vec)
                quat_spline = time_spline.warp(quat_spline, time_spline.model.frame_time)
            return quat_spline
        else:
            return self.motion_primitive.sample(use_time)

    def sample_low_dimensional_vector(self):
        if self.is_mgrd:
            return self.motion_primitive.get_random_samples(1)[0]
        else:
            return self.motion_primitive.sample_low_dimensional_vector()

    def back_project(self, s_vec, use_time_parameters=True):
        if self.is_mgrd:
            return self.motion_primitive.create_spatial_spline(s_vec)
        else:
            return self.motion_primitive.back_project(s_vec, use_time_parameters)

    def back_project_time_function(self, parameters):
        if self.is_mgrd:
            return np.asarray(self.motion_primitive.create_time_spline(parameters, labels=[]).evaluate_domain(step_size=1.0))#[:,0]
        else:

            time_parameters = parameters[self.get_n_spatial_components():]
            #print time_parameters, step.n_spatial_components, step.n_time_components, len(step.parameters)
            return self.motion_primitive.back_project_time_function(time_parameters)

    def get_n_canonical_frames(self):
        if self.is_mgrd:
            return self.motion_primitive.time.n_canonical_frames-1
        else:
            return self.motion_primitive.n_canonical_frames

    def get_n_spatial_components(self):
        if self.is_mgrd:
            return self.motion_primitive.spatial.get_n_components()
        else:
            return self.motion_primitive.get_n_spatial_components()

    def get_n_time_components(self):
        if self.is_mgrd:
            return self.motion_primitive.time.get_n_components()
        else:
            return self.motion_primitive.get_n_time_components()

    def get_gaussian_mixture_model(self):
        if self.is_mgrd:
            return MixtureModelWrapper(self.motion_primitive.mixture)
        else:
            return self.motion_primitive.gaussian_mixture_model