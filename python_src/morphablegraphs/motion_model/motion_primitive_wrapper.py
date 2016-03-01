import numpy as np
from motion_primitive import MotionPrimitive as MGMotionPrimitive
from motion_spline import MotionSpline
try:
    from .extended_mgrd_mixture_model import ExtendedMGRDMixtureModel
    from mgrd import MotionPrimitiveModel as MGRDMotionPrimitiveModel
    from mgrd import QuaternionSplineModel as MGRDQuaternionSplineModel
    from mgrd import TemporalSplineModel as MGRDTemporalSplineModel
    from legacy_temporal_spline_model import LegacyTemporalSplineModel
    has_mgrd = True
except ImportError:
    pass
    has_mgrd = False
from ..utilities import load_json_file
from sklearn.mixture.gmm import GMM


class MotionPrimitiveModelWrapper(object):
    """ Class that wraps the MGRD MotionPrimitiveModel

    """
    SPLINE_DEGREE = 3
    def __init__(self):
        self.motion_primitive = None
        self.use_mgrd_mixture_model = False

    def _load_from_file(self, mgrd_skeleton, file_name):
        data = load_json_file(file_name)
        if data is not None:
            self._initialize_from_json(mgrd_skeleton, data)

    def _initialize_from_json(self, mgrd_skeleton, data):
        if has_mgrd:
            if "semantic_annotation" in data.keys():
                self.motion_primitive = MotionPrimitiveModelWrapper.load_model_from_json(mgrd_skeleton, data, self.use_mgrd_mixture_model)
            else:
                mm = MotionPrimitiveModelWrapper.load_mixture_model(data, self.use_mgrd_mixture_model)

                tspm = LegacyTemporalSplineModel(data)
                #animated_joints = []
                #for node in mgrd_skeleton.skeleton_nodes:
                #    if node.get_type != "end" and  node.get_type != "fixed-joint":
                #        animated_joints.append(node.name)
                animated_joints = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]
                sspm = MGRDQuaternionSplineModel.load_from_json({
                                                            'eigen': np.asarray(data['eigen_vectors_spatial']).T,
                                                            'mean': np.asarray(data['mean_spatial_vector']),
                                                            'n_coeffs': data['n_basis_spatial'],
                                                            'n_dims': data['n_dim_spatial'],
                                                            'knots': np.asarray(data['b_spline_knots_spatial']),
                                                            'degree': self.SPLINE_DEGREE,
                                                            'translation_maxima': np.asarray(data['translation_maxima']),
                                                            'animated_joints': animated_joints
                                                        })
                self.motion_primitive = MGRDMotionPrimitiveModel(mgrd_skeleton, sspm, tspm, mm)
        else:
            self.motion_primitive = MGMotionPrimitive(None)
            self.motion_primitive._initialize_from_json(data)

    def sample_legacy(self, use_time=True):
        return self.motion_primitive.sample(use_time)

    def sample_mgrd(self, use_time=True):
        s_vec = self.sample_low_dimensional_vector()
        quat_spline = self.back_project(s_vec, use_time)
        return quat_spline
    sample = sample_mgrd if has_mgrd else sample_legacy

    def sample_vector_legacy(self):
        return self.motion_primitive.sample_low_dimensional_vector()

    def sample_vector_mgrd(self):
        return self.motion_primitive.get_random_samples(1)[0]
    sample_low_dimensional_vector = sample_vector_mgrd if has_mgrd else sample_vector_legacy

    def back_project_legacy(self, s_vec, use_time_parameters=True):
        return self.motion_primitive.back_project(s_vec, use_time_parameters)

    def back_project_mgrd(self, s_vec, use_time_parameters=True):
        quat_spline = self.motion_primitive.create_spatial_spline(s_vec)
        if use_time_parameters:
            time_spline = self.motion_primitive.create_time_spline(s_vec)
            quat_spline = time_spline.warp(quat_spline)
        return quat_spline
    back_project = back_project_mgrd if has_mgrd else back_project_legacy

    def back_project_time_function_legacy(self, s_vec):
        return self.motion_primitive._back_transform_gamma_to_canonical_time_function(s_vec[self.get_n_spatial_components():])

    def back_project_time_function_mgrd(self, s_vec):
        time_spline = self.motion_primitive.create_time_spline(s_vec, labels=[])
        return np.asarray(time_spline.evaluate_domain(step_size=1.0))#[:,0]
    back_project_time_function = back_project_time_function_mgrd if has_mgrd else back_project_time_function_legacy

    def get_n_canonical_frames_legacy(self):
        return self.motion_primitive.n_canonical_frames

    def get_n_canonical_frames_mgrd(self):
        return self.motion_primitive.time.n_canonical_frames-1
    get_n_canonical_frames = get_n_canonical_frames_mgrd if has_mgrd else get_n_canonical_frames_legacy

    def get_n_spatial_components_legacy(self):
        return self.motion_primitive.get_n_spatial_components()

    def get_n_spatial_components_mgrd(self):
        return self.motion_primitive.spatial.get_n_components()
    get_n_spatial_components = get_n_spatial_components_mgrd if has_mgrd else get_n_spatial_components_legacy

    def get_n_time_components_legacy(self):
        return self.motion_primitive.get_n_time_components()

    def get_n_time_components_mgrd(self):
        return self.motion_primitive.time.get_n_components()
    get_n_time_components = get_n_time_components_mgrd if has_mgrd else get_n_time_components_legacy

    def get_gaussian_mixture_model_legacy(self):
        return self.motion_primitive.gaussian_mixture_model

    def get_gaussian_mixture_model_mgrd(self):
        return self.motion_primitive.mixture
    get_gaussian_mixture_model = get_gaussian_mixture_model_mgrd if has_mgrd else get_gaussian_mixture_model_legacy



    @staticmethod
    def load_model_from_json(skeleton, mm_data, use_mgrd_mixture_model=True):


        # the eigen vectors for spatial spline is stored column major
        mm_data['eigen_vectors_spatial'] = np.ascontiguousarray(np.asarray(mm_data['eigen_vectors_spatial']).transpose())
        mm_data['eigen_vectors_temporal_semantic'] = np.ascontiguousarray(np.asarray(mm_data['eigen_vectors_temporal_semantic']).transpose())
        # TODO: serialize as objects to avoid mapping names
        sspm = MGRDQuaternionSplineModel.load_from_json({
            'eigen': mm_data['eigen_vectors_spatial'],
            'mean': mm_data['mean_spatial_vector'],
            'n_coeffs': mm_data['n_basis_spatial'],
            'n_dims': mm_data['n_dim_spatial'],
            'knots': mm_data['b_spline_knots_spatial'],
            'degree': MotionPrimitiveModelWrapper.SPLINE_DEGREE,
            'translation_maxima': mm_data['translation_maxima'],
            'animated_joints': mm_data['animated_joints']
        })
        tspm = MGRDTemporalSplineModel.load_from_json({
            'eigen': mm_data['eigen_vectors_temporal_semantic'],
            'mean': mm_data['mean_temporal_semantic_vector'],
            'n_coeffs': mm_data['n_basis_temporal_semantic'],
            'n_dims': mm_data['n_dim_temporal_semantic'],
            'knots': mm_data['b_spline_knots_temporal_semantic'],
            'degree': MotionPrimitiveModelWrapper.SPLINE_DEGREE,
            'n_canonical_frames': mm_data['n_canonical_frames'],
            'semantic_annotation': mm_data['semantic_annotation'],
            'frame_time' : mm_data['frame_time'],
            'semantic_labels': mm_data['semantic_annotation']
        })
        # TODO: serialize each cluster separately
        # TODO: store eigen and mean for each cluster to avoid prepare_eigen_vectors()
        mm = MotionPrimitiveModelWrapper.load_mixture_model(mm_data, use_mgrd_mixture_model)

        return MGRDMotionPrimitiveModel(skeleton, sspm, tspm, mm)

    @staticmethod
    def load_mixture_model(data, use_mgrd=True):
        if use_mgrd:
            mm = ExtendedMGRDMixtureModel.load_from_json({
                'covars': data['gmm_covars'],
                'means': data['gmm_means'],
                'weights': data['gmm_weights']
            })
        else:
            n_components = len(np.array(data['gmm_weights']))
            mm = GMM(n_components, covariance_type='full')
            mm.weights_ = np.array(data['gmm_weights'])
            mm.means_ = np.array(data['gmm_means'])
            mm.converged_ = True
            mm.covars_ = np.array(data['gmm_covars'])
            mm.n_dims = len(mm.covars_[0])
        return mm