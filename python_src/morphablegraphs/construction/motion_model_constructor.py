import numpy as np
from copy import copy
from fpca.fpca_spatial_data import FPCASpatialData
from fpca.fpca_time_semantic import FPCATimeSemantic
from preprocessing.motion_dtw import MotionDynamicTimeWarping
from motion_primitive.statistical_model_trainer import GMMTrainer
from ..animation_data.utils import pose_orientation_quat, get_rotation_angle
from ..external.transformations import quaternion_from_euler
from dtw import run_dtw, get_warping_function, find_reference_motion, warp_motion
from ..animation_data import SkeletonBuilder
from ..utilities.io_helper_functions import export_frames_to_bvh_file
from ..motion_model.motion_spline import MotionSpline
from utils import convert_poses_to_point_clouds, rotate_frames, align_quaternion_frames, get_cubic_b_spline_knots,\
                  normalize_root_translation, scale_root_translation_in_fpca_data, gen_gaussian_eigen, BSPLINE_DEGREE


class MotionModel(object):
    def __init__(self, bvh_reader, animated_joints=None):
        self._skeleton = SkeletonBuilder().load_from_bvh(bvh_reader, animated_joints, add_tool_joints=False)
        self._input_motions = []
        self._aligned_frames = []
        self._temporal_data = []
        self._spatial_fpca_data = None
        self._temporal_fpca_data = None
        self._gmm_data = None

    def back_project_sample(self, alpha):
        coeffs = np.dot(self._spatial_fpca_data["eigenvectors"].T, alpha)
        coeffs += self._spatial_fpca_data["mean"]
        coeffs = coeffs.reshape((self._spatial_fpca_data["n_basis"], self._spatial_fpca_data["n_dim"]))
        # undo the scaling on the translation
        translation_maxima = self._spatial_fpca_data["scale_vec"]
        coeffs[:, 0] *= translation_maxima[0]
        coeffs[:, 1] *= translation_maxima[1]
        coeffs[:, 2] *= translation_maxima[2]
        return coeffs

    def export_sample(self, name, alpha):
        spatial_coeffs = self.back_project_sample(alpha)
        self.export_coeffs(name, spatial_coeffs)

    def export_coeffs(self, name, spatial_coeffs):
        n_frames = len(self._aligned_frames[0])
        time_function = np.arange(0, n_frames)

        spatial_knots = get_cubic_b_spline_knots(self._spatial_fpca_data["n_basis"], n_frames).tolist()
        spline = MotionSpline(0, spatial_coeffs, time_function, spatial_knots, None)
        frames = spline.get_motion_vector()
        self._export_frames(name, frames)

    def _export_frames(self, name, frames):
        frames = np.array([self._skeleton.generate_complete_frame_vector_from_reference(f) for f in frames])
        export_frames_to_bvh_file(".", self._skeleton, frames, name, time_stamp=False)


class MotionModelConstructor(MotionModel):
    def __init__(self, bvh_reader, config, animated_joints=None):
        super(MotionModelConstructor, self).__init__(bvh_reader, animated_joints)
        self._bvh_reader = bvh_reader
        self.config = config
        self.ref_orientation = [0,-1]
        self._motion_dtw = MotionDynamicTimeWarping()
        self._motion_dtw.ref_bvhreader = self._bvh_reader

    def set_motions(self, motions):
        """ Set the input data.
        Args:
        -----
             motions (List): input motion data in quaternion format.
        """
        self._input_motions = motions

    def construct_model(self, name, version=1):
        """ Runs the construction pipeline
        Args:
        -----
             name (string): name of the motion primitive
             version (int): format supported values are 1, 2 and 3
        """

        self._align_frames()
        self.run_dimension_reduction()
        self.learn_statistical_model()
        model_data = self.convert_motion_model_to_json(name, version)
        return model_data

    def _align_frames(self):
        aligned_frames = self._align_frames_spatially(self._input_motions)
        self._aligned_frames, self._temporal_data = self._align_frames_temporally(aligned_frames)
        #self._export_aligned_frames()

    def _export_aligned_frames(self):
        for idx, frames in enumerate(self._aligned_frames):
            print np.array(frames).shape
            name = "aligned"+str(idx)
            self._export_frames(name, frames)

    def _align_frames_spatially(self, input_motions):
        print "run spatial alignment", self.ref_orientation
        aligned_frames = []
        frame_idx = 0
        for input_m in input_motions:
            ma = input_m[:]

            # align orientation to reference orientation
            m_orientation = pose_orientation_quat(ma[frame_idx])
            rot_angle = get_rotation_angle(self.ref_orientation, m_orientation)
            e = np.deg2rad([0, rot_angle, 0])
            q = quaternion_from_euler(*e)
            ma = rotate_frames(ma, q)

            # normalize position
            delta = copy(ma[0, :3])
            for f in ma:
                f[:3] -= delta + self._skeleton.nodes[self._skeleton.root].offset
            aligned_frames.append(ma)
        return aligned_frames

    def _align_frames_temporally_rpy2(self, input_motions):
        """
         run rpy2 dtw
        """
        self._motion_dtw.aligned_motions = dict(enumerate(input_motions))
        print "run temporal alignment", len(self._motion_dtw.aligned_motions)
        self._motion_dtw.dtw()
        aligned_frames = []
        temporal_data = []
        for key, motion in self._motion_dtw.warped_motions:
            aligned_frames.append(motion['frames'])
            temporal_data.append(motion['warping_index'])
        return aligned_frames, temporal_data

    def _align_frames_temporally(self, input_motions):
        print "run temporal alignment"
        point_clouds = convert_poses_to_point_clouds(self._skeleton, input_motions, normalize=False)
        ref_p = find_reference_motion(point_clouds)
        temp = zip(input_motions, point_clouds)
        warped_frames = []
        warping_functions = []
        for idx, data in enumerate(temp):
            m, p = data
            path, D = run_dtw(p, ref_p)
            warping_function = get_warping_function(path)
            #print path, warping_function
            warped_frames.append(warp_motion(m, warping_function))
            warping_functions.append(warping_function)
        return warped_frames, warping_functions

    def run_dimension_reduction(self):
        self.run_spatial_dimension_reduction()
        self.run_temporal_dimension_reduction()

    def run_spatial_dimension_reduction(self):
        scaled_quat_frames, scale_vec = normalize_root_translation(self._aligned_frames)
        smoothed_quat_frames = align_quaternion_frames(self._skeleton, scaled_quat_frames)
        fpca_input = dict(enumerate(smoothed_quat_frames))
        fpca_spatial = FPCASpatialData(self.config["n_basis_functions_spatial"],
                                       self.config["n_components"],
                                       self.config["fraction"])
        fpca_spatial.fit_motion_dictionary(fpca_input)

        result = dict()
        result['parameters'] = fpca_spatial.fpcaobj.low_vecs
        result['file_order'] = fpca_spatial.fileorder
        result['n_basis'] = fpca_spatial.fpcaobj.n_basis
        eigenvectors = fpca_spatial.fpcaobj.eigenvectors
        mean = fpca_spatial.fpcaobj.centerobj.mean
        data = fpca_spatial.fpcaobj.functional_data
        result["n_coeffs"] = len(data[0])
        result['n_dim'] = len(data[0][0])
        result["scale_vec"] = [1,1,1]
        mean, eigenvectors = scale_root_translation_in_fpca_data(mean,
                                                                 eigenvectors,
                                                                 scale_vec,
                                                                 result['n_coeffs'],
                                                                 result['n_dim'])
        result['mean'] = mean
        result['eigenvectors'] = eigenvectors
        self._spatial_fpca_data = result

    def run_temporal_dimension_reduction(self):
        fpca_temporal = FPCATimeSemantic(self.config["n_basis_functions_temporal"],
                                          n_components_temporal=self.config["npc_temporal"],
                                          precision_temporal=self.config["precision_temporal"])
        print np.array(self._temporal_data).shape
        fpca_temporal.temporal_semantic_data = self._temporal_data
        fpca_temporal.semantic_annotation_list = []
        fpca_temporal.functional_pca()
        result = dict()
        result['eigenvectors'] = fpca_temporal.eigenvectors
        result['mean'] = fpca_temporal.mean_vec
        result['parameters'] = fpca_temporal.lowVs
        result['n_basis'] = fpca_temporal.n_basis
        result['n_dim'] = len(fpca_temporal.semantic_annotation_list)+1
        result['semantic_annotation'] = fpca_temporal.semantic_annotation_list
        self._temporal_fpca_data = result

    def learn_statistical_model(self):
        if self._temporal_fpca_data is not None:

            spatial_parameters = self._spatial_fpca_data["parameters"]
            temporal_parameters = self._temporal_fpca_data["parameters"]
            print spatial_parameters, temporal_parameters
            motion_parameters = np.concatenate((spatial_parameters, temporal_parameters,),axis=1)
        else:
            motion_parameters = self._spatial_fpca_data["parameters"]
        trainer = GMMTrainer()
        trainer.fit(motion_parameters)
        self._gmm_data = trainer.convert_model_to_json()

    def convert_motion_model_to_json(self, name="", version=1):
        weights = self._gmm_data['gmm_weights']
        means = self._gmm_data['gmm_means']
        covars = self._gmm_data['gmm_covars']

        n_frames = len(self._aligned_frames[0])

        mean_motion = self._spatial_fpca_data["mean"].tolist()
        spatial_eigenvectors = self._spatial_fpca_data["eigenvectors"].tolist()
        scale_vec = self._spatial_fpca_data["scale_vec"]
        n_dim_spatial = self._spatial_fpca_data["n_dim"]
        n_basis_spatial = self._spatial_fpca_data["n_basis"]
        spatial_knots = get_cubic_b_spline_knots(n_basis_spatial, n_frames).tolist()

        if self._temporal_fpca_data is not None:
            temporal_mean = self._temporal_fpca_data["mean"].tolist()
            temporal_eigenvectors = self._temporal_fpca_data["eigenvectors"].tolist()
            n_basis_temporal = self._temporal_fpca_data["n_basis"]
            temporal_knots = get_cubic_b_spline_knots(n_basis_temporal, n_frames).tolist()
            semantic_label = self._temporal_fpca_data["semantic_annotation"]
        else:
            temporal_mean = []
            temporal_eigenvectors = []
            n_basis_temporal = 0
            temporal_knots = []
            semantic_label = dict()

        if version == 1:
            data = {'name': name,
                    'gmm_weights': weights,
                    'gmm_means': means,
                    'gmm_covars': covars,
                    'eigen_vectors_spatial': spatial_eigenvectors,
                    'mean_spatial_vector': mean_motion,
                    'n_canonical_frames': n_frames,
                    'translation_maxima': scale_vec,
                    'n_basis_spatial': n_basis_spatial,
                    'npc_spatial': len(spatial_eigenvectors),
                    'eigen_vectors_temporal_semantic': temporal_eigenvectors,
                    'mean_temporal_semantic_vector': temporal_mean,
                    'n_dim_spatial': n_dim_spatial,
                    'n_basis_temporal_semantic': n_basis_temporal,
                    'b_spline_knots_spatial': spatial_knots,
                    'b_spline_knots_temporal_semantic':temporal_knots,
                    'npc_temporal_semantic': self.config["npc_temporal"],
                    'semantic_annotation': {},
                    'n_dim_temporal_semantic': 1}
        elif version == 2:
            data = {'name': name,
                    'gmm_weights': weights,
                    'gmm_means': means,
                    'gmm_covars': covars,
                    'eigen_vectors_spatial': spatial_eigenvectors,
                    'mean_spatial_vector': mean_motion,
                    'n_canonical_frames': n_frames,
                    'translation_maxima': scale_vec,
                    'n_basis_spatial': n_basis_spatial,
                    'eigen_vectors_time': temporal_eigenvectors,
                    'mean_time_vector': temporal_mean,
                    'n_dim_spatial': n_dim_spatial,
                    'n_basis_time': n_basis_temporal,
                    'b_spline_knots_spatial': spatial_knots,
                    'b_spline_knots_time': temporal_knots}
                    #'semantic_label': semantic_label
        else:
            data = dict()
            data['sspm'] = dict()
            data['tspm'] = dict()
            data['gmm'] = dict()
            data['sspm']['eigen'] = spatial_eigenvectors
            data['sspm']['mean'] = mean_motion
            data['sspm']['n_coeffs'] = n_basis_spatial
            data['sspm']['n_dims'] = n_dim_spatial
            data['sspm']['knots'] = spatial_knots
            data['sspm']['animated_joints'] = self._skeleton.animated_joints
            data['sspm']['degree'] = BSPLINE_DEGREE
            data['gmm']['covars'] = covars
            data['gmm']['means'] = means
            data['gmm']['weights'] = weights
            data['gmm']['eigen'] = gen_gaussian_eigen(covars).tolist()
            data['tspm']['eigen'] = temporal_eigenvectors
            data['tspm']['mean'] = temporal_mean
            data['tspm']['n_coeffs'] = n_basis_temporal
            data['tspm']['n_dims'] = 1
            data['tspm']['knots'] = temporal_knots
            data['tspm']['degree'] = BSPLINE_DEGREE
            data['tspm']['semantic_labels'] = semantic_label
            data['tspm']['frame_time'] = self._skeleton.frame_time
        return data
