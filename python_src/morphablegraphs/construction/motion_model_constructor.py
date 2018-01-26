import numpy as np
from copy import copy
from .fpca.fpca_spatial_data import FPCASpatialData
from .fpca.fpca_time_semantic import FPCATimeSemantic
from .motion_primitive.statistical_model_trainer import GMMTrainer
from ..animation_data.utils import pose_orientation_quat, get_rotation_angle
from ..external.transformations import quaternion_from_euler
from .dtw import get_warping_function, find_optimal_dtw_async, warp_motion
from ..utilities.io_helper_functions import export_frames_to_bvh_file
from ..motion_model.motion_spline import MotionSpline
from .utils import convert_poses_to_point_clouds, rotate_frames, align_quaternion_frames, get_cubic_b_spline_knots,\
                  normalize_root_translation, scale_root_translation_in_fpca_data, gen_gaussian_eigen, BSPLINE_DEGREE


class MotionModel(object):
    def __init__(self, skeleton):
        self._skeleton = skeleton
        self._input_motions = []
        self._aligned_frames = []
        self._temporal_data = []
        self._spatial_fpca_data = None
        self._temporal_fpca_data = None
        self._gmm_data = None
        self._keyframes = dict()

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
        frames = np.array([self._skeleton.add_fixed_joint_parameters_to_frame(f) for f in frames])
        export_frames_to_bvh_file(".", self._skeleton, frames, name, time_stamp=False)


class MotionModelConstructor(MotionModel):
    def __init__(self, skeleton, config):
        super(MotionModelConstructor, self).__init__(skeleton)
        self.skeleton = skeleton
        self.config = config
        self.ref_orientation = [0,-1]  # look into -z direction in 2d
        self.dtw_sections = None

    def set_motions(self, motions):
        """ Set the input data.
        Args:
        -----
             motions (List): input motion data in quaternion format.
        """
        self._input_motions = motions

    def set_dtw_sections(self, dtw_sections):
        """ Set sections input data.
        Args:
        -----
             motions (List): input motion data in quaternion format.
        """
        self._dtw_sections = dtw_sections
        self._keyframes = dict()

    def construct_model(self, name, version=1, save_skeleton=False):
        """ Runs the construction pipeline
        Args:
        -----
             name (string): name of the motion primitive
             version (int): format supported values are 1, 2 and 3
        """

        self._align_frames()
        self.run_dimension_reduction()
        self.learn_statistical_model()
        model_data = self.convert_motion_model_to_json(name, version, save_skeleton)
        return model_data

    def _align_frames(self):
        aligned_frames = self._align_frames_spatially(self._input_motions)
        self._aligned_frames, self._temporal_data = self._align_frames_temporally_split(aligned_frames, self._dtw_sections)

        #self._export_aligned_frames()

    def _export_aligned_frames(self):
        for idx, frames in enumerate(self._aligned_frames):
            print(np.array(frames).shape)
            name = "aligned"+str(idx)
            self._export_frames(name, frames)

    def _align_frames_spatially(self, input_motions):
        print("run spatial alignment", self.ref_orientation)
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
            delta = np.array(ma[0, :3]) # + self._skeleton.nodes[self._skeleton.root].offset
            for f in ma:
                f[:3] -= delta
            aligned_frames.append(ma)
        return aligned_frames

    def get_average_time_line(self, input_motions):
        n_frames = [len(m) for m in input_motions]
        mean = np.mean(n_frames)
        best_idx = 0
        least_distance = np.inf
        for idx, n in enumerate(n_frames):
            if abs(n-mean) < least_distance:
                best_idx = idx
                least_distance = abs(n-mean)
        return best_idx

    def _align_frames_temporally(self, input_motions, mean_idx=None):
        print("run temporal alignment")
        print("convert motions to point clouds")
        point_clouds = convert_poses_to_point_clouds(self._skeleton, input_motions, normalize=False)
        print("find reference motion")
        if mean_idx is None:
            mean_idx = self.get_average_time_line(input_motions)
            print("set reference to index", mean_idx, "of", len(input_motions), "motions")
        dtw_results = find_optimal_dtw_async(point_clouds, mean_idx)
        warped_frames = []
        warping_functions = []
        for idx, m in enumerate(input_motions):
            print("align motion", idx)
            path = dtw_results[idx]
            warping_function = get_warping_function(path)
            warped_motion = warp_motion(m, warping_function)
            warped_frames.append(warped_motion)
            warping_functions.append(warping_function)
        warped_frames = np.array(warped_frames)
        n_samples = len(point_clouds)
        n_frames = len(warped_frames[0])
        n_dims = len(warped_frames[0][0])
        warped_frames = warped_frames.reshape((n_samples, n_frames, n_dims))
        return warped_frames, warping_functions

    def _align_frames_temporally_split(self, input_motions, sections=None):
        mean_idx = self.get_average_time_line(input_motions)
        if sections is not None:
            print("set reference to index", mean_idx, "of", len(input_motions), "motions", sections[mean_idx])
            for i, s in enumerate(sections[mean_idx]):
                self._keyframes["contact" + str(i)] = s["end_idx"]
        # split_motions into sections
        n_motions = len(input_motions)
        if sections is not None:
            splitted_motions = []
            for idx, input_motion in enumerate(input_motions):
                splitted_motion = []
                for section in sections[idx]:
                    start_idx = section["start_idx"]
                    end_idx = section["end_idx"]
                    split = input_motion[start_idx:end_idx]
                    print("split motion", idx, len(splitted_motion))
                    splitted_motion.append(split)
                splitted_motions.append(splitted_motion)
            splitted_motions = np.array(splitted_motions).T
        else:
            splitted_motions = [input_motions]

        # run dtw for each section
        splitted_dtw_results = []
        for section_samples in splitted_motions:
            result = self._align_frames_temporally(section_samples, mean_idx)
            splitted_dtw_results.append(result)

        # combine sections
        warped_frames = []
        warping_functions = []
        for motion_idx in range(n_motions):
            combined_frames = []
            combined_warping_function = []
            for section_idx, result in enumerate(splitted_dtw_results):
                print(motion_idx, section_idx, len(result),len(result[0]), n_motions)
                combined_frames += list(result[0][motion_idx])
                combined_warping_function += list(result[1][motion_idx])
            warped_frames.append(combined_frames)
            warping_functions.append(combined_warping_function)

        return warped_frames, warping_functions

    def run_dimension_reduction(self):
        self.run_spatial_dimension_reduction()
        self.run_temporal_dimension_reduction()

    def run_spatial_dimension_reduction(self):
        scaled_quat_frames, scale_vec = normalize_root_translation(self._aligned_frames)
        smoothed_quat_frames = np.array(align_quaternion_frames(self._skeleton, scaled_quat_frames))
        fpca_spatial = FPCASpatialData(self.config["n_basis_functions_spatial"],
                                       self.config["n_components"],
                                       self.config["fraction"])
        fpca_spatial.fileorder = list(range(len(smoothed_quat_frames)))
        print(smoothed_quat_frames.shape)
        fpca_spatial.fit(smoothed_quat_frames)

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
        print(np.array(self._temporal_data).shape)
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
            print(spatial_parameters, temporal_parameters)
            motion_parameters = np.concatenate((spatial_parameters, temporal_parameters,),axis=1)
        else:
            motion_parameters = self._spatial_fpca_data["parameters"]
        trainer = GMMTrainer()
        trainer.fit(motion_parameters)
        self._gmm_data = trainer.convert_model_to_json()

    def convert_motion_model_to_json(self, name="", version=1, save_skeleton=False):
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
        if save_skeleton:
            data["skeleton"] = self.skeleton.to_json()
        data["keyframes"] = self._keyframes
        return data

