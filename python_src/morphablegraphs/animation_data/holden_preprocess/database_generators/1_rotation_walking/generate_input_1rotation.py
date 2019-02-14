import sys
import numpy as np
import scipy.ndimage.filters as filters
import math

sys.path.append("../")

import helpers.motion.BVH as BVH
import helpers.motion.Animation as Animation
from helpers.motion.Quaternions import Quaternions
from helpers.motion.Pivots import Pivots

to_meters = 0.1
window = 60


masks = ["%s/%s_cutted.bvh", "%s/%s_1_cutted.bvh", "%s/%s_2_cutted.bvh","%s/%s_3_cutted.bvh","%s/%s_4_cutted.bvh"]
mirrored = ["%s/%s_mirror_cutted.bvh","%s/%s_1_mirror_cutted.bvh", "%s/%s_2_mirror_cutted.bvh", "%s/%s_3_mirror_cutted.bvh", "%s/%s_4_mirror_cutted.bvh"]


class Participant():
	def __init__(self, path, name, gender):
		self.gender = gender
		self.path, self.name = path, name
		Xpart, Ypart, Ppart = self.load_data_masks(masks)
		self.Xun = np.concatenate(Xpart, axis=0)
		self.Yun = np.concatenate(Ypart, axis=0)
		self.Pun = np.concatenate(Ppart, axis=0)
		Xpart, Ypart, Ppart = self.load_data_masks(mirrored)
		self.Xun_m = np.concatenate(Xpart, axis=0)
		self.Yun_m = np.concatenate(Ypart, axis=0)
		self.Pun_m = np.concatenate(Ppart, axis=0)


		self.train = {"Xun":self.Xun, "Yun": self.Yun, "Pun": self.Pun}
		self.test = {}

	def load_data_masks(self, mask_):
		Ppart, Xpart, Ypart = [], [], []
		for m in mask_:
			data = m%(self.path, self.name)
			print("processing: ", data)

			anim, names, _ = BVH.load(data)
			anim.offsets *= to_meters
			anim.directions *= to_meters
			anim.positions *= to_meters
			self.anim = anim[::2]  # reduce framerate from 120 to 60 fps
			self.names = names

			""" Load Phase / Gait """

			self.phase = np.loadtxt(data.replace('.bvh', '.phase'))[::2]
			gait = np.loadtxt(data.replace('.bvh', '.gait'))[::2]
			self.gait = np.concatenate([
				gait[:, 0:1],  # standing
				gait[:, 1:2] + gait[:, 3:4],  # walking + backwards walking
				gait[:, 2:3],  # running
			], axis=-1)

			""" Process Data """
			Pc, Xc, Yc = self.__process_data()
			Ppart.append(Pc.astype(np.float32))
			Xpart.append(Xc.astype(np.float32))
			Ypart.append(Yc.astype(np.float32))
		return (Xpart, Ypart, Ppart)

	def generate_train_test(self, rng, test_prop = 0.2, randomized_indices_file = False):
		if not randomized_indices_file:
			indices = np.arange(0, len(self.Xun))
			np.random.shuffle(indices)
			np.save(self.path + "/randomization_%s.npy"%self.name, indices)
		else:
			indices = np.load(self.path + "/randomization_%s.npy"%self.name)
			print("indices loaded from ", self.path + "/randomization_%s.npy"%self.name, indices)
		train_indices = indices[0:int(len(indices) * (1 - 0.1))]
		test_indices = indices[int(len(indices) * (1 - 0.1)):]
		xun = np.concatenate([self.Xun[train_indices], self.Xun_m[train_indices]], axis = 0)
		yun = np.concatenate([self.Yun[train_indices], self.Yun_m[train_indices]], axis=0)
		pun = np.concatenate([self.Pun[train_indices], self.Pun_m[train_indices]], axis=0)
		self.train = {"Xun": xun, "Yun": yun, "Pun": pun}
		xun = np.concatenate([self.Xun[test_indices], self.Xun_m[test_indices]], axis=0)
		yun = np.concatenate([self.Yun[test_indices], self.Yun_m[test_indices]], axis=0)
		pun = np.concatenate([self.Pun[test_indices], self.Pun_m[test_indices]], axis=0)
		self.test =  {"Xun": xun,  "Yun": yun,  "Pun": pun }

	def save_models(self, path):
		#np.savez_compressed(path%"all", Xun=self.Xun, Yun=self.Yun, Pun=self.Pun)
		np.savez_compressed(path%"train", Xun=self.train["Xun"], Yun=self.train["Yun"], Pun=self.train["Pun"])
		np.savez_compressed(path % "test", Xun=self.test["Xun"], Yun=self.test["Yun"], Pun=self.test["Pun"])

	def __process_data(self):
		anim = self.anim
		phase = self.phase
		gait = self.gait
		""" Do FK """
		global_xforms = Animation.transforms_global(anim)
		global_positions = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
		global_rotations = Quaternions.from_transforms(global_xforms)

		end_joints = [3, 6, 12, 16, 39]  # 19, 23, 27, 31, 35, 42, 46, 50, 54, 58] # finger joints are currently excluded

		# test:
		end_joint_locations = np.zeros([len(global_positions), len(end_joints), 3])
		for f in range(len(global_positions)):
			for i in range(0, len(end_joints)):
				end_joint_locations[f][i] = (global_rotations[f][end_joints[i]] * anim.directions[end_joints[i]] + global_positions[f][end_joints[i]])

		""" Extract Forward Direction """

		sdr_l, sdr_r, hip_r, hip_l = 36, 13, 1, 4  # 18, 25, 2, 7
		across = (
				(global_positions[:, sdr_l] - global_positions[:, sdr_r]) +
				(global_positions[:, hip_l] - global_positions[:, hip_r]))
		across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

		""" Smooth Forward Direction """

		direction_filterwidth = 20
		forward = filters.gaussian_filter1d(
			np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
		forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

		root_rotation = Quaternions.between(forward,
											np.array([[0, 0, 1]]).repeat(len(forward), axis=0))[:, np.newaxis]

		""" Local Space """

		# root_relevative positions
		global_positions = np.append(global_positions, end_joint_locations, axis=1)
		local_positions = global_positions.copy()
		local_positions[:, :, 0] = local_positions[:, :, 0] - local_positions[:, 0:1, 0]
		local_positions[:, :, 2] = local_positions[:, :, 2] - local_positions[:, 0:1, 2]

		# rotate around root_rotation to make forward (0,0,1)
		local_positions = root_rotation[:-1] * local_positions[:-1]
		local_velocities = root_rotation[:-1] * (global_positions[1:] - global_positions[:-1])
		local_rotations = root_rotation[:-1] * global_rotations[:-1]
		twists = self.__twistRotations(anim, local_rotations)
		twists[twists > math.pi] -= 2 * math.pi

		local_rotations = abs(local_rotations).log()

		root_velocity = root_rotation[:-1] * (global_positions[1:, 0:1] - global_positions[:-1, 0:1])
		root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps

		""" Phase """

		dphase = phase[1:] - phase[:-1]
		dphase[dphase < 0] = (1.0 - phase[:-1] + phase[1:])[dphase < 0]

		""" Start Windows """

		Pc, Xc, Yc = [], [], []

		for i in range(window, len(anim) - window - 1, 1):
			rootposs = root_rotation[i:i + 1, 0] * (global_positions[i - window:i + window:10, 0] - global_positions[i:i + 1, 0])
			rootdirs = root_rotation[i:i + 1, 0] * forward[i - window:i + window:10]

			rootgait = gait[i - window:i + window:10]

			Pc.append(phase[i])

			# remove unnecessary joints:
			locpos = np.concatenate((local_positions[i - 1][0:17], local_positions[i - 1][36:40], local_positions[i - 1][59:]))
			locvel = np.concatenate((local_velocities[i - 1][0:17], local_velocities[i - 1][36:40], local_velocities[i - 1][59:]))
			tw = np.concatenate((twists[i - 1][0:17], twists[i - 1][36:40]))

			Xc.append(np.hstack([
				rootposs[:, 0].ravel(), rootposs[:, 2].ravel(),  # Trajectory Pos
				rootdirs[:, 0].ravel(), rootdirs[:, 2].ravel(),  # Trajectory Dir
				rootgait[:, 0].ravel(), rootgait[:, 1].ravel(),  # Trajectory Gait
				rootgait[:, 2].ravel(),  # rootgait[:,3].ravel(),
				locpos.ravel(),
				locvel.ravel(),
				tw.ravel(),
				self.gender
			]))

			rootposs_next = root_rotation[i + 1:i + 2, 0] * (global_positions[i + 1:i + window + 1:10, 0] - global_positions[i + 1:i + 2, 0])
			rootdirs_next = root_rotation[i + 1:i + 2, 0] * forward[i + 1:i + window + 1:10]

			# remove unnecessary joints:
			locpos = np.concatenate((local_positions[i][0:17], local_positions[i][36:40], local_positions[i][59:]))
			locvel = np.concatenate((local_velocities[i][0:17], local_velocities[i][36:40], local_velocities[i][59:]))
			tw = np.concatenate((twists[i][0:17], twists[i][36:40]))

			Yc.append(np.hstack([
				root_velocity[i, 0, 0].ravel(),  # Root Vel X
				root_velocity[i, 0, 2].ravel(),  # Root Vel Y
				root_rvelocity[i].ravel(),  # Root Rot Vel
				dphase[i],  # Change in Phase
				rootposs_next[:, 0].ravel(), rootposs_next[:, 2].ravel(),  # Next Trajectory Pos
				rootdirs_next[:, 0].ravel(), rootdirs_next[:, 2].ravel(),  # Next Trajectory Dir
				locpos.ravel(),
				locvel.ravel(),
				tw.ravel(),
			]))
		return np.array(Pc), np.array(Xc), np.array(Yc)


	def __twistRotations(self, anim, global_rotations):
		twists = np.array([0.0] * global_rotations.shape[0] * global_rotations.shape[1]).reshape(global_rotations.shape[0], global_rotations.shape[1])
		for f in range(len(global_rotations)):
			for b in range(len(global_rotations[f])):
				q = Quaternions(global_rotations[f][b])
				base_direction = np.array(anim.directions[b])
				(swing, twist) = q.swing_twist_decomposition(base_direction)
				twist_angle = twist.angle_axis()[0][0]
				twists[f][b] = twist_angle
		return twists
