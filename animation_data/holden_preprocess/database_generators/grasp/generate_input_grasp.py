import sys
import numpy as np
import scipy.ndimage.filters as filters
import math

sys.path.append('../../')

import motion.BVH as BVH
import motion.Animation as Animation
from motion.Quaternions import Quaternions
from motion.Pivots import Pivots
import os

to_meters = 0.1
window = 60


class ParticipantGrasp():
	def __init__(self, path):
		Ppart, Xpart, Ypart = [], [], []
		data = path
		print("processing: ", data)

		anim, names, _ = BVH.load(data)
		anim.offsets *= to_meters
		anim.directions *= to_meters
		anim.positions *= to_meters

		self.anim = anim[::2]  # reduce framerate from 120 to 60 fps
		self.names = names

		""" Load Phase / Gait """

		self.phase = np.loadtxt(data.replace('.bvh', '.phase'))[::2]
		if os.path.exists(data.replace('.bvh', '.gait')):
			gait = np.loadtxt(data.replace('.bvh', '.gait'))[::2]
			self.gait = np.concatenate([
				gait[:, 0:1],  # standing
				gait[:, 1:2] + gait[:, 3:4],  # walking + backwards walking
				gait[:, 2:3],  # running
				np.zeros(gait[:,2:3].shape) 			# grasping
			], axis=-1)
		else:
			self.gait = np.reshape(([0, 0, 0, 1] * len(self.phase)), (len(self.phase), 4))


		""" Process Data """
		Pc, Xc, Yc = self.__process_data()
		Ppart.append(Pc.astype(np.float32))
		Xpart.append(Xc.astype(np.float32))
		Ypart.append(Yc.astype(np.float32))

		self.Xun = np.concatenate(Xpart, axis=0)
		self.Yun = np.concatenate(Ypart, axis=0)
		self.Pun = np.concatenate(Ppart, axis=0)

		self.train = {"Xun":self.Xun, "Yun": self.Yun, "Pun": self.Pun}
		self.test = {}

	def generate_train_test(self, rng, test_prop = 0.2):
		indices = np.arange(0, len(self.Xun))
		np.random.shuffle(indices)
		train_indices = indices[0:int(len(indices) * (1 - 0.2))]
		test_indices = indices[int(len(indices) * (1 - 0.2)):]
		self.train = {"Xun": self.Xun[train_indices], "Yun": self.Yun[train_indices], "Pun": self.Pun[train_indices]}
		self.test =  {"Xun": self.Xun[test_indices],  "Yun": self.Yun[test_indices],  "Pun":self.Pun[test_indices] }

	def save_models(self, path):
		np.savez_compressed(path%"all", Xun=self.Xun, Yun=self.Yun, Pun=self.Pun)
		np.savez_compressed(path%"train", Xun=self.train["Xun"], Yun=self.train["Yun"], Pun=self.train["Pun"])
		if len(self.test) > 0:
			np.savez_compressed(path % "test", Xun=self.test["Xun"], Yun=self.test["Yun"], Pun=self.test["Pun"])

	def __process_data(self):
		anim = self.anim
		phase = self.phase
		gait = self.gait
		""" Do FK """
		global_xforms = Animation.transforms_global(anim)
		global_positions = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
		global_rotations = Quaternions.from_transforms(global_xforms)

		end_joints = [3, 6, 12, 19, 23, 27, 31, 35, 42, 46, 50, 54, 58] # with fingers now

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

		""" Grasp End Effectors """
		end_effector_joints = 16 # just right hand grasping for now.
		#if len(np.where(phase[:] == 0.5)[0]) > 0:
		half_phase = np.where(phase[:] == 0.5)[0][0]
		target_global_position = np.reshape(([global_positions[half_phase][end_effector_joints]] * len(global_positions)), (len(global_positions), 1, 3))
		global_target_directions = global_positions[:,end_effector_joints:end_effector_joints+1,:] - target_global_position[:,0:1,:]

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

		local_target_directions = root_rotation[:-1] * global_target_directions[:-1]

		""" Phase """

		dphase = phase[1:] - phase[:-1]
		dphase[dphase < 0] = (1.0 - phase[:-1] + phase[1:])[dphase < 0]

		""" Start Windows """

		Pc, Xc, Yc = [], [], []

		for i in range(window, len(anim) - window - 1, 1):
			rootposs = root_rotation[i:i + 1, 0] * (global_positions[i - window:i + window:10, 0] - global_positions[i:i + 1, 0])
			rootdirs = root_rotation[i:i + 1, 0] * forward[i - window:i + window:10]
			#target_dirs = root_rotation[i:i + 1, 0] * global_target_directions[i - window: i + window:10]
			if gait[i][3] == 1:
				target_dirs = local_target_directions[i - 1]
			else:
				target_dirs = np.array([0,0,0])
			rootgait = gait[i - window:i + window:10]

			Pc.append(phase[i])

			# remove unnecessary joints:
			# locpos = np.concatenate((local_positions[i - 1][0:17], local_positions[i - 1][36:40], local_positions[i - 1][59:]))
			# locvel = np.concatenate((local_velocities[i - 1][0:17], local_velocities[i - 1][36:40], local_velocities[i - 1][59:]))
			locpos = local_positions[i-1][:]
			locvel = local_velocities[i-1][:]
			tw = twists[i-1][:]
			#tw = np.concatenate((twists[i - 1][0:17], twists[i - 1][36:40]))

			# print("lengths: ",
			# 	  len(rootposs[:, 0].ravel()), len(rootposs[:, 2].ravel()), "\n",
			# 	  len(rootdirs[:, 0].ravel()), len(rootdirs[:, 2].ravel()), "\n",
			# 	  len(rootgait[:, 0].ravel()), len(rootgait[:, 1].ravel()), "\n", # Trajectory Gait
			# 	  len(rootgait[:, 2].ravel()), len(rootgait[:, 3].ravel()), "\n",
			# 	  len(target_dirs.ravel()), "\n",
			# 	  len(locpos.ravel()),"\n",
			# 	  len(locvel.ravel()),"\n",
			# 	  len(tw.ravel()))

			Xc.append(np.hstack([
				rootposs[:, 0].ravel(), rootposs[:, 2].ravel(), # Trajectory Pos
				rootdirs[:, 0].ravel(), rootdirs[:, 2].ravel(),  # Trajectory Dir
				rootgait[:, 0].ravel(), rootgait[:, 1].ravel(),  # Trajectory Gait
				rootgait[:, 2].ravel(), rootgait[:,3].ravel(),
				target_dirs.ravel(),
				locpos.ravel(),
				locvel.ravel(),
				tw.ravel(),
			]))

			rootposs_next = root_rotation[i + 1:i + 2, 0] * (global_positions[i + 1:i + window + 1:10, 0] - global_positions[i + 1:i + 2, 0])
			rootdirs_next = root_rotation[i + 1:i + 2, 0] * forward[i + 1:i + window + 1:10]
			#target_dirs = root_rotation[i+1:i + 2, 0] * global_target_directions[i+1:  + window:10]
			if gait[i][3] == 1:
				target_dirs = local_target_directions[i]
			else:
				target_dirs = np.array([0,0,0])


			# remove unnecessary joints:
			# locpos = np.concatenate((local_positions[i][0:17], local_positions[i][36:40], local_positions[i][59:]))
			# locvel = np.concatenate((local_velocities[i][0:17], local_velocities[i][36:40], local_velocities[i][59:]))
			# tw = np.concatenate((twists[i][0:17], twists[i][36:40]))
			locpos = local_positions[i][:]
			locvel = local_velocities[i]
			tw = twists[i][:]

			# print("lenghts y: ",
			# 	  len(root_velocity[i, 0, 0].ravel()),  # Root Vel X
			# 	  len(root_velocity[i, 0, 2].ravel()),  # Root Vel Y
			# 	  len(root_rvelocity[i].ravel()),  # Root Rot Vel
			# 	  1,  # Change in Phase
			# 	  len(rootposs_next[:, 0].ravel()), len(rootposs_next[:, 2].ravel()),  # Next Trajectory Pos
			# 	  len(rootdirs_next[:, 0].ravel()), len(rootdirs_next[:, 2].ravel()),  # Next Trajectory Dir
			# 	  len(target_dirs.ravel()),
			# 	  len(locpos.ravel()),
			# 	  len(locvel.ravel()),
			# 	  len(tw.ravel()),
			# 	  )

			Yc.append(np.hstack([
				root_velocity[i, 0, 0].ravel(),  # Root Vel X
				root_velocity[i, 0, 2].ravel(),  # Root Vel Y
				root_rvelocity[i].ravel(),  # Root Rot Vel
				dphase[i],  # Change in Phase
				rootposs_next[:, 0].ravel(), rootposs_next[:, 2].ravel(),  # Next Trajectory Pos
				rootdirs_next[:, 0].ravel(), rootdirs_next[:, 2].ravel(),  # Next Trajectory Dir
				#target_dirs.ravel(),
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
