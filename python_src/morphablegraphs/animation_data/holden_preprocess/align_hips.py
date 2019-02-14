import math, os, glob, re
import numpy as np
from shutil import copyfile
from pathlib import Path


offsets = np.array([])
hip_heights = {}
text = [""]
skel_texts  = {}
motion_text = {}
write_text = True

for gl in glob.glob("../data/data_praktikum/*.*"):
	if "bvh" in gl:
		print("processing: ", gl)
		tmp = []
		motions = False
		with open(gl, "r") as fo:
			skel_texts[gl], motion_text[gl] = fo.read().split("MOTION")
			skel_text = skel_texts[gl]
			for line in skel_text.split("\n"):
				if "MOTION" in line:
					break
				if not motions:
					offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
					if offmatch:
						par = -1
						tmp.append(list(map(float, offmatch.groups())))
			fo.close()
		hip_heights[gl] = tmp[0][1]
		tmp = np.array(tmp)
		if len(offsets) == 0:
			offsets = np.array([tmp])
		else:
			offsets = np.append(offsets, [tmp], axis = 0)
		write_text = False
	else:
		copyfile(gl, gl.replace("data_praktikum", "data_praktikum_aligned"))

mean_offsets = np.mean(offsets, axis=0)
mean_hip_height = mean_offsets[0][1]

for gl in glob.glob("../data/data_praktikum/*.bvh"):
	gl_new = gl.replace("data_praktikum", "data_praktikum_aligned")

	with open(gl_new, "w+") as f:
		f.write(skel_texts[gl])
		f.write("\nMOTION\n")

		old_height, new_height = hip_heights[gl], mean_hip_height
		diff = old_height - new_height

		motion_splits = motion_text[gl].split("\n")
		for line_i in range(len(motion_splits)):
			data = motion_splits[line_i].split(" ")
			if len(data) > 20:
				data[0] = str(float(data[0]))
				data[1] = str(float(data[1]) - diff)
				data[2] = str(float(data[2]))
				motion_splits[line_i] = " ".join(data)
		motion_text[gl] = "\n".join(motion_splits)
		f.write(motion_text[gl])
		f.close()
#

