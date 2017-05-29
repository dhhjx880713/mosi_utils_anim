import os
# change working directory to the script file directory
file_dir_name, file_name = os.path.split(os.path.abspath(__file__))
import sys
sys.path.append(os.sep.join([file_dir_name,'..',  'mgrd']))  # add mgrd package to import path
from motion_generator import MotionGenerator, AlgorithmConfigurationBuilder, InverseKinematics, AnnotatedMotionVector
from motion_generator.motion_generator2 import MotionGenerator2
from constraints import ik_constraints
from utilities import load_json_file, write_to_json_file, get_bvh_writer
from animation_data import Skeleton, BVHReader