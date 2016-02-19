import os
# change working directory to the script file directory
file_dir_name, file_name = os.path.split(os.path.abspath(__file__))
import sys
sys.path.append(os.sep.join([file_dir_name,'..',  'mgrd']))  # add mgrd package to import path

from motion_generator import MotionGenerator, AlgorithmConfigurationBuilder, InverseKinematics, AnnotatedMotionVector
from utilities import load_json_file, get_bvh_writer
from animation_data import Skeleton, BVHReader