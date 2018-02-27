import os
# change working directory to the script file directory
file_dir_name, file_name = os.path.split(os.path.abspath(__file__))
import sys
sys.path.append(os.sep.join([file_dir_name,'..',  'mgrd']))  # add mgrd package to import path
from .motion_generator import MotionGenerator, GraphWalkOptimizer, DEFAULT_ALGORITHM_CONFIG, AnnotatedMotionVector
from .animation_data.motion_editing import LegacyInverseKinematics
from .constraints import ik_constraints
from .utilities import load_json_file, write_to_json_file, get_bvh_writer
#from animation_data import Skeleton, BVHReader, AnimationClip, MotionVector