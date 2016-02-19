ROTATION_TYPE_EULER = 0
ROTATION_TYPE_QUATERNION = 1
from .motion_vector import MotionVector
from .motion_editing import align_quaternion_frames, transform_euler_frames, transform_quaternion_frames
from .bvh import BVHReader, BVHWriter
from .skeleton import Skeleton