from ..motion_vector import MotionVector
from ..skeleton_builder import SkeletonBuilder

has_fbx = True
try:
    import FbxCommon
except ImportError as e:
    has_fbx = False
    print("Warning: could not import FBX library")
    pass


if has_fbx:
    from .fbx_import import load_fbx_file
    from .fbx_export import export_motion_vector_to_fbx_file
else:
    def load_fbx_file(file_path):
        raise NotImplementedError

    def export_motion_vector_to_fbx_file(skeleton, motion_vector, out_file_name):
        raise NotImplementedError


def load_skeleton_and_animations_from_fbx(file_path):
    mesh_list, skeleton_def, animations = load_fbx_file(file_path)
    skeleton = SkeletonBuilder().load_from_fbx_data(skeleton_def)
    anim_names = list(animations.keys())
    motion_vectors = []
    if len(anim_names) > 0:
        anim_name = anim_names[0]
        mv = MotionVector()
        mv.from_fbx(animations[anim_name], skeleton.animated_joints)
        motion_vectors.append(mv)

    return skeleton, motion_vectors

