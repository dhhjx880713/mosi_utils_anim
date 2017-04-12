"""
Code to export an animated skeleton to an FBX file based on a skeleton and a motion vector.
The code is based on Example01 of the FBX SDK samples.
"""


has_fbx = True
try:
    import FbxCommon
except ImportError, e:
    has_fbx = False
    print("Warning: could not import FBX library")
    pass

if has_fbx:
    from fbx_import import load_fbx_file
    from fbx_export import export_motion_vector_to_fbx_file

else:
    def load_fbx_file(file_path):
        raise NotImplementedError

    def export_motion_vector_to_fbx_file(skeleton, motion_vector, out_file_name):
        raise NotImplementedError


