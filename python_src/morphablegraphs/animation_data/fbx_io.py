"""
Code to export an animated skeleton to an FBX file based on a skeleton and a motion vector.
The code is based on Example01 of the FBX SDK samples.
"""

from ..external.transformations import euler_from_quaternion
import numpy as np
has_fbx = True
try:
    import FbxCommon
    from fbx import *
except ImportError, e:
    has_fbx = False
    print("Warning: could not import FBX library")
    pass

if has_fbx:
    def create_scene(sdk_manager, scene, skeleton, motion_vector):

        info = FbxDocumentInfo.Create(sdk_manager, "SceneInfo")
        info.mTitle = "MotionExportScene"
        scene.SetSceneInfo(info)
        root_node = create_skeleton(sdk_manager, "", skeleton)
        scene.GetRootNode().AddChild(root_node)
        set_rest_pose(sdk_manager, scene, root_node, skeleton)
        set_animation_curves(scene, root_node, skeleton, motion_vector)


    def create_skeleton(sdk_manager, name, skeleton):
        root_node = create_skeleton_nodes_recursively(sdk_manager, name, skeleton, skeleton.root)
        return root_node

    def create_skeleton_nodes_recursively(sdk_manager, skeleton_name, skeleton, node_name):
        node = skeleton.nodes[node_name]
        name = skeleton_name + node_name
        skeleton_node_attribute = FbxSkeleton.Create(sdk_manager, skeleton_name)
        if node.parent is None:
            node_type = FbxSkeleton.eRoot
        else:
            node_type = FbxSkeleton.eLimbNode
        skeleton_node_attribute.SetSkeletonType(node_type)
        skeleton_node = FbxNode.Create(sdk_manager, name)
        skeleton_node.SetNodeAttribute(skeleton_node_attribute)
        t = FbxDouble3(node.offset[0], node.offset[1], node.offset[2])
        skeleton_node.LclTranslation.Set(t)
        for c_node in node.children:
            c_name = c_node.node_name
            c_node = create_skeleton_nodes_recursively(sdk_manager, skeleton_name, skeleton, c_name)
            skeleton_node.AddChild(c_node)
        return skeleton_node

    def set_rest_pose_recursively(pose, fbx_node, skeleton):
        name = fbx_node.GetName()
        node = skeleton.nodes[name]
        t = node.offset

        l_t = FbxVector4(t[0], t[1], t[2])
        l_r = FbxVector4(0.0, 0.0, 0.0)
        l_s = FbxVector4(1.0, 1.0, 1.0)

        transform = FbxMatrix()
        transform.SetTRS(l_t, l_r, l_s)
        pose.Add(fbx_node, transform, True)
        n_children = fbx_node.GetChildCount()
        for idx in xrange(n_children):
            c_node = fbx_node.GetChild(idx)
            set_rest_pose_recursively(pose, c_node, skeleton)

    def set_rest_pose(sdk_manager, scene, root_node, skeleton):
        pose = FbxPose.Create(sdk_manager, "RestPose")
        set_rest_pose_recursively(pose, root_node, skeleton)
        scene.AddPose(pose)

    def create_translation_curves(fbx_node, anim_layer, euler_frames, frame_time):
        time = FbxTime()
        x_curve = fbx_node.LclTranslation.GetCurve(anim_layer, "X", True)
        x_curve.KeyModifyBegin()
        for idx, frame in enumerate(euler_frames):
            time.SetSecondDouble(idx * frame_time)
            key_index = x_curve.KeyAdd(time)[0]
            x_curve.KeySetValue(key_index, frame[0])
            x_curve.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationCubic)
        x_curve.KeyModifyEnd()

        y_curve = fbx_node.LclTranslation.GetCurve(anim_layer, "Y", True)
        y_curve.KeyModifyBegin()
        for idx, frame in enumerate(euler_frames):
            time.SetSecondDouble(idx * frame_time)
            key_index = y_curve.KeyAdd(time)[0]
            y_curve.KeySetValue(key_index, frame[1])
            y_curve.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationCubic)
        y_curve.KeyModifyEnd()

        z_curve = fbx_node.LclTranslation.GetCurve(anim_layer, "Z", True)
        z_curve.KeyModifyBegin()
        for idx, frame in enumerate(euler_frames):
            time.SetSecondDouble(idx * frame_time)
            key_index = z_curve.KeyAdd(time)[0]
            z_curve.KeySetValue(key_index, frame[2])
            z_curve.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationCubic)
        z_curve.KeyModifyEnd()


    def create_rotation_curves(fbx_node, anim_layer, skeleton, euler_frames, frame_time):
        node_name = fbx_node.GetName()
        if node_name not in skeleton.animated_joints:
            return
        node_idx = skeleton.animated_joints.index(node_name)
        offset = node_idx * 3 + 3
        time = FbxTime()

        x_curve = fbx_node.LclRotation.GetCurve(anim_layer, "X", True)
        x_curve.KeyModifyBegin()
        for idx, frame in enumerate(euler_frames):
            time.SetSecondDouble(idx*frame_time)
            key_index = x_curve.KeyAdd(time)[0]
            e = frame[offset:offset+3][0]
            x_curve.KeySetValue(key_index, e)
            x_curve.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationLinear)
        x_curve.KeyModifyEnd()

        y_curve = fbx_node.LclRotation.GetCurve(anim_layer, "Y", True)
        y_curve.KeyModifyBegin()
        for idx, frame in enumerate(euler_frames):
            time.SetSecondDouble(idx * frame_time)
            key_index = y_curve.KeyAdd(time)[0]
            e = frame[offset:offset + 3][1]
            y_curve.KeySetValue(key_index, e)
            y_curve.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationLinear)
        y_curve.KeyModifyEnd()

        z_curve = fbx_node.LclRotation.GetCurve(anim_layer, "Z", True)
        z_curve.KeyModifyBegin()
        for idx, frame in enumerate(euler_frames):
            time.SetSecondDouble(idx * frame_time)
            key_index = z_curve.KeyAdd(time)[0]
            e = frame[offset:offset + 3][2]
            z_curve.KeySetValue(key_index, e)
            z_curve.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationLinear)
        z_curve.KeyModifyEnd()

    def convert_quaternion_to_euler_frame(skeleton, frame):
        n_dims = len(skeleton.animated_joints) * 3 + 3
        euler_frame = np.zeros(n_dims)
        euler_frame[:3] = frame[:3]
        target_offset = 3
        src_offset = 3
        for node in skeleton.animated_joints:
            q = frame[src_offset:src_offset+4]
            e = euler_from_quaternion(q)
            euler_frame[target_offset:target_offset+3] = np.degrees(e)
            target_offset += 3
            src_offset += 4
        return euler_frame

    def add_rotation_curves_recursively(fbx_node, anim_layer, skeleton, euler_frames, frame_time, is_root=False):
        if is_root:
            create_translation_curves(fbx_node, anim_layer, euler_frames, frame_time)
        create_rotation_curves(fbx_node, anim_layer, skeleton, euler_frames, frame_time)
        n_children = fbx_node.GetChildCount()
        for idx in xrange(n_children):
            c_node = fbx_node.GetChild(idx)
            add_rotation_curves_recursively(c_node, anim_layer, skeleton, euler_frames, frame_time)

    def set_animation_curves(scene, root_node, skeleton, motion_vector):
        # convert frames from quaternion to euler
        euler_frames = []
        for frame in motion_vector.frames:
            euler_frame = convert_quaternion_to_euler_frame(skeleton, frame)
            euler_frames.append(euler_frame)

        anim_stack_name = "default"
        anim_stack = FbxAnimStack.Create(scene, anim_stack_name)
        anim_layer = FbxAnimLayer.Create(scene, "Base Layer")
        anim_stack.AddMember(anim_layer)
        add_rotation_curves_recursively(root_node, anim_layer, skeleton, euler_frames, motion_vector.frame_time, is_root=True)


    def export_motion_vector_to_fbx_file(skeleton, motion_vector, out_file_name):
        sdk_manager, scene = FbxCommon.InitializeSdkObjects()

        if create_scene(sdk_manager, scene, skeleton, motion_vector):

            FbxCommon.SaveScene(sdk_manager, scene, out_file_name)

        sdk_manager.Destroy()
else:
    def export_motion_vector_to_fbx_file(skeleton, motion_vector, out_file_name):
        raise NotImplementedError

