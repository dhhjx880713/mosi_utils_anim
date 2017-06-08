# encoding: UTF-8

from motion_dtw import MotionDynamicTimeWarping
import os
from ...animation_data import BVHReader, Skeleton
from ...motion_analysis import BVHAnalyzer


def run_dtw_demo():
    dtw_demo = MotionDynamicTimeWarping(verbose=False)
    cut_files = r'C:\repo\data\1 - MoCap\3 - Cutting\elementary_action_walk\leftStance'
    dtw_demo.load_motion_from_files_for_DTW(cut_files)
    ref_bvh = r'C:\repo\data\1 - MoCap\3 - Cutting\elementary_action_walk\leftStance\walk_011_2_leftStance_378_425.bvh'
    test_bvh = r'C:\repo\data\1 - MoCap\3 - Cutting\elementary_action_walk\leftStance\walk_s_014_leftStance_711_760.bvh'
    test_filename = os.path.split(test_bvh)[-1]
    ref_filename = os.path.split(ref_bvh)[-1]
    dtw_demo.set_ref_motion(ref_bvh)
    # (filename, frames) = dtw_demo.aligned_motions.popitem()

    test_motion = {'filename': test_filename, 'frames': dtw_demo.aligned_motions[test_filename]}
    ref_motion = {'filename': ref_filename, 'frames': dtw_demo.aligned_motions[ref_filename]}
    # dtw_demo.warp_test_motion_to_ref_motion(dtw_demo.ref_motion, test_motion)
    distgrid = dtw_demo.get_distgrid(test_motion, ref_motion)
    n_test_frames, n_ref_frames = distgrid.shape

    print(distgrid[0, 0])

    print(distgrid[n_test_frames-1, 0])
    print(distgrid[n_test_frames-1, n_ref_frames-1])

def test_frame_distance():
    from morphablegraphs.utilities import load_json_file
    from morphablegraphs.animation_data.motion_editing import *
    ref_bvh = r'C:\repo\data\tmp\tmp\walk_011_2_leftStance_378_425.bvh'
    test_bvh = r'C:\repo\data\tmp\tmp\walk_s_014_leftStance_711_760.bvh'
    # test_bvh = r'C:\repo\data\1 - MoCap\3 - Cutting\elementary_action_walk\leftStance\walk_033_3_leftStance_487_543.bvh'
    ref_bvhreader = BVHReader(ref_bvh)
    test_bvhreader = BVHReader(test_bvh)
    skeleton = Skeleton()
    skeleton.load_from_bvh(ref_bvhreader)

    # error1, theta, offset_x, offset_z = calculate_frame_distance(skeleton, ref_bvhreader.frames[0],
    #                                                             test_bvhreader.frames[0], return_transform=True)
    # print(error1)
    # error2, theta, offset_x, offset_z = calculate_frame_distance(skeleton, ref_bvhreader.frames[0],
    #                                                              test_bvhreader.frames[-1], return_transform=True)
    # print(error2)
    # error3, theta, offset_x, offset_z = calculate_frame_distance(skeleton, ref_bvhreader.frames[-1],
    #                                                              test_bvhreader.frames[-1], return_transform=True)
    # print(error3)
    # error4, theta, offset_x, offset_z = calculate_frame_distance(skeleton, ref_bvhreader.frames[-1],
    #                                                              test_bvhreader.frames[0], return_transform=True)
    # print(error4)
    # test_joint_name = 'Head'
    # head_pos = get_cartesian_coordinates_from_euler(skeleton, test_joint_name, ref_bvhreader.frames[0])
    # print(head_pos)


    # ref_bvh = r'C:\repo\data\tmp\raw\walk_011_2_leftStance_378_425.bvh'
    # test_bvh = r'C:\repo\data\tmp\raw\walk_s_014_leftStance_711_760.bvh'
    # # test_bvh = r'C:\repo\data\1 - MoCap\3 - Cutting\elementary_action_walk\leftStance\walk_033_3_leftStance_487_543.bvh'
    # ref_bvhreader = BVHReader(ref_bvh)
    # test_bvhreader = BVHReader(test_bvh)
    # skeleton = Skeleton()
    # skeleton.load_from_bvh(ref_bvhreader)
    #
    # error1, theta, offset_x, offset_z = calculate_frame_distance(skeleton, ref_bvhreader.frames[0],
    #                                                             test_bvhreader.frames[0], return_transform=True)
    # print(error1)
    # error2, theta, offset_x, offset_z = calculate_frame_distance(skeleton, ref_bvhreader.frames[0],
    #                                                              test_bvhreader.frames[-1], return_transform=True)
    # print(error2)
    # error3, theta, offset_x, offset_z = calculate_frame_distance(skeleton, ref_bvhreader.frames[-1],
    #                                                              test_bvhreader.frames[-1], return_transform=True)
    # print(error3)
    # error4, theta, offset_x, offset_z = calculate_frame_distance(skeleton, ref_bvhreader.frames[-1],
    #                                                              test_bvhreader.frames[0], return_transform=True)
    # print(error4)




    # print(theta)
    # print(offset_x)
    # print(offset_z)
    ref_analyzer = BVHAnalyzer(ref_bvhreader)
    # print(ref_analyzer.get_global_pos(test_joint_name, 0))
    test_analyzer = BVHAnalyzer(test_bvhreader)
    test_joint = 'RightForeArm'
    # print("#################################")
    # print(test_analyzer.get_global_pos(test_joint, 0))
    #
    # print(get_cartesian_coordinates_from_euler(skeleton, test_joint, ref_bvhreader.frames[0]))

    point_cloud_json_data = load_json_file(r'C:\Users\hadu01\git-repos\ulm\morphablegraphs\test_data\animation_data\point_cloud_data.json')
    point_cloud_dic_a = point_cloud_json_data['point_cloud_a']
    point_cloud_a = []
    point_cloud_b = []
    joint_names = []
    for key in point_cloud_dic_a.keys():
        if not 'end' in key.lower() and not 'bip' in key.lower():
        # if 'end' in key.lower():
        #     key = key[:-3] + '_EndSite'
            joint_names.append(key)
            point_cloud_a.append(ref_analyzer.get_global_pos(key, 0))
            point_cloud_b.append(test_analyzer.get_global_pos(key,0))

    point_cloud_a1 = convert_euler_frame_to_cartesian_frame(skeleton, ref_bvhreader.frames[0])
    print(np.asarray(point_cloud_a1).shape)

    # print(len(point_cloud_a1))
    # print(len(point_cloud_a))
    # for i in range(len(point_cloud_a)):
    #     print("################################")
    #     print(joint_names[i])
    #     print(point_cloud_a[i])
    #     print(point_cloud_a1[i])
    # theta, offset_x, offset_z = align_point_clouds_2D(point_cloud_a, point_cloud_b)
    # t_point_cloud_b = transform_point_cloud(point_cloud_b, theta, offset_x, offset_z)
    # error = calculate_point_cloud_distance(point_cloud_a, t_point_cloud_b)
    # print(error)
    #
    # point_cloud_a = []
    # point_cloud_b = []
    # for key in point_cloud_dic_a.keys():
    #     if not 'end' in key.lower() and not 'bip' in key.lower():
    #     # if 'end' in key.lower():
    #     #     key = key[:-3] + '_EndSite'
    #         point_cloud_a.append(ref_analyzer.get_global_pos(key, -1))
    #         point_cloud_b.append(test_analyzer.get_global_pos(key, 0))
    # theta, offset_x, offset_z = align_point_clouds_2D(point_cloud_a, point_cloud_b)
    # t_point_cloud_b = transform_point_cloud(point_cloud_b, theta, offset_x, offset_z)
    # error = calculate_point_cloud_distance(point_cloud_a, t_point_cloud_b)
    # print(error)
    #
    #
    # point_cloud_a = []
    # point_cloud_b = []
    # for key in point_cloud_dic_a.keys():
    #     if not 'end' in key.lower() and not 'bip' in key.lower():
    #     # if 'end' in key.lower():
    #     #     key = key[:-3] + '_EndSite'
    #         point_cloud_a.append(ref_analyzer.get_global_pos(key, -1))
    #         point_cloud_b.append(test_analyzer.get_global_pos(key, -1))
    # theta, offset_x, offset_z = align_point_clouds_2D(point_cloud_a, point_cloud_b)
    # t_point_cloud_b = transform_point_cloud(point_cloud_b, theta, offset_x, offset_z)
    # error = calculate_point_cloud_distance(point_cloud_a, t_point_cloud_b)
    # print(error)




if __name__ == "__main__":
    # run_dtw_demo()
    test_frame_distance()