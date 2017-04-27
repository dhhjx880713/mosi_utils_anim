import json
import urllib2
from copy import deepcopy
import numpy as np
from ..constraints.spatial_constraints.keyframe_constraints import GlobalTransformCAConstraint
from ..animation_data.motion_editing import euler_angles_to_rotation_matrix
from ..animation_data.motion_concatenation import get_node_aligning_2d_transform, transform_quaternion_frames, get_transform_from_start_pose
from ..utilities import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR, get_bvh_writer, TCPClient


class CAInterface(object):
    def __init__(self, ea_generator, service_config):
        self.ea_generator = ea_generator
        self.activate_coordinate_transform = service_config["activate_coordinate_transform"]
        self.ca_service_url = service_config["collision_avoidance_service_url"]
        self.ca_service_port = service_config["collision_avoidance_service_port"]
        if self.ca_service_url is not None:
            self.tcp_client = TCPClient(self.ca_service_url, self.ca_service_port)
        else:
            self.tcp_client = None
        self.coordinate_transform_matrix = np.array([[1, 0, 0, 0],
                                                     [0, 0, -1, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 1]])

    def get_constraints(self, group_id, new_node, new_motion_spline, graph_walk):
        """ Generate constraints using the rest interface of the collision avoidance module directly.
        """
        aligned_motion_spline, global_transformation = self._get_aligned_motion_spline(new_motion_spline,
                                                                                       graph_walk.get_quat_frames())
        if self.activate_coordinate_transform:
            global_transformation = np.dot(self.coordinate_transform_matrix, global_transformation)
        frames = aligned_motion_spline.get_motion_vector()
        global_bvh_string = get_bvh_writer(self.ea_generator.motion_state_graph.skeleton, frames).generate_bvh_string()
        ca_input = {"groupId": group_id, "command":"GenerateConstraints"}
        ca_input["parameters"] = {"elementary_action_name": new_node[0],
                    "motion_primitive_name": new_node[1],
                    "global_transform": global_transformation.tolist(),
                    "global_bvh_frames": global_bvh_string}
        message = json.dumps(ca_input)
        if self.tcp_client is not None:
            ca_output_string = self.tcp_client.send_message(message)
            ca_output_string = ca_output_string.replace("right", "Right")
            ca_output_string = ca_output_string.replace("left", "Left")
            try:
                ca_output = json.loads(ca_output_string)
            except Exception as e:
                ca_output = {}
            if ca_output is not None:
                return self._create_ca_constraints(new_node, ca_output, graph_walk)
            else:
                return None

    def _create_ca_constraints(self, new_node, ca_output, graph_walk):
        ca_constraints = []
        n_canonical_frames = int(self.ea_generator.motion_state_graph.nodes[new_node].get_n_canonical_frames())
        for joint_name in ca_output.keys():
            for ca_constraint_desc in ca_output[joint_name]:
                if "position" in ca_constraint_desc.keys() and len(ca_constraint_desc["position"]) == 3:
                    if self.activate_coordinate_transform:
                        position = np.array([ca_constraint_desc["position"][0],ca_constraint_desc["position"][2],-ca_constraint_desc["position"][1]])
                    else:
                        position = np.array(ca_constraint_desc["position"])
                    ca_constraint = GlobalTransformCAConstraint(self.ea_generator.motion_state_graph.skeleton,
                                                                {"joint": joint_name, "canonical_keyframe": -1,
                                                                 "n_canonical_frames": n_canonical_frames,
                                                                 "position": position,
                                                                 "semanticAnnotation":  {"generated": True, "keyframeLabel": None},
                                                                 "ca_constraint": True},
                                                                1.0, 1.0, len(graph_walk.steps))
                    write_message_to_log("CREATE CA constraint "+ str(joint_name)+" " +str(ca_constraint.position), LOG_MODE_DEBUG)
                    ca_constraints.append(ca_constraint)
        return ca_constraints

    def _call_ca_rest_interface(self, ca_input):
            """ call ca rest interface using a json payload
            """
            if self.ca_service_url is not None:
                write_message_to_log("Call CA interface "+ str(self.ca_service_url,) + " for " + ca_input["elementary_action_name"]+ \
                          ca_input["motion_primitive_name"], LOG_MODE_DEBUG)
                request = urllib2.Request("http://" + self.ca_service_url, json.dumps(ca_input))
                try:
                    handler = urllib2.urlopen(request)
                    ca_output_string = handler.read()
                    ca_result = json.loads(ca_output_string)
                    return ca_result
                except urllib2.HTTPError, e:
                    write_message_to_log(str(e.code), LOG_MODE_ERROR)
                except urllib2.URLError, e:
                    write_message_to_log(str(e.args), LOG_MODE_ERROR)
            return None

    def _get_aligned_motion_spline(self, new_motion_spline, prev_frames):
        aligned_motion_spline = deepcopy(new_motion_spline)
        if prev_frames is not None:
            global_transformation = get_node_aligning_2d_transform(skeleton, node_name, prev_frames, new_motion_spline.coeffs)
            aligned_motion_spline.coeffs = transform_quaternion_frames(aligned_motion_spline.coeffs, global_transformation)
        elif self.ea_generator.action_constraints.start_pose is not None:
            global_transformation = get_transform_from_start_pose(self.ea_generator.action_constraints.start_pose)
            aligned_motion_spline.coeffs = transform_quaternion_frames(aligned_motion_spline.coeffs, global_transformation)
        else:
            global_transformation = np.eye(4, 4)
        return aligned_motion_spline, global_transformation

