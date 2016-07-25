import numpy as np
import json
from copy import deepcopy
import urllib2
from ..utilities import write_log, get_bvh_writer, TCPClient
from constraints.spatial_constraints.keyframe_constraints.global_transform_ca_constraint import GlobalTransformCAConstraint
from ..animation_data.motion_editing import fast_quat_frames_transformation, transform_quaternion_frames, euler_angles_to_rotation_matrix


class CAInterface(object):
    def __init__(self, ea_generator, service_config):
        self.ea_generator = ea_generator
        self.activate_coordinate_transform = service_config["activate_coordinate_transform"]
        self.ca_service_url = service_config["collision_avoidance_service_url"]
        self.ca_service_port = service_config["collision_avoidance_service_port"]
        self.tcp_client = TCPClient(self.ca_service_url, self.ca_service_port)
        self.coordinate_transform_matrix = np.array([[1, 0, 0, 0],
                                                     [0, 0, -1, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 1]])

    def get_constraints(self, groupd_id, new_node, new_motion_spline, graph_walk):
        """ Generate constraints using the rest interface of the collision avoidance module directly.
        """
        aligned_motion_spline, global_transformation = self._get_aligned_motion_spline(new_motion_spline,
                                                                                       graph_walk.get_quat_frames())
        if self.activate_coordinate_transform:
            global_transformation = np.dot(global_transformation, self.coordinate_transform_matrix)
        frames = aligned_motion_spline.get_motion_vector()
        global_bvh_string = get_bvh_writer(self.ea_generator.motion_state_graph.skeleton, frames).generate_bvh_string()
        ca_input ={"groupId": groupd_id, "command":"GenerateConstraints"}
        ca_input["parameters"] = {"elementary_action_name": new_node[0],
                    "motion_primitive_name": new_node[1],
                    "global_transform": global_transformation.tolist(),
                    "global_bvh_frames": global_bvh_string}

        # ca_output = self._call_ca_rest_interface(ca_input)
        message = json.dumps(ca_input)
        ca_output_string = self.tcp_client.send_message(message)
        ca_output = json.loads(ca_output_string)
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
                    print "CREATE CA constraint", joint_name, ca_constraint.position
                    ca_constraints.append(ca_constraint)
        return ca_constraints

    def _call_ca_rest_interface(self, ca_input):
            """ call ca rest interface using a json payload
            """
            if self.ca_service_url is not None:
                write_log("Call CA interface", self.ca_service_url, "for", ca_input["elementary_action_name"],
                          ca_input["motion_primitive_name"])
                request = urllib2.Request("http://" + self.ca_service_url, json.dumps(ca_input))
                try:
                    handler = urllib2.urlopen(request)
                    ca_output_string = handler.read()
                    ca_result = json.loads(ca_output_string)
                    return ca_result
                except urllib2.HTTPError, e:
                    write_log(e.code)
                except urllib2.URLError, e:
                    write_log(e.args)
            return None

    def _get_aligned_motion_spline(self, new_motion_spline, prev_frames):
        aligned_motion_spline = deepcopy(new_motion_spline)
        if prev_frames is not None:
            angle, offset = fast_quat_frames_transformation(prev_frames, new_motion_spline.coeffs)
            aligned_motion_spline.coeffs = transform_quaternion_frames(aligned_motion_spline.coeffs,
                                                                       [0, angle, 0], offset)
            global_transformation = euler_angles_to_rotation_matrix([0, angle, 0])
            global_transformation[:3, 3] = offset
        elif self.ea_generator.action_constraints.start_pose is not None:
            aligned_motion_spline.coeffs = transform_quaternion_frames(aligned_motion_spline.coeffs,
                                                                       self.ea_generator.action_constraints.start_pose[
                                                                           "orientation"],
                                                                       self.ea_generator.action_constraints.start_pose[
                                                                           "position"])
            global_transformation = euler_angles_to_rotation_matrix(
                self.ea_generator.action_constraints.start_pose["orientation"])
            global_transformation[:3, 3] = self.ea_generator.action_constraints.start_pose["position"]
        else:
            global_transformation = np.eye(4, 4)
        return aligned_motion_spline, global_transformation

