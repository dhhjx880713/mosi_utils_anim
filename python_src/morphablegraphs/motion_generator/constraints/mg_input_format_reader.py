__author__ = 'erhe01'

import numpy as np
import json
from ...utilities.log import write_log


class MGInputFormatReader(object):
    """Implements functions used for the processing of the constraints from the input file
    generated by CNL processing.

    Parameters
    ----------
    * mg_input_file : file path or json data read from a file
        Contains elementary action list with constraints, start pose and keyframe annotations.
    """

    constraint_types = ["keyframeConstraints", "directionConstraints"]

    def __init__(self, mg_input_file, activate_joint_mapping=False, activate_coordinate_transform=True):
        self.mg_input_file = mg_input_file
        self.elementary_action_list = list()
        self.keyframe_annotations = list()
        self.joint_name_map = dict()
        self.inverse_joint_name_map = dict()
        self._fill_joint_name_map()
        self.activate_joint_mapping = activate_joint_mapping
        self.activate_coordinate_transform = activate_coordinate_transform
        if type(mg_input_file) != dict:
            mg_input_file = open(mg_input_file)
            input_string = mg_input_file.read()
            if self.activate_joint_mapping:
                input_string = self._apply_joint_mapping_on_string(input_string)
            self.mg_input_file = json.loads(input_string)
            #self.mg_input_file = load_json_file(mg_input_file)
        else:
            if self.activate_joint_mapping:
                input_string = self._apply_joint_mapping_on_string(json.dumps(mg_input_file))
                self.mg_input_file = json.loads(input_string)
            else:
                self.mg_input_file = mg_input_file

        self._extract_elementary_actions()

    def _fill_joint_name_map(self):
        self.joint_name_map["RightHand"] = "RightToolEndSite"
        self.joint_name_map["LeftHand"] = "LeftToolEndSite"
        self.inverse_joint_name_map["RightToolEndSite"] = "RightHand"
        self.inverse_joint_name_map["LeftToolEndSite"] = "LeftHand"

    def _apply_joint_mapping_on_string(self, input_string):
        for key in self.joint_name_map.keys():
            input_string = input_string.replace(key, self.joint_name_map[key])
        return input_string

    def _extract_elementary_actions(self):
        if "elementaryActions" in self.mg_input_file.keys():
            self.elementary_action_list = self.mg_input_file["elementaryActions"]
        elif "tasks" in self.mg_input_file.keys():
            self.elementary_action_list = []
            for task in self.mg_input_file["tasks"]:
                if "elementaryActions" in task.keys():
                    self.elementary_action_list += task["elementaryActions"]
        self.keyframe_annotations = self._extract_keyframe_annotations()

    def get_number_of_actions(self):
        return len(self.elementary_action_list)

    def get_start_pose(self):
        start_pose = dict()
        if None in self.mg_input_file["startPose"]["orientation"]:
            start_pose["orientation"] = None
        else:
            start_pose["orientation"] = self._transform_point_from_cad_to_opengl_cs(self.mg_input_file["startPose"]["orientation"])
        start_pose["position"] = self._transform_point_from_cad_to_opengl_cs(self.mg_input_file["startPose"]["position"])
        return start_pose

    def get_elementary_action_name(self, action_index):
        return self.elementary_action_list[action_index]["action"]

    def get_ordered_keyframe_constraints(self, action_index, node_group):
        """
        Returns
        -------
            reordered_constraints: dict of lists
            dict of constraints lists applicable to a specific motion primitive of the node_group
        """
        keyframe_constraints = self._extract_all_keyframe_constraints(self.elementary_action_list[action_index]["constraints"], node_group)
        return self._reorder_keyframe_constraints_by_motion_primitive_name(node_group, keyframe_constraints)

    def _extract_control_points_from_trajectory_constraint_definition(self, trajectory_constraint_desc, scale_factor=1.0, distance_threshold=0.0):
        control_points = list()
        previous_point = None
        n_control_points = len(trajectory_constraint_desc)
        if "semanticAnnotation" in trajectory_constraint_desc[0].keys():
            active_region = dict()
            active_region["start_point"] = None
            active_region["end_point"] = None
        else:
            active_region = None

        last_distance = None
        for i in xrange(n_control_points):
            if trajectory_constraint_desc[i]["position"] == [None, None, None]:
                write_log("Warning: skip undefined control point")
                continue

            #where the a component of the position is set None it is set it to 0 to allow a 3D spline definition
            point = [p*scale_factor if p is not None else 0 for p in trajectory_constraint_desc[i]["position"]]
            point = np.asarray(self._transform_point_from_cad_to_opengl_cs(point))

            if previous_point is not None:
                distance = np.linalg.norm(point-previous_point)
            else:
                distance = None
            #add the point if there is no distance threshold, it is the first point, it is the last point or larger than or equal to the distance threshold
            if active_region is not None or (distance_threshold <= 0.0 or
                                             previous_point is None or
                                             np.linalg.norm(point-previous_point) >= distance_threshold):
                if last_distance is None or distance >= last_distance/10.0:
                    control_points.append(point)
                    last_distance = distance
            elif i == n_control_points-1:
                last_added_point_idx = len(control_points)-1
                if np.linalg.norm(control_points[last_added_point_idx] - point) < distance_threshold:
                    control_points[last_added_point_idx] = (control_points[last_added_point_idx] - point) * 1.00 + control_points[last_added_point_idx]
                    write_log("Warning: shift second to last control point because it is too close to the last control point")
                control_points.append(point)

            #set active region if it is a collision avoidance trajectory
            if active_region is not None and "semanticAnnotation" in trajectory_constraint_desc[i].keys():
                active = trajectory_constraint_desc[i]["semanticAnnotation"]["collisionAvoidance"]
                if active and active_region["start_point"] is None:
                    #print "set start", point
                    active_region["start_point"] = point
                elif not active and active_region["start_point"] is not None and active_region["end_point"] is None:
                    #print "set end", point
                    active_region["end_point"] = point

            previous_point = point

        #handle invalid region specification
        if active_region is not None:
            if active_region["start_point"] is None:
                active_region["start_point"] = control_points[0]
            if active_region["end_point"] is None:
                active_region["end_point"] = control_points[-1]
        #print "loaded", len(control_points), "points"
        return control_points, active_region

    def extract_trajectory_desc(self, action_index, joint_name, scale_factor=1.0, distance_threshold=-1):
        """ Extract the trajectory information from the constraint list
        Returns:
        -------
        * control_points: list of dict
            Constraint definition that contains a list of control points.
        * unconstrained_indices : list
        \t List of indices of unconstrained dimensions
        """
        trajectory_constraint_desc = self._extract_trajectory_constraint_desc(self.elementary_action_list[action_index]["constraints"], joint_name)
        if trajectory_constraint_desc is not None:
            #extract unconstrained dimensions
            unconstrained_indices = list()
            idx = 0
            for v in trajectory_constraint_desc[0]["position"]:
                if v is None:
                    unconstrained_indices.append(idx)
                idx += 1
            unconstrained_indices = self._transform_unconstrained_indices_from_cad_to_opengl_cs(unconstrained_indices)
            control_points, active_region = self._extract_control_points_from_trajectory_constraint_definition(trajectory_constraint_desc, distance_threshold=distance_threshold)
            #print control_points
            #active_region = self._check_for_collision_avoidance_annotation(trajectory_constraint_desc, control_points)

            return control_points, unconstrained_indices, active_region
        return None, None, False

    def _check_for_collision_avoidance_annotation(self, trajectory_constraint_desc, control_points):
        """ find start and end control point of an active region if there exists one.
        Note this functions expects that there are not more than one active region.

        :param trajectory_constraint_desc:
        :param control_points:
        :return: dict containing "start_point" and "end_point" or None
        """
        assert len(trajectory_constraint_desc) == len(control_points), str(len(trajectory_constraint_desc)) +" != " +  str(  len(control_points))
        active_region = None
        if "semanticAnnotation" in trajectory_constraint_desc[0].keys():
            active_region = dict()
            active_region["start_point"] = None
            active_region["end_point"] = None
            c_index = 0
            for c in trajectory_constraint_desc:
                if "semanticAnnotation" in c.keys():
                    if c["semanticAnnotation"]["collisionAvoidance"]:
                        active_region["start_point"] = control_points[c_index]
                    elif active_region["start_point"] is not None and active_region["end_point"] is None:
                        active_region["end_point"] = control_points[c_index]
                        break
                c_index += 1
        return active_region

    def _extract_keyframe_annotations(self):
        """
        Returns
        ------
        * keyframe_annotations : a list of dicts
          Contains for every elementary action a dict that associates of events/actions with certain keyframes
        """
        keyframe_annotations = []
        for action_index, entry in enumerate(self.elementary_action_list):
            keyframe_annotations.append(self.get_keyframe_annotations(action_index))
        return keyframe_annotations

    def get_keyframe_annotations(self, action_index):
        """
        Returns
        ------
        * keyframe_annotations : a list of dicts
          Contains for every elementary action a dict that associates of events/actions with certain keyframes
        """
        annotations = dict()
        if "keyframeAnnotations" in self.elementary_action_list[action_index].keys():
            for annotation in self.elementary_action_list[action_index]["keyframeAnnotations"]:
                keyframe_label = annotation["keyframe"]
                annotations[keyframe_label] = annotation
        return annotations

    def _reorder_keyframe_constraints_by_motion_primitive_name(self, node_group, keyframe_constraints):
        """ Order constraints extracted by _extract_all_keyframe_constraints for each state
        Returns
        -------
        reordered_constraints: dict of lists
        """
        reordered_constraints = dict()
        # iterate over keyframe labels
        for keyframe_label in keyframe_constraints.keys():
            motion_primitive_name = node_group.label_to_motion_primitive_map[keyframe_label]
            time_information = node_group.motion_primitive_annotations[motion_primitive_name][keyframe_label]
            reordered_constraints[motion_primitive_name] = list()
            # iterate over joints constrained at that keyframe
            for joint_name in keyframe_constraints[keyframe_label].keys():
                # iterate over constraints for that joint
                for constraint_type in self.constraint_types:
                    if constraint_type in keyframe_constraints[keyframe_label][joint_name].keys():
                        for constraint_definition in keyframe_constraints[keyframe_label][joint_name][constraint_type]:
                            # create constraint definition usable by the algorithm
                            # and add it to the list of constraints for that state
                            constraint_desc = self._create_keyframe_constraint(keyframe_label, joint_name, constraint_definition, time_information)
                            reordered_constraints[motion_primitive_name].append(constraint_desc)
        return reordered_constraints

    def _extract_constraints_for_keyframe_label(self, input_constraint_list, label):
        """ Returns the constraints associated with the given label. Ordered
            based on joint names.
        Returns
        ------
        * key_constraints : dict of lists
        \t contains the list of the constrained joints
        """
        key_constraints = dict()
        for joint_constraints in input_constraint_list:
            joint_name = joint_constraints["joint"]
            key_constraints[joint_name] = dict()
            for constraint_type in self.constraint_types:
                if constraint_type in joint_constraints.keys():
                    key_constraints[joint_name][constraint_type] = self.filter_constraint_definitions_by_label(joint_constraints[constraint_type], label)
        return key_constraints

    def filter_constraint_definitions_by_label(self, constraint_definitions, label):
        keyframe_constraint_definitions = list()
        for constraint_definition in constraint_definitions:
            #print "read constraint", constraint_definition, joint_name
            if self._constraint_definition_has_label(constraint_definition, label):
                keyframe_constraint_definitions.append(constraint_definition)
        return keyframe_constraint_definitions

    def _extract_all_keyframe_constraints(self, constraint_list, node_group):
        """Orders the keyframe constraint for the labels found in the metainformation of
           the elementary actions based on labels as keys
        Returns
        -------
        * keyframe_constraints : dict of dict of lists
          Lists of constraints for each motion primitive in the subgraph.
          access as keyframe_constraints["label"]["joint"][index]
        """
        keyframe_constraints = dict()
        annotations = node_group.label_to_motion_primitive_map.keys()
        for label in annotations:
            keyframe_constraints[label] = self._extract_constraints_for_keyframe_label(constraint_list, label)
        return keyframe_constraints

    def _extract_trajectory_constraint_desc(self, input_constraint_list, joint_name):
        """Returns a single trajectory constraint definition for joint joint out of a elementary action constraint list
        """
        for c in input_constraint_list:
            if joint_name == c["joint"]:
                if "trajectoryConstraints" in c.keys():
                    return c["trajectoryConstraints"]
        return None

    def _create_keyframe_constraint(self, keyframe_label, joint_name, constraint, time_information):
        """ Creates a dict containing all properties stated explicitly or implicitly in the input constraint
        Parameters
        ----------
        * keyframe_label : string
          keyframe label
        * joint_name : string
          Name of the joint
        * constraint : dict
          Read from json input file
        * time_information : string
          Time information corresponding to an annotation read from morphable graph meta information

         Returns
         -------
         *constraint_desc : dict
          Contains the keys joint, position, orientation, time, semanticAnnotation
        """
        position = [None, None, None]
        orientation = [None, None, None]
        first_frame = None
        last_frame = None
        time = None
        if "position" in constraint.keys():
            position = constraint["position"]
        if "orientation" in constraint.keys():
            orientation = constraint["orientation"]
        if "time" in constraint.keys():
            time = constraint["time"]
        #check if last or fist frame from annotation
        position = self._transform_point_from_cad_to_opengl_cs(position)
        if time_information == "lastFrame":
            last_frame = True
        elif time_information == "firstFrame":
            first_frame = True
        if "semanticAnnotation" in constraint.keys():
            semantic_annotation = constraint["semanticAnnotation"]
        else:
            semantic_annotation = {}
        semantic_annotation["firstFrame"] = first_frame
        semantic_annotation["lastFrame"] = last_frame
        semantic_annotation["keyframeLabel"] = keyframe_label
        constraint_desc = {"joint": joint_name,
                           "position": position,
                           "orientation": orientation,
                           "time": time,
                           "semanticAnnotation": semantic_annotation}
        return constraint_desc

    def _constraint_definition_has_label(self, constraint_definition, label):
        """ Checks if the label is in the semantic annotation dict of a constraint
        """
        if "semanticAnnotation" in constraint_definition.keys():
            annotation = constraint_definition["semanticAnnotation"]
            #print "semantic Annotation",annotation
            if label in annotation.keys():
                return True
        return False

    def _transform_point_from_cad_to_opengl_cs(self, point):
        """ Transforms a 3d point represented as a list from a CAD to a
            opengl coordinate system by a -90 degree rotation around the x axis
        """
        if self.activate_coordinate_transform:
            transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            return np.dot(transform_matrix, point).tolist()
        else:
            return point

    def _transform_unconstrained_indices_from_cad_to_opengl_cs(self, indices):
        """ Transforms a list indicating unconstrained dimensions from cad to opengl
            coordinate system.
        """
        if self.activate_coordinate_transform:
            new_indices = []
            for i in indices:
                if i == 0:
                    new_indices.append(0)
                elif i == 1:
                    new_indices.append(2)
                elif i == 2:
                    new_indices.append(1)
            return new_indices
        else:
            return indices

    def inverse_map_joint(self, joint_name):
        if joint_name in self.inverse_joint_name_map.keys() and self.activate_joint_mapping:
            #print "map joint", joint_name
            return self.inverse_joint_name_map[joint_name]
        else:
            return joint_name