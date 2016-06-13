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
        self.scale_factor = 1.0
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
        #TODO: read from file
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


    def _is_active_trajectory_region(self, traj_constraint, index):
        if "semanticAnnotation" in traj_constraint[index].keys():
            return traj_constraint[index]["semanticAnnotation"]["collisionAvoidance"]
        else:
            return True

    def _extract_trajectory_control_points(self, traj_constraint, distance_threshold=0.0):
        control_point_list = list()
        active_regions = list()
        previous_point = None
        n_control_points = len(traj_constraint)
        was_active = False
        last_distance = None
        count = -1
        for idx in xrange(n_control_points):
            is_active = self._is_active_trajectory_region(traj_constraint, idx)
            if not is_active:
                was_active = is_active
                continue
            if not was_active and is_active:
                active_region = self._init_active_region(traj_constraint)
                control_point_list.append(list())
                active_regions.append(active_region)
                count += 1
            if count < 0:
                continue
            tmp_distance_threshold = distance_threshold
            if active_regions[count] is not None:
                tmp_distance_threshold = -1
            result = self._extract_control_point(traj_constraint, n_control_points, idx, previous_point, last_distance, tmp_distance_threshold)
            if result is None:
                continue
            else:
                point, last_distance = result
                n_points = len(control_point_list[count])
                if idx == n_control_points-1:
                    last_added_point_idx = n_points-1
                    delta = control_point_list[count][last_added_point_idx] - point
                    if np.linalg.norm(delta) < distance_threshold:
                        control_point_list[count][last_added_point_idx] += delta
                        write_log("Warning: shift second to last control point because it is too close to the last control point")
                control_point_list[count].append(point)

                if active_regions[count] is not None:
                    self._update_active_region(active_regions[count], point, is_active)
                previous_point = point
                was_active = is_active

        #handle invalid region specification
        region_points = list()
        for idx in range(len(control_point_list)):
            region_points.append(len(control_point_list[idx]))
            if active_regions[idx] is not None:
                if len(control_point_list[idx]) < 2:
                    control_point_list[idx] = None
                else:
                    self._end_active_region(active_regions[idx], control_point_list[idx])
        print "loaded", len(control_point_list),"active regions with",region_points,"points"
        return control_point_list, active_regions

    def _init_active_region(self,traj_constraint):
        if "semanticAnnotation" in traj_constraint[0].keys():
            active_region = dict()
            active_region["start_point"] = None
            active_region["end_point"] = None
            return active_region
        else:
            return None

    def _end_active_region(self, active_region, control_points):
        if active_region["start_point"] is None:
            active_region["start_point"] = control_points[0]
        if active_region["end_point"] is None:
            active_region["end_point"] = control_points[-1]

    def _update_active_region(self, active_region, point, new_active):
         #set active region if it is a collision avoidance trajectory
        if new_active and active_region["start_point"] is None:
            #print "set start", point
            active_region["start_point"] = point
        elif not new_active and active_region["start_point"] is not None and active_region["end_point"] is None:
            #print "set end", point
            active_region["end_point"] = point


    def _extract_control_point(self, traj_constraint, n_control_points, index, previous_point, last_distance, distance_threshold):
        if traj_constraint[index]["position"] == [None, None, None]:
            write_log("Warning: skip undefined control point")
            return None
        #set component of the position to 0 where it is is set to None to allow a 3D spline definition
        point = [p*self.scale_factor if p is not None else 0 for p in traj_constraint[index]["position"]]
        point = np.asarray(self._transform_point_from_cad_to_opengl_cs(point))

        if previous_point is None or index == n_control_points-1:
            return point, last_distance
        else:
            distance = np.linalg.norm(point-previous_point)
            #add the point if there is no distance threshold, it is the first point, it is the last point or larger than or equal to the distance threshold
            if (distance_threshold <= 0.0 or np.linalg.norm(point-previous_point) >= distance_threshold) and (last_distance is None or distance >= last_distance/10.0):#'TODO' add toggle of filter to config
                return point, distance
            else:
                return None

    def extract_trajectory_desc(self, action_index, joint_name, distance_threshold=-1):
        """ Extract the trajectory information from the constraint list
        Returns:
        -------
        * control_points: list of dict
            Constraint definition that contains a list of control points.
        * unconstrained_indices : list
        \t List of indices of unconstrained dimensions
        """
        constraint_desc = self._extract_trajectory_constraint_data(self.elementary_action_list[action_index]["constraints"],
                                                                    joint_name)
        if constraint_desc is None:
            return [], None, []
        unconstrained_indices = self._find_unconstrained_indices(constraint_desc)
        control_point_list, active_regions = self._extract_trajectory_control_points(constraint_desc, distance_threshold)
        return control_point_list, unconstrained_indices, active_regions

    def _find_unconstrained_indices(self, trajectory_constraint_data):
        """extract unconstrained dimensions"""
        unconstrained_indices = list()
        idx = 0
        for v in trajectory_constraint_data[0]["position"]:
            if v is None:
                unconstrained_indices.append(idx)
            idx += 1
        return self._transform_unconstrained_indices_from_cad_to_opengl_cs(unconstrained_indices)

    def _check_for_collision_avoidance_annotation(self, trajectory_constraint_desc, control_points):
        """ find start and end control point of an active region if there exists one.
        Note this functions expects that there is not more than one active region.

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
            mp_name = node_group.label_to_motion_primitive_map[keyframe_label]
            time_info = node_group.motion_primitive_annotations[mp_name][keyframe_label]
            reordered_constraints[mp_name] = list()
            # iterate over joints constrained at that keyframe
            for joint_name in keyframe_constraints[keyframe_label].keys():
                # iterate over constraints for that joint
                for c_type in self.constraint_types:
                    reordered_constraints[mp_name] += self._filter_by_constraint_type(keyframe_constraints,keyframe_label, joint_name, time_info, c_type)
        return reordered_constraints

    def _filter_by_constraint_type(self, constraints,keyframe_label, joint_name, time_info, c_type):
        filtered_constraints = list()
        if c_type in constraints[keyframe_label][joint_name].keys():
            for constraint in constraints[keyframe_label][joint_name][c_type]:
                # create constraint definition usable by the algorithm
                # and add it to the list of constraints for that state
                constraint_desc = self._extend_keyframe_constraint_definition(keyframe_label, joint_name, constraint, time_info, c_type)
                filtered_constraints.append(constraint_desc)
        return filtered_constraints
    def filter_constraints_by_label(self, constraints, label):
        keyframe_constraints = []
        for constraint in constraints:
            if self._constraint_definition_has_label(constraint, label):
                keyframe_constraints.append(constraint)
        return keyframe_constraints

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

    def _extract_constraints_for_keyframe_label(self, input_constraint_list, label):
        """ Returns the constraints associated with the given label. Ordered
            based on joint names.
        Returns
        ------
        * key_constraints : dict of lists
        \t contains the list of the constrained joints
        """

        keyframe_constraints = dict()
        for joint_constraints in input_constraint_list:
            if "joint" in joint_constraints.keys():
                joint_name = joint_constraints["joint"]
                if joint_name not in keyframe_constraints:
                    keyframe_constraints[joint_name] = dict()
                for constraint_type in self.constraint_types:
                    if constraint_type not in keyframe_constraints[joint_name]:
                        keyframe_constraints[joint_name][constraint_type] = list()
                    if constraint_type in joint_constraints.keys():
                        filtered_list = self.filter_constraints_by_label(joint_constraints[constraint_type], label)
                        keyframe_constraints[joint_name][constraint_type] = filtered_list
        return keyframe_constraints

    def _extract_trajectory_constraint_data(self, input_constraint_list, joint_name):
        """Returns a single trajectory constraint definition for joint joint out of a elementary action constraint list
        """
        for c in input_constraint_list:
            if "joint" in c.keys() and "trajectoryConstraints" in c.keys() and joint_name == c["joint"]:
                return c["trajectoryConstraints"]
        return None

    def _extend_keyframe_constraint_definition(self, keyframe_label, joint_name, constraint, time_info, c_type):
        """ Creates a dict containing all properties stated explicitly or implicitly in the input constraint
        Parameters
        ----------
        * keyframe_label : string
          keyframe label
        * joint_name : string
          Name of the joint
        * constraint : dict
          Read from json input file
        * time_info : string
          Time information corresponding to an annotation read from morphable graph meta information

         Returns
         -------
         *constraint_desc : dict
          Contains the keys joint, position, orientation, time, semanticAnnotation
        """
        position = [None, None, None]
        orientation = [None, None, None]
        time = None
        if "position" in constraint.keys():
            position = constraint["position"]
        if "orientation" in constraint.keys():
            orientation = constraint["orientation"]
        if "time" in constraint.keys():
            time = constraint["time"]
        #check if last or fist frame from annotation
        position = self._transform_point_from_cad_to_opengl_cs(position)
        if "semanticAnnotation" in constraint.keys():
            semantic_annotation = constraint["semanticAnnotation"]
        else:
            semantic_annotation = dict()
        #self._add_legacy_constrained_gmm_info(time_info, semantic_annotation)
        semantic_annotation["keyframeLabel"] = keyframe_label
        constraint_desc = {"joint": joint_name,
                           "position": position,
                           "orientation": orientation,
                           "time": time,
                           "semanticAnnotation": semantic_annotation}

        if c_type == "directionConstraints":
            #position is interpreted as look at target
            #TODO improve integration of look at constraints
            constraint_desc["look_at"] = True
        return constraint_desc

    def _add_legacy_constrained_gmm_info(self, time_information, semantic_annotation):
        first_frame = None
        last_frame = None
        if time_information == "lastFrame":
            last_frame = True
        elif time_information == "firstFrame":
            first_frame = True
        semantic_annotation["firstFrame"] = first_frame
        semantic_annotation["lastFrame"] = last_frame

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
        """ Transforms a 3D point represented as a list from a CAD to a
            opengl coordinate system by a -90 degree rotation around the x axis
        """
        if not self.activate_coordinate_transform:
            return point
        transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        return np.dot(transform_matrix, point).tolist()

    def _transform_unconstrained_indices_from_cad_to_opengl_cs(self, indices):
        """ Transforms a list indicating unconstrained dimensions from cad to opengl
            coordinate system.
        """
        if not self.activate_coordinate_transform:
            return indices
        new_indices = []
        for i in indices:
            if i == 0:
                new_indices.append(0)
            elif i == 1:
                new_indices.append(2)
            elif i == 2:
                new_indices.append(1)
        return new_indices

    def inverse_map_joint(self, joint_name):
        if joint_name in self.inverse_joint_name_map.keys() and self.activate_joint_mapping:
            #print "map joint", joint_name
            return self.inverse_joint_name_map[joint_name]
        else:
            return joint_name
