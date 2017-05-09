import numpy as np
from ...utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_ERROR, LOG_MODE_INFO
from utils import _transform_point_from_cad_to_opengl_cs, _transform_unconstrained_indices_from_cad_to_opengl_cs


class TrajectoryConstraintReader(object):
    def __init__(self, activate_coordinate_transform=True, scale_factor=1.0):
        self.activate_coordinate_transform = activate_coordinate_transform
        self.scale_factor = scale_factor

    def _is_active_trajectory_region(self, traj_constraint, index):
        if "semanticAnnotation" in traj_constraint[index].keys():
            if "collisionAvoidance" in traj_constraint[index]["semanticAnnotation"].keys():
                return traj_constraint[index]["semanticAnnotation"]["collisionAvoidance"]
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
            result = self._extract_control_point(traj_constraint, n_control_points, idx, previous_point,
                                                 last_distance, tmp_distance_threshold)
            if result is None:
                continue
            else:
                point, last_distance = result
                n_points = len(control_point_list[count])
                if idx == n_control_points - 1:
                    last_added_point_idx = n_points - 1
                    delta = control_point_list[count][last_added_point_idx] - point
                    if np.linalg.norm(delta) < distance_threshold:
                        control_point_list[count][last_added_point_idx] += delta
                        write_log(
                            "Warning: shift second to last control point because it is too close to the last control point")
                control_point_list[count].append(point)

                if active_regions[count] is not None:
                    self._update_active_region(active_regions[count], point, is_active)
                previous_point = point
                was_active = is_active

        # handle invalid region specification
        region_points = list()
        for idx in range(len(control_point_list)):
            region_points.append(len(control_point_list[idx]))
            if active_regions[idx] is not None:
                if len(control_point_list[idx]) < 2:
                    control_point_list[idx] = None
                else:
                    self._end_active_region(active_regions[idx], control_point_list[idx])
        # print "loaded", len(control_point_list),"active regions with",region_points,"points"
        return control_point_list, active_regions

    def _init_active_region(self, traj_constraint):
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
        if new_active and active_region["start_point"] is None:
            active_region["start_point"] = point
        elif not new_active and active_region["start_point"] is not None and active_region["end_point"] is None:
            active_region["end_point"] = point

    def _extract_control_point(self, traj_constraint, n_control_points, index, previous_point, last_distance,
                               distance_threshold):
        if "position" not in traj_constraint[index].keys() or traj_constraint[index]["position"] == [None, None,
                                                                                                     None]:
            write_log("Warning: skip undefined control point")
            return None
        # set component of the position to 0 where it is is set to None to allow a 3D spline definition
        point = [p * self.scale_factor if p is not None else 0 for p in traj_constraint[index]["position"]]
        point = np.asarray(_transform_point_from_cad_to_opengl_cs(point, self.activate_coordinate_transform))

        if previous_point is None or index == n_control_points - 1:
            return point, last_distance
        else:
            distance = np.linalg.norm(point - previous_point)
            # add the point if there is no distance threshold, it is the first point, it is the last point or larger than or equal to the distance threshold
            if distance_threshold > 0.0 and distance < distance_threshold:
                return None
            if last_distance is not None and distance < last_distance / 10.0:  # TODO add toggle of filter to config
                return None
            return point, distance

    def extract_trajectory_desc(self, elementary_action_list, action_index, joint_name, distance_threshold=-1):
        """ Extract the trajectory information from the constraint list
        Returns:
        -------
        * desc : dict
        \tConstraint definition that contains a list of control points, unconstrained_indices, active_regions and a possible
        annotation.
        """
        desc = dict()
        desc["control_points_list"] = []
        desc["orientation_list"] = []
        desc["unconstrained_indices"] = None
        desc["active_regions"] = []
        desc["semantic_annotation"] = None
        constraint_data = None
        for c in elementary_action_list[action_index]["constraints"]:
            if "joint" in c.keys() and "trajectoryConstraints" in c.keys() and joint_name == c["joint"]:
                constraint_data = c["trajectoryConstraints"]
                break  # there should only be one list per joint and elementary action
        if constraint_data is not None:
            for p in constraint_data:
                if "semanticAnnotation" in p.keys() and not "collisionAvoidance" in p["semanticAnnotation"].keys():
                    desc["semantic_annotation"] = p["semanticAnnotation"]
                    break
            desc["unconstrained_indices"] = self._find_unconstrained_indices(constraint_data)
            desc["control_points_list"], desc["active_regions"] = self._extract_trajectory_control_points(
                constraint_data, distance_threshold)
        return desc

    def _find_unconstrained_indices(self, trajectory_constraint_data):
        """extract unconstrained dimensions"""
        unconstrained_indices = list()
        idx = 0
        for p in trajectory_constraint_data:
            if ["position"] in p.keys():
                for v in p["position"]:
                    if v is None:
                        unconstrained_indices.append(idx)
                    idx += 1
                break # check only one point
        return _transform_unconstrained_indices_from_cad_to_opengl_cs(unconstrained_indices, self.activate_coordinate_transform)

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
