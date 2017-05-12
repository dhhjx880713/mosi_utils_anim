from parameterized_spline import ParameterizedSpline
import numpy as np
import math
from ....external.transformations import  quaternion_from_euler

REF_VECTOR = [0.0,1.0]


def get_tangent_at_parameter(spline, u, eval_range=0.5):
    """
    Returns
    ------
    * dir_vector : np.ndarray
      The normalized direction vector
    * start : np.ndarry
      start of the tangent line / the point evaluated at arc length
    """
    tangent = [1.0,0.0]
    magnitude = 0
    while magnitude == 0:  # handle cases where the granularity of the spline is too low
        l1 = u - eval_range
        l2 = u + eval_range
        p1 = spline.query_point_by_parameter(l1)
        p2 = spline.query_point_by_parameter(l2)
        tangent = p2 - p1
        magnitude = np.linalg.norm(tangent)
        eval_range += 0.1
        if magnitude != 0:
            tangent /= magnitude
    return tangent


def get_angle_between_vectors(a,b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    angle = math.acos((a[0] * b[0] + a[1] * b[1]))
    return angle


def get_tangents2d(translation, eval_range=0.5):
    """ Create a list of tangents for a list of translations to be used for an AnnotatedSpline"""
    """ TODO fix """
    steps = len(translation)
    spline = ParameterizedSpline(translation)
    parameters = np.linspace(0, 1, steps)# this is not correct
    tangents = []
    for u in parameters:
        tangent = get_tangent_at_parameter(spline, u, eval_range)
        tangents.append(tangent)
    return tangents


def complete_tangents(translation, given_tangents, eval_range=0.5):
    steps = len(translation)
    spline = ParameterizedSpline(translation)
    parameters = np.linspace(0, 1, steps)
    tangents = given_tangents
    for idx, u in enumerate(parameters):
        if tangents[idx] is None:
            tangents[idx] = get_tangent_at_parameter(spline, u, eval_range)
    return tangents


def complete_orientations_from_tangents(translation, given_orientations, eval_range=0.5, ref_vector=REF_VECTOR):
    steps = len(translation)
    spline = ParameterizedSpline(translation)
    parameters = np.linspace(0, 1, steps)
    orientations = given_orientations
    for idx, u in enumerate(parameters):
        if orientations[idx] is None:
            tangent = get_tangent_at_parameter(spline, u, eval_range)
            print "estimate tangent",idx, tangent
            orientations[idx] = tangent_to_quaternion(tangent, ref_vector)
    return orientations


def tangent_to_quaternion(tangent, ref_vector=REF_VECTOR):
    a = ref_vector
    b = np.array([tangent[0], tangent[2]])
    angle = get_angle_between_vectors(a, b)
    return quaternion_from_euler(0, angle, 0)


def tangents_to_quaternions(tangents, ref_vector=REF_VECTOR):
    quaternions = []
    for tangent in tangents:
        q = tangent_to_quaternion(tangent, ref_vector)
        quaternions.append(q)
    return quaternions


def get_orientations_from_tangents2d(translation, ref_vector=REF_VECTOR):
    """ Create a list of orientations for a list of translations to be used for an AnnotatedSpline.
        Note it seems that as long as the number of points are the same, the same spline parameters can be used for the
        query of the spline.
    """
    """ TODO fix """
    ref_vector = np.array(ref_vector)
    steps = len(translation)
    spline = ParameterizedSpline(translation)
    parameters = np.linspace(0,1, steps)
    orientation = []
    for u in parameters:
        tangent = get_tangent_at_parameter(spline, u, eval_range=0.1)
        a = ref_vector
        b = np.array([tangent[0], tangent[2]])
        angle = get_angle_between_vectors(a, b)
        orientation.append(quaternion_from_euler(*np.radians([0, angle, 0])))
    return orientation

