import numpy as np

CONSTRAINT_TYPES = ["keyframeConstraints", "directionConstraints"]

def _transform_point_from_cad_to_opengl_cs(point, activate_coordinate_transform=False):
    """ Transforms a 3D point represented as a list from a CAD to a
        opengl coordinate system by a -90 degree rotation around the x axis
    """
    if not activate_coordinate_transform:
        return point
    transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    return np.dot(transform_matrix, point).tolist()


def _transform_unconstrained_indices_from_cad_to_opengl_cs(indices, activate_coordinate_transform=False):
    """ Transforms a list indicating unconstrained dimensions from cad to opengl
        coordinate system.
    """
    if not activate_coordinate_transform:
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

