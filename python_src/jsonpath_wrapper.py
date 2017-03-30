""" Wrapper around the jsonpath-rw library which does not support changing values in version 1.3.0.

    Some functions are copied from the following sources:
    https://github.com/kennknowles/python-jsonpath-rw/issues/2
    http://stackoverflow.com/questions/2103071/determine-the-type-of-a-value-which-is-represented-as-string-in-python
    http://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
"""

from jsonpath_rw import jsonpath, parse
from morphablegraphs.utilities import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_ERROR, set_log_mode
import distutils


TYPE_CONVERTER = {
    "int": int,
    "float": float,
    "bool": lambda value: bool(distutils.util.strtobool(value)),
    "str": str
}


def update_json(data, path, value):
    """Update JSON dictionnary PATH with VALUE. Return updated JSON"""
    try:
        first = next(path)
        # check if item is an array
        if first.startswith('[') and first.endswith(']'):
            try:
                first = int(first[1:-1])
            except ValueError:
                pass
        data[first] = update_json(data[first], path, value)
        return data
    except StopIteration:
        return value


def get_path(match):
    """return an iterator based upon MATCH.PATH. Each item is a path component,
        start from outer most item.
    """
    if match.context is not None:
        for path_element in get_path(match.context):
            yield path_element
        yield str(match.path)


def get_type_of_string(value):
    for type, test in TYPE_CONVERTER.items():
        try:
            v = test(value)
            if value == str(v):
                return type
        except ValueError:
            continue
    # No match
    return "str"


def update_data_from_jsonpath(data, expressions, split_str="="):
    """Takes a dictionary and a list of expressions in the form JSONPath=value, e.g. "$.write_log=True".
        Expressions and values should not contain the split_str="="
    """
    for expr in expressions:
        expr_t = expr.split(split_str)
        path_str = expr_t[0]
        value = expr_t[1]
        value_type = get_type_of_string(value)
        value = TYPE_CONVERTER[value_type](value)
        path = parse(path_str)
        matches = path.find(data)
        if len(matches) > 0:
            match = matches[0]
            before = match.value
            update_json(data, get_path(match), value)
            match = path.find(data)[0]
            message = "set value of " + path_str + " from " + str(before) + " to " + str(match.value) + " with " + str(type(match.value))
            write_message_to_log(message, LOG_MODE_DEBUG)
        else:
            write_message_to_log("Warning: Did not find JSONPath " + path_str + " in data", LOG_MODE_ERROR)


if __name__ == "__main__":
    test_data = {
        "model_data": "motion_primitives_quaternion_PCA95_unity-integration-1.0.0",
        "port": 8888,
        "write_log": True,
        "log_level": 1
    }

    import argparse
    set_log_mode(LOG_MODE_DEBUG)
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("-set", nargs='+', default=[], help="JSONPath expression")
    args = parser.parse_args()

    update_data_from_jsonpath(test_data, args.set)
