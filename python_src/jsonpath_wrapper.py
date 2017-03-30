""" Simple JSONPath search library because existing libaries do not support changing values.

    Some functions are copied from the following sources:
    https://github.com/kennknowles/python-jsonpath-rw/issues/2
    http://stackoverflow.com/questions/2103071/determine-the-type-of-a-value-which-is-represented-as-string-in-python
    http://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
"""


from morphablegraphs.utilities import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_ERROR, set_log_mode
import distutils
import re

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
        data[first] = update_json(data[first], path, value)
        return data
    except StopIteration:
        return value


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


def get_path_from_string(path_str):
    temp_path_list = path_str.split(".")[1:]
    path_list = []
    for idx, key in enumerate(temp_path_list):
        match = re.search("[-?\d+]", key)
        if (match):
            span = match.span()
            index = int(key[span[0]:span[1]])
            key = key[:span[0] - 1]
            path_list.append(key)
            path_list.append(index)
        else:
            path_list.append(key)
    return path_list


def search_for_path(data, path_str):
    path = get_path_from_string(path_str)
    current = data
    for key in path:
        try:
            current = current[key]
        except:
            return None
    return current


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
        match = search_for_path(data, path_str)
        if match is not None:
            before = match
            path_list = iter(get_path_from_string(path_str))
            update_json(data, path_list, value)
            match = search_for_path(data, path_str)
            message = "set value of " + path_str + " from " + str(before) + " to " + str(match) + " with " + str(type(match))
            write_message_to_log(message, LOG_MODE_DEBUG)
        else:
            write_message_to_log("Warning: Did not find JSONPath " + path_str + " in data", LOG_MODE_ERROR)


if __name__ == "__main__":
    test_data = {
        "model_data": "motion_primitives_quaternion_PCA95_unity-integration-1.0.0",
        "port": 8888,
        "write_log": True,
        "log_level": 1,
        "list_test": [{"ab":1},{"ab":3}]
    }

    import argparse
    set_log_mode(LOG_MODE_DEBUG)
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("-set", nargs='+', default=[], help="JSONPath expression")
    args = parser.parse_args()

    update_data_from_jsonpath(test_data, args.set)