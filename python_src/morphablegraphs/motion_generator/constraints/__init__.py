__author__ = 'erhe01'

OPTIMIZATION_MODE_ALL = "all"
OPTIMIZATION_MODE_KEYFRAMES = "keyframes"
OPTIMIZATION_MODE_NONE = "none"
LEFT_HAND_JOINT = "LeftToolEndSite"
RIGHT_HAND_JOINT = "RightToolEndSite"
CA_CONSTRAINTS_MODE_NONE = "none"
CA_CONSTRAINTS_MODE_INDIVIDUAL = "individual"
CA_CONSTRAINTS_MODE_SET = "create_constraint_set"

from mg_input_format_reader import MGInputFormatReader
from elementary_action_constraints_builder import ElementaryActionConstraintsBuilder