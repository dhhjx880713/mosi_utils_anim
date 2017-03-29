#import sys
#sys.path.append('..')
from io_helper_functions import load_json_file, get_bvh_writer, write_to_json_file, write_to_logfile, export_frames_to_bvh_file
from log import write_log, save_log, clear_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR, set_log_mode
from tcp_client import TCPClient