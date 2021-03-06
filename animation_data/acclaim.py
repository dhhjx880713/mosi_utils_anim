""" parser of asf and amc formats
    https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html
 """  
import sys
import math
import numpy as np
from transformations import euler_matrix

def rotate_around_x(alpha):
    #Note vectors represent columns
    cx = math.cos(alpha)
    sx = math.sin(alpha)
    m = np.array([[1.0 , 0.0, 0.0, 0.0],
                             [0.0, cx ,  sx,0.0],
                              [0.0, -sx,  cx,0.0],
                             [0.0,0.0,0.0,1.0]],np.float32)
    return m.T


def rotate_around_y(beta):
    #Note vectors represent columns
    cy = math.cos(beta)
    sy = math.sin(beta)
    m = np.array([[ cy,0.0,-sy ,0.0],
                                  [0.0,1.0,0.0,0.0],
                                  [ sy,0.0,cy,0.0],
                                   [0.0,0.0,0.0,1.0]],np.float32)
    return m.T


def rotate_around_z(gamma):
    #Note vectors represent columns
    cz = math.cos(gamma)
    sz = math.sin(gamma)
    m = np.array([[ cz, sz,0.0,0.0],
                    [ -sz, cz,0.0,0.0],
                    [0.0,0.0,1.0,0.0],
                    [0.0,0.0,0.0,1.0]],np.float32)
    return m.T


AXES = "rxyz"
def create_euler_matrix2(angles, order):
    m = np.eye(4)
    for idx, d in enumerate(order):
        a = np.radians(angles[idx])
        d = d[-1].lower()
        local_rot = np.eye(4)
        if d =="x":
            local_rot = euler_matrix(a,0,0, AXES)
        elif d =="y":
            local_rot = euler_matrix(0,a,0, AXES)
        elif d =="z":
            local_rot = euler_matrix(0,0,a, AXES)
        m = np.dot(local_rot, m)
    return m

def create_euler_matrix(angles, order):
    m = np.eye(4)
    for idx, d in enumerate(order):
        a = np.radians(angles[idx])
        d = d[-1].lower()
        local_rot = np.eye(4)
        if d =="x":
            local_rot = rotate_around_x(a)
        elif d =="y":
            local_rot = rotate_around_y(a)
        elif d =="z":
            local_rot = rotate_around_z(a)
        m = np.dot(local_rot, m)
    return m

def create_c_matrices(data):
    angles = data["root"]["orientation"]
    order = data["root"]["axis"]
    C = create_euler_matrix(angles, order)
    data["root"]["C"] = C
    data["root"]["Cinv"] = np.linalg.inv(C)

    for key in data["bones"]:
        if "axis" in data["bones"][key]:
            angles, order = data["bones"][key]["axis"]
            data["bones"][key]["C"] = create_euler_matrix(angles, order)
            data["bones"][key]["Cinv"] =np.linalg.inv(data["bones"][key]["C"] ) #create_euler_matrix(-np.array(angles), order)
       
    return data

def set_parents(data):
    for key in data["bones"]:
        data["bones"][key]["parent"] = "root"
        
    for key in data["bones"]:
        if key in data["children"]:
            for c_key in data["children"][key]:
                data["bones"][c_key]["parent"] = key
    return data
    
def parse_asf_file(filepath):
    with open(filepath, "rb") as in_file:
        lines = in_file.readlines()

    #lines = list(map(str, lines))
    lines = [str(l, "utf-8") for l in lines]
    data = dict()
    data["bones"] = dict()
    idx = 0
    print("read", idx)
    while idx < len(lines):
        print("read", idx, lines[idx].lstrip())
        next_line = lines[idx].lstrip()
        if next_line.startswith(":root"):
            idx += 1
            data["root"], idx = read_root_data(lines, idx)
            idx -=1
        if next_line.startswith(":name"):
            data["name"] = next_line.split(" ")[1]
            idx+=1
        elif next_line.startswith(":bonedata"):
            print("found bones")
            idx+=1
            next_line = lines[idx].lstrip()
            while not next_line.startswith(":hierarchy") and idx+1 < len(lines):
                node, idx = read_bone_data(lines, idx)
                if "name" in node:
                    name = node["name"]
                    data["bones"][name] = node
                if idx < len(lines):
                    next_line = lines[idx].lstrip()
        elif next_line.startswith(":hierarchy"):
            data["children"], idx = read_hierarchy(lines, idx)
        else:
            print("ignore", next_line)
            idx+=1
    print("read", len(data["bones"]), "bones")
    data = create_c_matrices(data)
    return data

def read_root_data(lines, idx):
    data = dict()
    print("start root", idx)
    next_line = lines[idx].strip()
    while not next_line.startswith(":bonedata") and idx+1 < len(lines):
        values = next_line.split(" ")
        if len(values) > 0:
            key = values[0]
            if key == "position":
                data["position"] =  [values[1], values[2], values[3]]
            elif key == "orientation":
                data["orientation"] =  [float(values[1]),float(values[2]), float(values[3])]
            elif key == "axis":
                data["axis"] =  values[1]
            elif key == "order":
                data["order"] =  [v for v in values if v != "order"]
            #elif key == "limits":
            #    data[key] =  [values[1], values[2], values[3]]
        if idx+1 < len(lines):
            idx+=1
            next_line = lines[idx].strip() # remove empty lines

    print("end root", idx, next_line)
    return data, idx

def read_bone_data(lines, idx):
    idx +=1 #skip begin
    data = dict()
    #print("start bone", idx)
    next_line = lines[idx].strip()
    while not next_line.startswith("end") and idx+1 < len(lines):
        values = next_line.split(" ")
        values = [v for v in values if v != ""]
        if len(values) > 0:
            key = values[0]
            if key == "id":
                data["id"] = values[1]
            elif key == "name":
                data["name"] = values[1]
            elif key == "direction":
                direction =  np.array([float(v) for v in values if v != "direction"])
                direction /= np.linalg.norm(direction)
                data["direction"] = direction.tolist()
            elif key == "length":
                print(values)
                data["length"] =  float(values[1])
            elif key == "axis":
                data["axis"] =  [float(values[1]),  float(values[2]),  float(values[3])], values[4]
            elif key == "dof":
                data["dof"] =  [v for v in values if v != "dof"]
            #elif key == "limits":
            #    data[key] =  [values[1], values[2], values[3]]
        if idx+1 < len(lines):
            idx+=1
            next_line = lines[idx].strip() # remove empty lines
    idx +=1 #skip end
    #print("end", idx, lines[idx])
    return data, idx


def read_hierarchy(lines, idx):
    print("found hierarchy")
    idx +=1 #skip begin
    child_dict = dict()
    next_line = lines[idx].strip()
    while not next_line.startswith("end"):
        values = next_line.split(" ")
        if len(values) > 1:
            child_dict[values[0]] = values[1:] 
        idx+=1
        next_line = lines[idx].strip() # remove empty lines
    idx +=1 #skip end
    return child_dict, idx
    
def parse_amc_file(skeleton_data, filepath):
    with open(filepath, "rb") as in_file:
        lines = in_file.readlines()
    lines = [str(l, "utf-8") for l in lines]
    frames = []
    current_frame = None
    frame_idx = 0
    idx = 0
    while idx < len(lines):
        next_line = lines[idx].strip()
        values = next_line.split(" ")
        if len(values) == 1:
            if values[0].isdigit():
                if current_frame is not None:
                    frames.append(current_frame)
                current_frame = dict()
        elif len(values) > 1 and current_frame is not None:
            key = values[0]
            if key == "root" or (key in skeleton_data["bones"] and len(values)-1 == len(skeleton_data["bones"][key]["dof"])):
                current_frame[key] = [float(v) for v in values if v != key]

        idx+=1
    return frames
