import numpy as np

from joblib import Parallel, delayed
import glob, sys, os

from generate_input_grasp import ParticipantGrasp

sys.path.append('./motion')
sys.path.append("../../../")

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


""" Options """

rng = np.random.RandomState(1234)

""" Data """
 
path ="../../../data/data_grasp"

data = [m for m in glob.glob(os.path.join(path, "*.bvh"))]
path = "../../../data/data_praktikum"

participants = {
    'female' : [
        "NXDS9232",
        "JUIDM3242",
        # "JKYRF234F",
        # "MNCXJUD2",
        # "ASD23S2D",
        # "NXJD23356Z",
        # "2EDRCG34",
        # "98211XYDS",
        # "985CCLPFE",
        # "IJMX0867DSX",
        # "IXJE534",
        # "898MXJDFI", # 12
        #"898XDXYDS",
        #"JXDSD93223",
    ],
    'male' : [
        "JDXD98783",
        "ZXD2132",
        # "89XFDD21FD",
        # "NXHDS231",
        # "XDSD123",
        # "565MNMXDEFC",
        # "JNGXI873",
        # "23KJINCSE3",
        # "214CSDF",
        # "673DCDSD",
        # "121MDXDEW",
        # "787DXDECX", # 12
    ]
}

for m in glob.glob(os.path.join(path, "*.bvh"))[0:20]:
    data.append(m)
print(data)


def format_print(x, y, slc_start):
    j = 21
    w = 12
    g = 3

    print("Path Location")
    print("    f" , end="")
    for i in range(len(x)):
        print("%15d               "%((slc_start + i)*2), end="")
    print("", end="\n")
    for j in range(w):
        print("%5d"%(j - w//2), end = "")
        for i in range(len(x)):
            # if j >= 6:
            #     offset = 4
            #     pred = (y[i,offset + j-6], y[i,offset + j])
            # else:
            #     pred = (0,0)
            print("{:{w}.{p}f},{:{w}.{p}f} |{:{w}.{p}f},{:{w}.{p}f}  ".format(x[i,j],x[i,w + j],x[i,2*w + j], x[i,3 * w + j], w = 6, p = 2), end="")
        print("", end="\n")

    print("\nLocation Pred")
    print("     ", end="")
    for f in range(len(x)):
        print("   (%5.2f,%5.2f), %5.2f       "%(y[f,0], y[f,1], y[f,2]), end="")

    print("\nGait Input")
    base = 4 * w
    for j in range(w):
        print("%5d"%(j - w//2), end = "")
        for f in range(len(x)):
            print("     %6.4f,%6.4f,%6.4f     "%(x[f,base+j], x[f,base+w+j],x[f,base+2*w+j]), end="")
        print("",end="\n")

""" Phases, Inputs, Outputs """

P, X, Y = [], [], []

participant_data = {}

def process(path):
    global participant_data
    part = os.path.basename(path)[:-4]
    participant_data[part] = ParticipantGrasp(path)
    participant_data[part].generate_train_test(rng)
    participant_data[part].save_models("../../../experimental/grasp/" + part + "_%s.npz")

#for data in data_terrain:

Parallel(n_jobs=12, backend="threading")(delayed(process)(path) for path in data)
# for path in data:
#     process(path)


""" Clip Statistics """

# print('Total Clips: %i' % len(X))
# print('Shortest Clip: %i' % min(map(len,X)))
# print('Longest Clip: %i' % max(map(len,X)))
# print('Average Clip: %i' % np.mean(list(map(len,X))))



""" Merge Clips """

print('Merging Clips...')

Xun = np.concatenate(X, axis=0)
Yun = np.concatenate(Y, axis=0)
Pun = np.concatenate(P, axis=0)


print(Xun.shape, Yun.shape, Pun.shape)

print('Saving Database...')

#np.savez_compressed('./masterThesis/database.npz', Xun=Xun, Yun=Yun, Pun=Pun)


# def print_positions(pos):
#     names = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg", "LeftFoot", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]
#     pos = np.concatenate([pos[0:17], pos[36:40]], axis=0)
#     for i in range(len(pos)):
#         print(names[i], pos[i])
