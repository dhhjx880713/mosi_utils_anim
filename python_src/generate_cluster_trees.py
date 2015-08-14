import os
import time
from morphablegraphs.construction.cluster_tree_builder import ClusterTreeBuilder
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)

CONIFG_FILE_PATH = "config" + os.sep + "space_partitioning.json"

def main():

    cluster_tree_builder = ClusterTreeBuilder()
    cluster_tree_builder.set_config(CONIFG_FILE_PATH)
    start = time.clock()
    success = cluster_tree_builder.build()

    time_in_seconds = time.clock()-start
    if success:
        print "Finished construction in", int(time_in_seconds/60), "minutes and", time_in_seconds % 60, "seconds"
    else:
        print "Failed to read data from directory"

if __name__ == "__main__":
    main()