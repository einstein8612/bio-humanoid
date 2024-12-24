import os
import sys

from poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.visualization.common import plot_skeleton_motion_interactive

if len(sys.argv) != 2:
    print("Please provide a motion name, eg: 01_01")
    os._exit(1)

motion_name = sys.argv[1]

motion_file = f"data/h1_motions/{motion_name}.npy"

print("Loading motion")
motion = SkeletonMotion.from_file(motion_file)

# visualize motion
plot_skeleton_motion_interactive(motion)
