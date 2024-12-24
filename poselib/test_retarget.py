import os
import sys

import torch

from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_motion_interactive

if len(sys.argv) != 2:
    print("Please provide a motion name, eg: 01_01")
    os._exit(1)

motion_name = sys.argv[1]

# source fbx file path
fbx_file = f"data/cmu/{motion_name}.fbx"

print("Loading T-Poses")

source_tpose = SkeletonState.from_file("data/tpose/cmu_tpose.npy")
target_tpose = SkeletonState.from_file("data/tpose/h1_tpose.npy")

print("Loading source motion")

source_motion = SkeletonMotion.from_fbx(
    fbx_file_path=fbx_file,
    root_joint="Hips",
    fps=60
)

joint_mapping = {
         "Hips": "pelvis",
         "LeftUpLeg": "left_hip_yaw_link",
         "LeftLeg": "left_knee_link",
         "LeftFoot": "left_ankle_link",
         "RightUpLeg": "right_hip_yaw_link",
         "RightLeg": "right_knee_link",
         "RightFoot": "right_ankle_link",
         "Spine1": "torso_link",
         "LeftArm": "left_shoulder_pitch_link",
         "LeftForeArm": "left_elbow_link",
         "LeftHand": "left_hand_link",
         "RightArm": "right_shoulder_pitch_link",
         "RightForeArm": "right_elbow_link",
         "RightHand": "right_hand_link"
    }
rotation_to_target_skeleton = torch.tensor([ 0.5, 0.5, 0.5, 0.5])

print("Retargeting")

# run retargeting
target_motion = source_motion.retarget_to_by_tpose(
    joint_mapping=joint_mapping,
    source_tpose=source_tpose,
    target_tpose=target_tpose,
    rotation_to_target_skeleton=rotation_to_target_skeleton,
    scale_to_target_skeleton=0.056444
)

print("Visualizing")

# visualize motion
plot_skeleton_motion_interactive(target_motion)
