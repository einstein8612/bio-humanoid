# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

from isaacgym.torch_utils import *
import torch
import json
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion

from tqdm import tqdm

VISUALIZE = False

def project_joints(motion):
    right_upper_arm_id = motion.skeleton_tree._node_indices["right_shoulder_pitch_link"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_elbow_link"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand_link"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_shoulder_pitch_link"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_elbow_link"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand_link"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_hip_pitch_link"]
    right_shin_id = motion.skeleton_tree._node_indices["right_knee_link"]
    right_foot_id = motion.skeleton_tree._node_indices["right_ankle_link"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_hip_pitch_link"]
    left_shin_id = motion.skeleton_tree._node_indices["left_knee_link"]
    left_foot_id = motion.skeleton_tree._node_indices["left_ankle_link"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    # print("right_arm_delta0: ", right_arm_delta0, " right_arm_delta1: ", right_arm_delta1)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)
    
    # right leg
    right_thigh_pos = motion.global_translation[..., right_thigh_id, :]
    right_shin_pos = motion.global_translation[..., right_shin_id, :]
    right_foot_pos = motion.global_translation[..., right_foot_id, :]
    right_hip_rot = motion.local_rotation[..., right_thigh_id, :]
    right_knee_rot = motion.local_rotation[..., right_shin_id, :]
    
    right_leg_delta0 = right_thigh_pos - right_shin_pos
    right_leg_delta1 = right_foot_pos - right_shin_pos
    right_leg_delta0 = right_leg_delta0 / torch.norm(right_leg_delta0, dim=-1, keepdim=True)
    right_leg_delta1 = right_leg_delta1 / torch.norm(right_leg_delta1, dim=-1, keepdim=True)
    right_knee_dot = torch.sum(-right_leg_delta0 * right_leg_delta1, dim=-1)
    right_knee_dot = torch.clamp(right_knee_dot, -1.0, 1.0)
    right_knee_theta = torch.acos(right_knee_dot)
    right_knee_q = quat_from_angle_axis(torch.abs(right_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    right_knee_local_dir = motion.skeleton_tree.local_translation[right_foot_id]
    right_knee_local_dir = right_knee_local_dir / torch.norm(right_knee_local_dir)
    right_knee_local_dir_tile = torch.tile(right_knee_local_dir.unsqueeze(0), [right_knee_rot.shape[0], 1])
    right_knee_local_dir0 = quat_rotate(right_knee_rot, right_knee_local_dir_tile)
    right_knee_local_dir1 = quat_rotate(right_knee_q, right_knee_local_dir_tile)
    right_leg_dot = torch.sum(right_knee_local_dir0 * right_knee_local_dir1, dim=-1)
    right_leg_dot = torch.clamp(right_leg_dot, -1.0, 1.0)
    right_leg_theta = torch.acos(right_leg_dot)
    right_leg_theta = torch.where(right_knee_local_dir0[..., 1] >= 0, right_leg_theta, -right_leg_theta)
    right_leg_q = quat_from_angle_axis(right_leg_theta, right_knee_local_dir.unsqueeze(0))
    right_hip_rot = quat_mul(right_hip_rot, right_leg_q)
    
    # left leg
    left_thigh_pos = motion.global_translation[..., left_thigh_id, :]
    left_shin_pos = motion.global_translation[..., left_shin_id, :]
    left_foot_pos = motion.global_translation[..., left_foot_id, :]
    left_hip_rot = motion.local_rotation[..., left_thigh_id, :]
    left_knee_rot = motion.local_rotation[..., left_shin_id, :]
    
    left_leg_delta0 = left_thigh_pos - left_shin_pos
    left_leg_delta1 = left_foot_pos - left_shin_pos
    left_leg_delta0 = left_leg_delta0 / torch.norm(left_leg_delta0, dim=-1, keepdim=True)
    left_leg_delta1 = left_leg_delta1 / torch.norm(left_leg_delta1, dim=-1, keepdim=True)
    left_knee_dot = torch.sum(-left_leg_delta0 * left_leg_delta1, dim=-1)
    left_knee_dot = torch.clamp(left_knee_dot, -1.0, 1.0)
    left_knee_theta = torch.acos(left_knee_dot)
    left_knee_q = quat_from_angle_axis(torch.abs(left_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    left_knee_local_dir = motion.skeleton_tree.local_translation[left_foot_id]
    left_knee_local_dir = left_knee_local_dir / torch.norm(left_knee_local_dir)
    left_knee_local_dir_tile = torch.tile(left_knee_local_dir.unsqueeze(0), [left_knee_rot.shape[0], 1])
    left_knee_local_dir0 = quat_rotate(left_knee_rot, left_knee_local_dir_tile)
    left_knee_local_dir1 = quat_rotate(left_knee_q, left_knee_local_dir_tile)
    left_leg_dot = torch.sum(left_knee_local_dir0 * left_knee_local_dir1, dim=-1)
    left_leg_dot = torch.clamp(left_leg_dot, -1.0, 1.0)
    left_leg_theta = torch.acos(left_leg_dot)
    left_leg_theta = torch.where(left_knee_local_dir0[..., 1] >= 0, left_leg_theta, -left_leg_theta)
    left_leg_q = quat_from_angle_axis(left_leg_theta, left_knee_local_dir.unsqueeze(0))
    left_hip_rot = quat_mul(left_hip_rot, left_leg_q)
    

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., right_thigh_id, :] = right_hip_rot
    new_local_rotation[..., right_shin_id, :] = right_knee_q
    new_local_rotation[..., left_thigh_id, :] = left_hip_rot
    new_local_rotation[..., left_shin_id, :] = left_knee_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)
    
    return new_motion

def process(file_name):
    source_motion_file = os.path.join(source_motion_path, file_name)
    target_motion_file = os.path.join(target_motion_path, file_name)
    
    source_motion = SkeletonMotion.from_file(source_motion_file)
    
    target_motion = source_motion.retarget_to_by_tpose(
        joint_mapping=joint_mapping,
        source_tpose=source_tpose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=scale_to_target_skeleton
    )
    
    # Post processing rotation
    frame_beg = retarget_data["trim_frame_beg"]
    frame_end = retarget_data["trim_frame_end"]
    if (frame_beg == -1):
        frame_beg = 0
        
    if (frame_end == -1):
        frame_end = target_motion.local_rotation.shape[0]
        
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]
      
    tar_global_pos = target_motion.global_translation[frame_beg:frame_end, ...]
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h

    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # save retargeted motion
    target_motion.to_file(target_motion_file)


# load retarget config
retarget_data_path = "data/configs/retarget_cmu_to_h1.json"
with open(retarget_data_path) as f:
    retarget_data = json.load(f)

joint_mapping = retarget_data["joint_mapping"]
rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])
scale_to_target_skeleton = retarget_data["scale"]

# load and visualize t-pose files
source_tpose = SkeletonState.from_file(retarget_data["source_tpose"])
target_tpose = SkeletonState.from_file(retarget_data["target_tpose"])

source_motion_path = retarget_data["source_motion_path"]
target_motion_path = retarget_data["target_motion_path"]

for source_file in tqdm(os.listdir(source_motion_path), desc="Retargeting motions"):
    process(source_file)