# This is a script to retarget the CMU mocap data to the RobotEra XBot-L

import os
import numpy as np
from tqdm import tqdm

def retarget_cmu_mocap(mocap_data_path: str, retargeted_data_path: str):
    if not os.path.exists(retargeted_data_path):
        os.makedirs(retargeted_data_path, exist_ok=True)
        
    for file in tqdm(os.listdir(mocap_data_path), "Retargeting CMU mocap data"):
        retargeted_data = np.zeros((1, 24)) # 24 joints in the XBot-L
        if file.endswith(".bvh") and "01_01" in file:
            with open(os.path.join(mocap_data_path, file), "r") as f:
                lines = f.readlines()
                motions = lines[187:] # skip the hierarchy, we know this is 187 lines since it's CMU mocap data
                motions = np.array([motion.split() for motion in motions], dtype=np.float32) # Dataset is f32
                motions = motions[:,3:] # We don't care about positional data
                
                hips = motions[:,0:3]
                left_hip = motions[:,3:6]
                left_up_leg = motions[:,6:9]
                left_leg = motions[:,9:12]
                left_foot = motions[:,12:15]
                left_toe = motions[:,15:18]
                right_hip = motions[:,18:21]
                right_up_leg = motions[:,21:24]
                right_leg = motions[:,24:27]
                right_foot = motions[:,27:30]
                right_toe = motions[:,30:33]
                lower_back = motions[:,33:36]
                spine = motions[:,36:39]
                spine_1 = motions[:,39:42]
                neck = motions[:,42:45]
                neck_1 = motions[:,45:48]
                head = motions[:,48:51]
                left_shoulder = motions[:,51:54]
                left_arm = motions[:,54:57]
                left_forearm = motions[:,57:60]
                left_hand = motions[:,60:63]
                left_finger = motions[:,63:66]
                left_index_finger = motions[:,66:69]
                left_thumb = motions[:,69:72]
                right_shoulder = motions[:,72:75]
                right_arm = motions[:,75:78]
                right_forearm = motions[:,78:81]
                right_hand = motions[:,81:84]
                right_finger = motions[:,84:87]
                right_index_finger = motions[:,87:90]
                right_thumb = motions[:,90:93]
                
                # We need to retarget the data to the XBot-L
                # When doing this we know that for the bvh data
                # it's in the order Z,Y,X. We need to convert this
                # to yaw pitch roll, which are decided by the urdf file for
                # the XBot-L.
                retargeted_data[0, 0] = left_shoulder[1]  # left_shoulder_roll_joint
                retargeted_data[0, 1] = left_shoulder[1]  # left_shoulder_pitch_joint
                retargeted_data[0, 2] = left_arm[0]       # left_elbow_yaw_joint
                retargeted_data[0, 3] = left_arm[1]       # left_elbow_pitch_joint
                retargeted_data[0, 4] = left_forearm[0]   # left_wrist_roll_joint
                retargeted_data[0, 5] = left_forearm[1]   # left_wrist_yaw_joint
                retargeted_data[0, 6] = right_shoulder[0] # right_shoulder_roll_joint
                retargeted_data[0, 7] = right_shoulder[1] # right_shoulder_pitch_joint
                retargeted_data[0, 8] = right_arm[0]      # right_elbow_yaw_joint
                retargeted_data[0, 9] = right_arm[1]      # right_elbow_pitch_joint
                retargeted_data[0, 10] = right_forearm[0] # right_wrist_roll_joint
                retargeted_data[0, 11] = right_forearm[1] # right_wrist_yaw_joint
                
                # Legs
                retargeted_data[0, 12] = left_hip[0]      # left_leg_roll_joint
                retargeted_data[0, 13] = left_hip[1]      # left_leg_yaw_joint
                retargeted_data[0, 14] = left_up_leg[0]   # left_leg_pitch_joint
                retargeted_data[0, 15] = left_leg[0]      # left_knee_joint
                retargeted_data[0, 16] = left_foot[0]     # left_ankle_pitch_joint
                retargeted_data[0, 17] = left_foot[1]     # left_ankle_roll_joint
                retargeted_data[0, 18] = right_hip[0]     # right_leg_roll_joint
                retargeted_data[0, 19] = right_hip[1]     # right_leg_yaw_joint
                retargeted_data[0, 20] = right_up_leg[0]  # right_leg_pitch_joint
                retargeted_data[0, 21] = right_leg[0]     # right_knee_joint
                retargeted_data[0, 22] = right_foot[0]    # right_ankle_pitch_joint
                retargeted_data[0, 23] = right_foot[1]    # right_ankle_roll_joint
                
                np.save(os.path.join(retargeted_data_path, file.replace(".bvh", ".npy")), retargeted_data)
                
                return