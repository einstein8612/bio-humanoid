from humanoid.envs.custom.h1_dance_config import H1DanceCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import HumanoidTerrain


from ...scripts.sim2sim import quaternion_to_euler_array

from typing import cast

import sys
from pathlib import Path
# To import poselib, add the path to the poselib package to the system path
sys.path.append(
    str(Path(__file__).absolute().parent.parent.parent.parent.parent))
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion


class H1DanceFreeEnv(LeggedRobot):
    '''
    H1DanceFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''

    def __init__(self, cfg: H1DanceCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)

        self.motion = cast(SkeletonMotion, SkeletonMotion.from_file(cfg.motion.motion_file))
        # self.motion.tensor = self.motion.tensor.to(device=self.device)

        self.total_frames = self.motion.tensor.shape[0]
        self.start_frame_buf = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)

        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def _local_rotation_to_dof_pos(self, motion: SkeletonMotion, frame: int):
        """
        Match the motion's frame information which has all joints modeled as ball joints
        to the actual joints, which are just a bunch of normal hinge joints.

        This works, for all frames except the TPose but we don't care anyways
        """
        eulers_xyz_rpy = torch.tensor(
            np.array(list(map(quaternion_to_euler_array, motion.local_rotation[frame]))))

        # print(motion.skeleton_tree.node_names)

        ref_dof_pos = torch.zeros(self.num_dof)
        # Left leg
        ref_dof_pos[0] = eulers_xyz_rpy[1, 2]
        ref_dof_pos[1] = eulers_xyz_rpy[1, 0]
        ref_dof_pos[2] = eulers_xyz_rpy[1, 1]
        ref_dof_pos[3] = eulers_xyz_rpy[2, 1]
        ref_dof_pos[4] = eulers_xyz_rpy[3, 1]
        # Right leg
        ref_dof_pos[5] = eulers_xyz_rpy[4, 2]
        ref_dof_pos[6] = eulers_xyz_rpy[4, 0]
        ref_dof_pos[7] = eulers_xyz_rpy[4, 1]
        ref_dof_pos[8] = eulers_xyz_rpy[5, 1]
        ref_dof_pos[9] = eulers_xyz_rpy[6, 1]
        # Torso
        ref_dof_pos[10] = eulers_xyz_rpy[7, 2]
        # Left arm
        ref_dof_pos[11] = -eulers_xyz_rpy[8, 0]
        ref_dof_pos[12] = -eulers_xyz_rpy[8, 2]
        ref_dof_pos[13] = eulers_xyz_rpy[8,  1]
        ref_dof_pos[14] = eulers_xyz_rpy[9,  1]
        # eulers_xyz_rpy[10] # Left Hand
        # Right arm
        ref_dof_pos[15] = eulers_xyz_rpy[11, 0]
        ref_dof_pos[16] = -eulers_xyz_rpy[11, 2]
        ref_dof_pos[17] = eulers_xyz_rpy[11,  1]
        ref_dof_pos[18] = eulers_xyz_rpy[12,  1]
        # eulers_xyz_rpy[13] # Right Hand

        return ref_dof_pos.fmod(torch.pi)

    def _local_vel_to_dof_vel(self, local_velocity: torch.Tensor, frames: torch.Tensor):
        ref_dof_vel = torch.zeros_like(self.dof_vel)
        
        for i, frame in enumerate(frames):
            # Left leg
            ref_dof_vel[i, 0] = local_velocity[frame, 1, 2]
            ref_dof_vel[i, 1] = local_velocity[frame, 1, 0]
            ref_dof_vel[i, 2] = local_velocity[frame, 1, 1]
            ref_dof_vel[i, 3] = local_velocity[frame, 2, 1]
            ref_dof_vel[i, 4] = local_velocity[frame, 3, 1]
            # Right leg
            ref_dof_vel[i, 5] = local_velocity[frame, 4, 2]
            ref_dof_vel[i, 6] = local_velocity[frame, 4, 0]
            ref_dof_vel[i, 7] = local_velocity[frame, 4, 1]
            ref_dof_vel[i, 8] = local_velocity[frame, 5, 1]
            ref_dof_vel[i, 9] = local_velocity[frame, 6, 1]
            # Torso
            ref_dof_vel[i, 10] = local_velocity[frame, 7, 2]
            # Left arm
            ref_dof_vel[i, 11] = -local_velocity[frame, 8, 0]
            ref_dof_vel[i, 12] = -local_velocity[frame, 8, 2]
            ref_dof_vel[i, 13] = local_velocity[frame, 8,  1]
            ref_dof_vel[i, 14] = local_velocity[frame, 9,  1]
            # local_velocity[frame, 10] # Left Hand
            # Right arm
            ref_dof_vel[i, 15] = local_velocity[frame, 11, 0]
            ref_dof_vel[i, 16] = -local_velocity[frame, 11, 2]
            ref_dof_vel[i, 17] = local_velocity[frame, 11,  1]
            ref_dof_vel[i, 18] = local_velocity[frame, 12,  1]
            # local_velocity[frame, 13] # Right Hand

        return ref_dof_vel


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def compute_ref_state(self, time: torch.Tensor):
        # Just treat time as seconds, it's slower but then the movement can also be slow
        # self.motion.
        frames = self.start_frame_buf + (time * self.motion.fps).floor().to(torch.int64)

        # Stop at the last one
        frames = torch.minimum(frames, torch.tensor(self.total_frames-1))

        if torch.any(frames >= self.total_frames):
            # TODO, step? or stand
            return
        
        self.ref_dof_pos = torch.vstack([
            self._local_rotation_to_dof_pos(self.motion, frame=frame) 
            for frame in frames
        ]).to(device=self.device)

        # self.ref_dof_vel = self._local_vel_to_dof_vel(self.motion._compute_velocity(p=self.motion.local_translation, time_delta=1/self.motion.fps), frames)
        # self.ref_dof_vel = self._local_vel_to_dof_vel(self.motion.global_velocity, frames)
        self.ref_dof_root_ang_vel = self.motion.global_root_angular_velocity[frames.cpu()].clone().to(device=self.device)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _get_current_time(self):
        return self.episode_length_buf * self.dt

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands
        noise_vec[5: 24] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[24: 43] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[43: 62] = 0.  # previous actions
        noise_vec[62: 65] = noise_scales.ang_vel * \
            self.obs_scales.ang_vel   # ang vel
        noise_vec[65: 68] = noise_scales.quat * \
            self.obs_scales.quat         # euler x,y
        return noise_vec

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        actions = torch.clip(
            actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * \
            self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * \
            torch.randn_like(actions) * actions

        return super().step(actions)

    def compute_observations(self):
        time = self._get_current_time()
        self.compute_ref_state(time)

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos - self.ref_dof_pos
        
        # Used since it's a sequence
        time = self._get_current_time().unsqueeze(1)

        self.privileged_obs_buf = torch.cat((
            (self.dof_pos - self.default_joint_pd_target) *
            self.obs_scales.dof_pos,  # 19
            self.dof_vel * self.obs_scales.dof_vel,  # 19
            self.actions,  # 19
            diff,  # 19
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1,
            time, #1
        ), dim=-1)

        obs_buf = torch.cat((
            q,    # 19D
            dq,  # 19D
            self.actions,   # 19D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            time, #1
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(
                1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat(
                (self.obs_buf, heights), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * \
                self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

        self.start_frame_buf[env_ids] = torch.randint(1, self.total_frames, (len(env_ids),), device=self.device)

# ================================================ Rewards ================================================== #
    def _reward_alive(self):
        return 1.

    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-0.7 * torch.norm(diff, dim=1))
        return r
    
    def _reward_joint_vel(self):
        """
        Calculates the reward based on the difference between the current joint vel and the target joint vel.
        """
        joint_pos = self.dof_vel.clone()
        pos_target = self.ref_dof_vel.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-0.5 * torch.norm(diff, dim=1))
        return r
    
    def _reward_movement(self):
        movement_sum = torch.sum(torch.abs(self.dof_vel), dim=1)
        r = torch.exp(0.025 * movement_sum) - 1
        return r.clip(None, 0.05)

    def _reward_joint_ang_vel(self):
        demo_ang_vel = self.ref_dof_root_ang_vel.clone()
        rew = torch.exp(-torch.norm(self.base_ang_vel - demo_ang_vel, dim=1))
        return rew

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        dof_pos_error = torch.norm((self.dof_pos - self.default_dof_pos)[:, :11], dim=1)
        dof_vel_error = torch.norm(self.dof_vel[:, :11], dim=1)
        rew = torch.exp(- 0.1*dof_vel_error) * torch.exp(- dof_pos_error)
        return rew
    

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(
            self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    # def _reward_track_vel_hard(self):
    #     """
    #     Calculates a reward for accurately tracking both linear and angular velocity commands.
    #     Penalizes deviations from specified linear and angular velocity targets.
    #     """
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.norm(
    #         self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
    #     lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

    #     # Tracking of angular velocity commands (yaw)
    #     ang_vel_error = torch.abs(
    #         self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

    #     linear_error = 0.2 * (lin_vel_error + ang_vel_error)

    #     return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    # def _reward_tracking_lin_vel(self):
    #     """
    #     Tracks linear velocity commands along the xy axes. 
    #     Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
    #     """
    #     lin_vel_error = torch.sum(torch.square(
    #         self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    # def _reward_tracking_ang_vel(self):
    #     """
    #     Tracks angular velocity commands for yaw rotation.
    #     Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
    #     """

    #     ang_vel_error = torch.square(
    #         self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
