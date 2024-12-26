from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1DanceCfg(LeggedRobotCfg):
    """
    Configuration class for the H1 humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 5
        num_actions = 19

        # Observables
        num_single_obs = 7 + num_actions*3
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 17 + num_actions*4
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        # Training envs
        num_envs = 2000
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class motion:
        # motion_file = "../../poselib/data/h1_motions/18_15.npy"
        motion_file = "../../poselib/data/h1_motions/21_01.npy"

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/H1/urdf/h1.urdf'

        name = "H1"
        foot_name = "ankle"
        knee_name = "knee"

        terminate_after_contacts_on = ["pelvis", "torso_link"]
        penalize_contacts_on = ["torso_link", "shoulder", "elbow", "hip"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.95]

        # 19 Actions
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0.,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.4,
            'left_knee_joint': 0.8,
            'left_ankle_joint': -0.4,
            'right_hip_yaw_joint': 0.,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.4,
            'right_knee_joint': 0.8,
            'right_ankle_joint': -0.4,
            'torso_joint': 0.,
            'left_shoulder_pitch_joint': 0.,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0.,
            'left_elbow_joint': 0.,
            'right_shoulder_pitch_joint': 0.,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.,
            'right_elbow_joint': 0.,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            'hip_roll': 200.0,
            'hip_yaw': 200.0,
            'hip_pitch': 200.0,
            'knee': 200.0,
            'ankle': 15.0,
            'torso': 300.0,
            'shoulder_roll': 50.0,
            'shoulder_pitch': 50.0,
            'shoulder_yaw': 50.0,
            'elbow': 150.0
        }
        damping = {
            'hip_roll': 5.0,
            'hip_yaw': 5.0,
            'hip_pitch': 10.0,
            'knee': 10.0,
            'ankle': 2.0,
            'torso': 6.0,
            'shoulder_roll': 2.0,
            'shoulder_pitch': 2.0,
            'shoulder_yaw': 2.0,
            'elbow': 2.0
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.89
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 700  # Forces above this value are penalized

        class scales:
            alive = 0.5
            # reference motion tracking
            joint_pos = 5
            arm_joint_pos = 2.5
            joint_vel = 1
            joint_ang_vel = 0.5
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            # tracking_lin_vel = 1.2
            # tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            base_acc = 0.2
            # movement = 0.25
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.5

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class H1DanceCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 500  # number of policy updates

        # logging
        # Please check for potential savings every `save_interval` iterations.
        save_interval = 100
        experiment_name = 'H1_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
