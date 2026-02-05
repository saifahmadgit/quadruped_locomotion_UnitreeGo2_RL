import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs

from go2_env_DR_curriculum.py import Go2Env


def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }


def get_cfgs():
    # -----------------------------
    # CURRICULUM + DR KNOBS (EDIT THESE)
    # -----------------------------
    # Each level overrides:
    #   - command ranges
    #   - friction/kp/kd randomization ranges
    #   - pushes
    #   - latency, action_scale
    #   - init_dof_pos_noise (radians)
    #
    # Recommended philosophy:
    #   Level 0-1: no pushes, narrow DR, no latency, small speeds
    #   Level 2-3: widen commands, mild pushes, mild DR
    #   Level 4-5: full commands, pushes, wider DR, latency
    curriculum_levels = [
        # L0: learn to stand + tiny forward (super easy)
        {
            "lin_vel_x_range": [0.0, 0.1],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [0.0, 0.0],
            "friction_range": (0.9, 1.1),
            "kp_scale_range": (0.95, 1.05),
            "kd_scale_range": (0.95, 1.05),
            "rand_friction_enable": True,
            "rand_pd_enable": True,
            "push_enable": False,
            "simulate_action_latency": False,
            "action_scale": 0.20,
            "init_dof_pos_noise": 0.02,  # ~1 deg
        },
        # L1: slow forward
        {
            "lin_vel_x_range": [0.1, 0.4],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [0.0, 0.0],
            "friction_range": (0.8, 1.2),
            "kp_scale_range": (0.9, 1.1),
            "kd_scale_range": (0.9, 1.1),
            "push_enable": False,
            "simulate_action_latency": False,
            "action_scale": 0.22,
            "init_dof_pos_noise": 0.03,
        },
        # L2: faster forward, start tiny yaw
        {
            "lin_vel_x_range": [0.2, 0.8],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [-0.3, 0.3],
            "friction_range": (0.7, 1.3),
            "kp_scale_range": (0.85, 1.15),
            "kd_scale_range": (0.85, 1.15),
            "push_enable": True,
            "push_interval_s": 2.0,
            "push_prob": 0.10,
            "push_force_range": (30.0, 80.0),
            "push_duration_s": 0.12,
            "push_direction_mode": "random",
            "simulate_action_latency": False,
            "action_scale": 0.25,
            "init_dof_pos_noise": 0.04,
        },
        # L3: wide forward + yaw, stronger pushes
        {
            "lin_vel_x_range": [0.1, 1.2],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [-0.8, 0.8],
            "friction_range": (0.5, 1.5),
            "kp_scale_range": (0.8, 1.2),
            "kd_scale_range": (0.8, 1.2),
            "push_enable": True,
            "push_interval_s": 1.5,
            "push_prob": 0.20,
            "push_force_range": (50.0, 120.0),
            "push_duration_s": 0.15,
            "simulate_action_latency": True,  # introduce latency here
            "action_scale": 0.27,
            "init_dof_pos_noise": 0.05,
        },
        # L4: full forward range (no lateral), tougher DR
        {
            "lin_vel_x_range": [0.1, 1.5],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [-1.0, 1.0],
            "friction_range": (0.4, 1.6),
            "kp_scale_range": (0.75, 1.25),
            "kd_scale_range": (0.75, 1.25),
            "push_enable": True,
            "push_interval_s": 1.0,
            "push_prob": 0.25,
            "push_force_range": (70.0, 160.0),
            "push_duration_s": 0.15,
            "simulate_action_latency": True,
            "action_scale": 0.30,
            "init_dof_pos_noise": 0.06,
        },
    ]

    env_cfg = {
        "num_actions": 12,

        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },

        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],

        # PD nominal (still used; DR scales around these)
        "kp": 60.0,
        "kd": 2.0,

        # Base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],

        # Termination
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "termination_if_z_vel_greater_than": 100.0,
        "termination_if_y_vel_greater_than": 100.0,

        # Timing
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,

        # Action interpretation (will be overridden by curriculum levels)
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,

        # Default DR fallback (used if curriculum disabled)
        "friction_range": (0.4, 1.2),
        "kp_scale_range": (0.75, 1.25),
        "kd_scale_range": (0.75, 1.25),
        "rand_friction_enable": True,
        "rand_pd_enable": True,

        # Default push fallback (used if curriculum disabled)
        "push_enable": True,
        "push_interval_s": 1.0,
        "push_prob": 0.15,
        "push_force_range": (50.0, 150.0),
        "push_z_scale": 0.0,
        "push_duration_s": 0.15,
        "push_direction_mode": "random",

        # Default init DOF noise fallback (used if curriculum disabled)
        "init_dof_pos_noise": 0.03,

        # Curriculum master switch + knobs
        "curriculum": {
            "enabled": True,
            "start_level": 0,

            # After this many finished episodes (across all envs),
            # compute success_rate = timeouts/dones and adjust level.
            "window_episodes": 2048,

            # If success_rate >= up -> harder; if <= down -> easier
            "success_threshold_up": 0.80,
            "success_threshold_down": 0.50,

            "levels": curriculum_levels,
        },
    }

    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }

    # NOTE: command_cfg is still passed in, but curriculum overwrites ranges inside env at runtime
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.1, 1.5],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-curriculum")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
