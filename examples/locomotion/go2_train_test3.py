# =========================
# File: go2_train_test3.py
# =========================

import argparse
import os
import pickle
import shutil
from importlib import metadata

# -------- rsl-rl version guard --------
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

from go2_env_test3 import Go2Env


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
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
        "save_interval": 1000,
        "empirical_normalization": None,
        "seed": 1,
    }
    return train_cfg_dict


def get_cfgs():
    # =================================================================
    #              DOMAIN RANDOMISATION — TRAINING CONFIG
    # =================================================================

    # -------------------- 1. SURFACE FRICTION -----------------------
    friction_enable = True
    friction_range = [0.4, 1.2]

    # -------------------- 2. PD GAIN RANDOMISATION ------------------
    kp_kd_enable = True
    kp_nominal = 60.0
    kd_nominal = 2.0
    kp_range = [50.0, 80.0]
    kd_range = [2.0, 5.0]

    # -------------------- 3. OBSERVATION NOISE ----------------------
    obs_noise_enable = True
    obs_noise_level = 0.15
    obs_noise = {
        "ang_vel": 0.2,
        "gravity": 0.05,
        "dof_pos": 0.02,
        "dof_vel": 1.0,
    }

    # -------------------- 3b. ACTION NOISE --------------------------
    action_noise_enable = True
    action_noise_std = 0.3  # rad

    # -------------------- 4. EXTERNAL PUSHES ------------------------
    push_enable = True
    push_interval_s = 2.0
    push_force_range = [-70.0, 70.0]  # N (x, y sampled independently)

    # (optional, used by the new push impl; safe defaults)
    push_prob = 1.0
    push_duration_s = 0.15
    push_z_scale = 0.0
    push_direction_mode = "random"

    # -------------------- 5. INITIAL POSE PERTURBATION --------------
    init_pose_enable = True
    init_pos_z_range = [0.42, 0.42]
    init_euler_range = [0.0, 0.0]

    # -------------------- 6. MASS & COM SHIFT -----------------------
    mass_enable = False
    mass_shift_range = [-0.5, 0.5]
    com_shift_range = [-0.02, 0.02]

    simulate_action_latency = True

    env_cfg = {
        "num_actions": 12,

        "kp": kp_nominal,
        "kd": kd_nominal,

        "simulate_action_latency": simulate_action_latency,

        "default_joint_angles": {  # [rad]
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
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],

        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "termination_if_z_vel_greater_than": 100.0,
        "termination_if_y_vel_greater_than": 100.0,

        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],

        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "clip_actions": 100.0,
    }

    if friction_enable:
        env_cfg["friction_range"] = friction_range

    if kp_kd_enable:
        env_cfg["kp_range"] = kp_range
        env_cfg["kd_range"] = kd_range

    if obs_noise_enable:
        env_cfg["obs_noise"] = obs_noise
        env_cfg["obs_noise_level"] = obs_noise_level

    if action_noise_enable:
        env_cfg["action_noise_std"] = action_noise_std

    if push_enable:
        env_cfg["push_force_range"] = push_force_range
        env_cfg["push_interval_s"] = push_interval_s
        env_cfg["push_prob"] = push_prob
        env_cfg["push_duration_s"] = push_duration_s
        env_cfg["push_z_scale"] = push_z_scale
        env_cfg["push_direction_mode"] = push_direction_mode

    if init_pose_enable:
        env_cfg["init_pos_z_range"] = init_pos_z_range
        env_cfg["init_euler_range"] = init_euler_range

    if mass_enable:
        env_cfg["mass_shift_range"] = mass_shift_range
        env_cfg["com_shift_range"] = com_shift_range

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
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 1.0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
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
