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

from go2_env_test2 import Go2Env


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
    # ---------------- DR maxima (your existing knobs) ----------------
    friction_enable = True
    friction_range = [0.4, 1.2]  # HARD max

    kp_kd_enable = True
    kp_nominal = 60.0
    kd_nominal = 2.0
    kp_range = [50.0, 70.0]  # HARD max
    kd_range = [2.0, 5.0]    # HARD max

    obs_noise_enable = True
    obs_noise_level = 0.2  # HARD max
    obs_noise = {
        "ang_vel": 0.2,
        "gravity": 0.05,
        "dof_pos": 0.02,
        "dof_vel": 1.0,
    }

    action_noise_enable = True
    action_noise_std = 0.3  # HARD max

    push_enable = True
    push_interval_s = 2.0   # HARD max intensity (more frequent)
    push_force_range = [-80.0, 80.0]  # HARD max

    init_pose_enable = True
    init_pos_z_range = [0.42, 0.42]
    init_euler_range = [0.0, 0.0]

    mass_enable = False
    mass_shift_range = [-0.5, 0.5]
    com_shift_range = [-0.02, 0.02]

    simulate_action_latency = True

    # ---------------- Curriculum (Option 2) ----------------
    curriculum_enable = True
    curriculum_cfg = {
        "enabled": curriculum_enable,

        # Level (0 easy -> 1 hard)
        "level_init": 0.10,
        "level_min": 0.0,
        "level_max": 1.0,

        # EMA smoothing
        "ema_alpha": 0.03,

        # Update gating based on episode aggregates
        # Increase when: stable + good tracking
        "ready_timeout_rate": 0.90,  # when 70 percent stable
        "ready_tracking": 0.70,      # tracking_per_sec in [0..~1.2] for your scales
        "ready_fall_rate": 0.30,
        "ready_streak": 2,

        # Decrease when: too many failures
        "hard_fall_rate": 0.35,
        "hard_streak": 2,

        # Step sizes
        "step_up": 0.02,
        "step_down": 0.03,
        "cooldown_updates": 2,

        # How often to check/update (in #episodes across all envs)
        # With 4096 envs, 2048 gives reasonably frequent updates without thrashing.
        "update_every_episodes": 4096,

        # Mixture at reset-time sampling for friction/kp/kd
        # Note: if Genesis setters are global, mixture is approximate (still useful).
        "mix_prob_current": 0.80,
        "mix_level_low": 0.00,
        "mix_level_high": 0.50,

        # Explicit "easy" ranges (optional overrides)
        "friction_easy": [0.7, 0.8],
        "kp_easy": [0.90 * kp_nominal, 1.10 * kp_nominal],
        "kd_easy": [0.75 * kd_nominal, 1.25 * kd_nominal],

        # Push starts later in curriculum; also uses an easy interval initially
        "push_start": 0.30,
        "push_interval_easy_s": 6.0,
    }

    env_cfg = {
        "num_actions": 12,

        "kp": kp_nominal,
        "kd": kd_nominal,

        "simulate_action_latency": simulate_action_latency,

        "foot_names": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],

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

        # Curriculum config goes here (easy to edit)
        "curriculum": curriculum_cfg,
    }

    # Keep your existing DR flags (these define HARD maxima)
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
        env_cfg["push_interval_s"] = push_interval_s
        env_cfg["push_force_range"] = push_force_range
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

            "torque_load": -0.001,
            "dof_acc": 0.0,  # -2.5e-7
            "dof_vel": 0.0, # -0.0005

            "orientation_penalty": -1.0,
            "stand_still": 0.0,  # -0.5
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 1.0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  TRAINING CONFIG (DR maxima + Curriculum Option2)")
    print("=" * 70)

    # DR maxima
    dr_items = {
        "Friction (HARD)":        ("friction_range",   lambda: str(env_cfg["friction_range"])),
        "Kp range (HARD)":        ("kp_range",         lambda: str(env_cfg["kp_range"])),
        "Kd range (HARD)":        ("kd_range",         lambda: str(env_cfg["kd_range"])),
        "Obs noise (HARD)":       ("obs_noise",        lambda: f'level={env_cfg.get("obs_noise_level", 0.0)}  {env_cfg["obs_noise"]}'),
        "Action noise (HARD)":    ("action_noise_std", lambda: f'std={env_cfg["action_noise_std"]} rad'),
        "Pushes (HARD)":          ("push_force_range", lambda: f'{env_cfg["push_force_range"]} N  every {env_cfg["push_interval_s"]}s'),
    }
    for label, (key, fmt) in dr_items.items():
        status = f"ON   {fmt()}" if key in env_cfg else "OFF"
        print(f"  {label:22s}: {status}")

    latency = "1 step (fixed)" if env_cfg.get("simulate_action_latency", True) else "OFF"
    print(f"  {'Action latency':22s}: {latency}")

    # Curriculum
    cc = env_cfg.get("curriculum", {})
    print("-" * 70)
    print(f"  Curriculum enabled     : {cc.get('enabled', False)}")
    if cc.get("enabled", False):
        print(f"  level_init             : {cc.get('level_init')}")
        print(f"  update_every_episodes   : {cc.get('update_every_episodes')}")
        print(f"  ready thresholds        : timeout>={cc.get('ready_timeout_rate')}, "
              f"tracking>={cc.get('ready_tracking')}, fall<={cc.get('ready_fall_rate')}")
        print(f"  hard threshold          : fall>={cc.get('hard_fall_rate')}")
        print(f"  step_up / step_down     : {cc.get('step_up')} / {cc.get('step_down')}")
        print(f"  mix_prob_current        : {cc.get('mix_prob_current')}")
        print(f"  friction_easy           : {cc.get('friction_easy')}")
        print(f"  kp_easy / kd_easy       : {cc.get('kp_easy')} / {cc.get('kd_easy')}")
        print(f"  push_start / easy_int_s  : {cc.get('push_start')} / {cc.get('push_interval_easy_s')}")
    print("=" * 70 + "\n")

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
