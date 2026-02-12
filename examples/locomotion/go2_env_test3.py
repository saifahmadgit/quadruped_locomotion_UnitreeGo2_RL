# =========================
# File: go2_env_test3.py
# =========================

import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def euler_to_quat_wxyz(roll, pitch, yaw):
    """Convert batched euler angles (rad) to quaternion (w,x,y,z).
    All inputs should be 1-D tensors of the same length."""
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=30,
            ),
            show_viewer=show_viewer,
        )

        # add ground
        self.ground = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters (nominal; overwritten per-episode if kp_range/kd_range set)
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # ==========================================================
        # Domain randomisation setup
        # ==========================================================

        # --- Observation noise std vector (45-dim, pre-scaled) ---
        if "obs_noise" in self.env_cfg:
            nc = self.env_cfg["obs_noise"]
            level = self.env_cfg.get("obs_noise_level", 1.0)
            nv = torch.zeros(self.num_obs, device=gs.device, dtype=gs.tc_float)
            nv[0:3]   = nc.get("ang_vel", 0.0) * self.obs_scales["ang_vel"] * level
            nv[3:6]   = nc.get("gravity", 0.0) * level   # gravity not scaled in obs
            nv[6:9]   = 0.0                                # commands: no sensor noise
            nv[9:21]  = nc.get("dof_pos", 0.0) * self.obs_scales["dof_pos"] * level
            nv[21:33] = nc.get("dof_vel", 0.0) * self.obs_scales["dof_vel"] * level
            nv[33:45] = 0.0                                # last_actions: no noise
            self.obs_noise_vec = nv
        else:
            self.obs_noise_vec = None

        # --- Action noise std ---
        if "action_noise_std" in self.env_cfg:
            self.action_noise_std = self.env_cfg["action_noise_std"]
        else:
            self.action_noise_std = None

        # ==========================================================
        # PUSH (REPLACED IMPLEMENTATION ONLY)
        # ==========================================================
        # Use rigid_solver.apply_links_external_force with per-env timers and cached forces
        self.push_enable = bool(self.env_cfg.get("push_enable", "push_force_range" in self.env_cfg))
        if self.push_enable and ("push_force_range" in self.env_cfg):
            self.push_interval_s = float(self.env_cfg.get("push_interval_s", 5.0))
            self.push_prob = float(self.env_cfg.get("push_prob", 1.0))
            self.push_force_range = self.env_cfg.get("push_force_range", (-70.0, 70.0))
            self.push_z_scale = float(self.env_cfg.get("push_z_scale", 0.0))
            self.push_direction_mode = self.env_cfg.get("push_direction_mode", "random")

            self._push_interval = max(1, int(self.push_interval_s / self.dt))
            self.push_duration_steps = max(1, int(float(self.env_cfg.get("push_duration_s", 0.15)) / self.dt))

            self._push_timer = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_int)
            self._push_force_cache = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

            # base link idx (same convention you used in eval)
            self._base_link_idx = self.robot.links[1].idx
        else:
            self.push_enable = False
        # ==========================================================

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()
        self.extras["observations"] = dict()

    # ==========================================================
    # Domain randomisation — per episode (called at reset)
    # ==========================================================

    def _randomize_friction(self):
        """Surface friction: sampled once, applied globally."""
        if "friction_range" not in self.env_cfg:
            return
        low, high = self.env_cfg["friction_range"]
        mu = float(gs_rand_float(low, high, (1,), gs.device).item())
        self.ground.set_friction(mu)
        self.robot.set_friction(mu)
        self.extras.setdefault("domain_randomization", {})
        self.extras["domain_randomization"]["friction"] = mu

    def _randomize_kp_kd(self):
        """PD gains: sampled once, applied globally.
        Also covers motor strength variation."""
        if "kp_range" not in self.env_cfg:
            return
        kp = float(gs_rand_float(*self.env_cfg["kp_range"], (1,), gs.device).item())
        kd = float(gs_rand_float(*self.env_cfg["kd_range"], (1,), gs.device).item())
        self.robot.set_dofs_kp([kp] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([kd] * self.num_actions, self.motors_dof_idx)
        self.extras.setdefault("domain_randomization", {})
        self.extras["domain_randomization"]["kp"] = kp
        self.extras["domain_randomization"]["kd"] = kd

    def _randomize_mass(self):
        """Base link mass and CoM: sampled once, applied globally."""
        if "mass_shift_range" in self.env_cfg:
            low, high = self.env_cfg["mass_shift_range"]
            shift = float(gs_rand_float(low, high, (1,), gs.device).item())
            self.robot.set_mass_shift([shift], [0])
            self.extras.setdefault("domain_randomization", {})
            self.extras["domain_randomization"]["mass_shift"] = shift

        if "com_shift_range" in self.env_cfg:
            low, high = self.env_cfg["com_shift_range"]
            dx = float(gs_rand_float(low, high, (1,), gs.device).item())
            dy = float(gs_rand_float(low, high, (1,), gs.device).item())
            dz = float(gs_rand_float(low, high, (1,), gs.device).item())
            self.robot.set_COM_shift([[dx, dy, dz]], [0])
            self.extras.setdefault("domain_randomization", {})
            self.extras["domain_randomization"]["com_shift"] = [dx, dy, dz]

    # ==========================================================
    # PUSH — per step (REPLACED IMPLEMENTATION ONLY)
    # ==========================================================

    def _apply_push(self):
        """
        Batched, multi-step pushes using rigid_solver.apply_links_external_force
        (copied from your working file style).
        """
        if not getattr(self, "push_enable", False):
            return

        interval_steps = self._push_interval

        start_mask = (
            (self._push_timer == 0)
            & (self.episode_length_buf % interval_steps == 0)
            & (torch.rand(self.num_envs, device=gs.device) < self.push_prob)
        )
        start_envs = start_mask.nonzero(as_tuple=False).squeeze(-1)

        if start_envs.numel() > 0:
            n = start_envs.numel()

            fmin, fmax = self.push_force_range
            mag = gs_rand_float(fmin, fmax, (n, 1), gs.device).to(gs.tc_float)

            if self.push_direction_mode == "fixed":
                force_dir = torch.zeros((n, 3), device=gs.device, dtype=gs.tc_float)
                force_dir[:, 1] = 1.0
            else:
                theta = gs_rand_float(0.0, 2.0 * math.pi, (n, 1), gs.device)
                force_dir = torch.zeros((n, 3), device=gs.device, dtype=gs.tc_float)
                force_dir[:, 0] = torch.cos(theta[:, 0])
                force_dir[:, 1] = torch.sin(theta[:, 0])

            force_dir[:, 2] = self.push_z_scale
            force = force_dir * mag

            self._push_force_cache[start_envs] = force
            self._push_timer[start_envs] = self.push_duration_steps

        active_envs = (self._push_timer > 0).nonzero(as_tuple=False).squeeze(-1)
        if active_envs.numel() > 0:
            try:
                self.scene.sim.rigid_solver.apply_links_external_force(
                    force=self._push_force_cache[active_envs],
                    links_idx=[self._base_link_idx],
                    envs_idx=active_envs,
                )
            except TypeError:
                self.scene.sim.rigid_solver.apply_links_external_force(
                    force=self._push_force_cache[active_envs].detach().cpu().numpy(),
                    links_idx=[int(self._base_link_idx)],
                    envs_idx=active_envs.detach().cpu().numpy(),
                )

            self._push_timer[active_envs] -= 1

    # ==========================================================
    # Noise
    # ==========================================================

    def _add_obs_noise(self):
        """Additive Gaussian noise on observation buffer (every step)."""
        if self.obs_noise_vec is not None:
            self.obs_buf += torch.randn_like(self.obs_buf) * self.obs_noise_vec

    # ==========================================================

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # --- Per-step DR: action noise (before PD controller) ---
        if self.action_noise_std is not None:
            target_dof_pos = target_dof_pos + torch.randn_like(target_dof_pos) * self.action_noise_std

        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        # --- Per-step DR: push force ---
        self._apply_push()

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= torch.abs(self.base_lin_vel[:, 2]) > self.env_cfg["termination_if_z_vel_greater_than"]
        self.reset_buf |= torch.abs(self.base_lin_vel[:, 1]) > self.env_cfg["termination_if_y_vel_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        # --- Per-step DR: sensor noise ---
        self._add_obs_noise()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # ==========================================================
        # Per-episode domain randomisation
        # ==========================================================
        self._randomize_friction()
        self._randomize_kp_kd()
        self._randomize_mass()

        # (push impl only) clear any active pushes on reset envs
        if getattr(self, "push_enable", False):
            self._push_timer[envs_idx] = 0
            self._push_force_cache[envs_idx] = 0.0

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base (with optional pose perturbation)
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        if "init_pos_z_range" in self.env_cfg:
            low, high = self.env_cfg["init_pos_z_range"]
            self.base_pos[envs_idx, 2] = gs_rand_float(low, high, (len(envs_idx),), gs.device)

        if "init_euler_range" in self.env_cfg:
            low_deg, high_deg = self.env_cfg["init_euler_range"]
            low_rad = math.radians(low_deg)
            high_rad = math.radians(high_deg)
            n = len(envs_idx)
            roll = gs_rand_float(low_rad, high_rad, (n,), gs.device)
            pitch = gs_rand_float(low_rad, high_rad, (n,), gs.device)
            yaw = torch.zeros(n, device=gs.device, dtype=gs.tc_float)
            self.base_quat[envs_idx] = euler_to_quat_wxyz(roll, pitch, yaw)

        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    ######################################## helper function

    ################## reward written for jump task

    def _reward_jump_impulse(self):
        vz = self.base_lin_vel[:, 2]
        z = self.base_pos[:, 2]
        gate = (z < 0.50).float()
        return gate * torch.clamp(vz, min=0.0)

    def _reward_jump_apex(self):
        target = self.reward_cfg["jump_apex_height"]
        sigma = self.reward_cfg.get("jump_apex_sigma", 0.05)
        z = self.base_pos[:, 2]
        return torch.exp(-torch.square((z - target) / sigma))

    def _reward_xy_stability(self):
        v = self.robot.get_vel()
        return -(v[:, 0] ** 2 + v[:, 1] ** 2)

    def _reward_orientation(self):
        return -self.projected_gravity[:, 2]

    def _reward_no_shake(self):
        return -torch.sum(self.base_ang_vel ** 2, dim=1) / 1.0

    def _reward_crouch(self):
        z = self.base_pos[:, 2]
        return (z < 0.25).float()

    ################## reward written for crouch task

    def _reward_crouch_2(self):
        z = self.base_pos[:, 2]
        in_band = (z <= 0.30) & (z >= 0.20)
        return in_band.float()

    def _reward_ground_penalty(self):
        z = self.base_pos[:, 2]
        z_safe = 0.15
        z_min = 0.05
        x = (z_safe - z) / (z_safe - z_min)
        x = torch.clamp(x, 0.0, 1.0)
        return -(x ** 2)

    def _reward_crouch_target(self):
        z = self.base_pos[:, 2]
        z_t = 0.15
        sigma = 0.03
        return torch.exp(-((z - z_t) / sigma) ** 2)

    def _reward_no_fall(self):
        vz = self.base_lin_vel[:, 2]
        thresh = 0.5
        downward = torch.clamp(-vz - thresh, min=0.0)
        return -(downward ** 2)

    def _reward_y_stability(self):
        v = self.robot.get_vel()
        return -(v[:, 1] ** 2)

    def _reward_torque_load(self):
        tau = self.robot.get_dofs_control_force(self.motors_dof_idx)
        return -0.001 * torch.sum(torch.abs(tau), dim=1)

    def _reward_crouch_progress(self):
        z = self.base_pos[:, 2]
        return torch.clamp(0.35 - z, min=0.0)

    def _reward_crouch_speed(self):
        vz = self.base_lin_vel[:, 2]
        return -(vz ** 2)
