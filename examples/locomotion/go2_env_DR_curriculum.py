import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    """
    Updated:
      - Curriculum (level-based, auto-adjusts using episode success rate)
      - DR schedule tied to curriculum levels
      - Initial DOF randomization on reset
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        # control
        self.dt = 0.02
        self.simulate_action_latency = bool(env_cfg.get("simulate_action_latency", True))
        self.action_scale = float(env_cfg.get("action_scale", 0.25))
        self.episode_length_s = float(env_cfg["episode_length_s"])
        self.max_episode_length = math.ceil(self.episode_length_s / self.dt)

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # ------------------ Curriculum config ------------------
        cur_cfg = env_cfg.get("curriculum", {})
        self.curriculum_enabled = bool(cur_cfg.get("enabled", False))

        # A list of per-level dicts:
        # levels[i] can override command ranges and DR ranges, pushes, latency, etc.
        self.curriculum_levels = cur_cfg.get("levels", [])
        if self.curriculum_enabled and (not isinstance(self.curriculum_levels, list) or len(self.curriculum_levels) == 0):
            raise ValueError("env_cfg['curriculum']['enabled']=True but no curriculum levels provided.")

        self.curr_level = int(cur_cfg.get("start_level", 0))
        self.curr_level = max(0, min(self.curr_level, len(self.curriculum_levels) - 1)) if self.curriculum_enabled else 0

        # How often to evaluate success and potentially change difficulty
        self.curr_window_episodes = int(cur_cfg.get("window_episodes", 2048))
        self.curr_up_threshold = float(cur_cfg.get("success_threshold_up", 0.80))
        self.curr_down_threshold = float(cur_cfg.get("success_threshold_down", 0.50))

        # Counters (python ints; cheap + stable)
        self._curr_done = 0
        self._curr_success = 0
        self._curr_last_success_rate = 0.0

        # ------------------ Domain Randomization Config ------------------
        # Defaults; will be overwritten by curriculum level if enabled
        self.friction_min, self.friction_max = env_cfg.get("friction_range", (0.4, 1.2))
        self.kp_nom = float(env_cfg["kp"])
        self.kd_nom = float(env_cfg["kd"])
        self.kp_scale_min, self.kp_scale_max = env_cfg.get("kp_scale_range", (1.0, 1.0))
        self.kd_scale_min, self.kd_scale_max = env_cfg.get("kd_scale_range", (1.0, 1.0))

        # Init joint-angle randomization (uniform noise around default)
        self.init_dof_pos_noise = float(env_cfg.get("init_dof_pos_noise", 0.0))  # radians

        # Push disturbances (will be overridden by curriculum if enabled)
        self.push_enable = bool(env_cfg.get("push_enable", True))
        self.push_interval_s = float(env_cfg.get("push_interval_s", 2.0))
        self.push_prob = float(env_cfg.get("push_prob", 0.15))
        self.push_force_range = env_cfg.get("push_force_range", (50.0, 150.0))  # N
        self.push_z_scale = float(env_cfg.get("push_z_scale", 0.0))
        self.push_duration_steps = int(env_cfg.get("push_duration_s", 0.15) / self.dt)
        self.push_direction_mode = env_cfg.get("push_direction_mode", "random")

        # Toggles: allow you to disable specific DR components per level
        self.rand_friction_enable = bool(env_cfg.get("rand_friction_enable", True))
        self.rand_pd_enable = bool(env_cfg.get("rand_pd_enable", True))

        # DR log buffers
        self.friction_last = torch.ones((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)
        self.kp_last = torch.ones((self.num_envs, 1), device=gs.device, dtype=gs.tc_float) * self.kp_nom
        self.kd_last = torch.ones((self.num_envs, 1), device=gs.device, dtype=gs.tc_float) * self.kd_nom

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
                # REQUIRED for batched DR APIs
                batch_dofs_info=True,
                batch_links_info=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device, dtype=gs.tc_float)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device, dtype=gs.tc_float)
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

        # base link solver idx (you said 1)
        self._base_link_idx = 1

        # push state
        self._push_timer = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_int)
        self._push_force_cache = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # rewards
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            fn = getattr(self, "_reward_" + name)
            self.reward_functions[name] = fn
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(self.num_envs, 1)

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

        # Apply initial curriculum level overrides (commands + DR + pushes + latency + action scale)
        if self.curriculum_enabled:
            self.apply_curriculum_level(self.curr_level)

        # Initial DR once after build
        all_envs = torch.arange(self.num_envs, device=gs.device)
        self._apply_dr_on_reset(all_envs)

    # ------------------ Curriculum helpers ------------------

    def apply_curriculum_level(self, level: int):
        """Apply per-level knobs: command ranges, DR ranges, pushes, latency, action_scale, init noise, episode length."""
        if not self.curriculum_enabled:
            return
        level = int(level)
        level = max(0, min(level, len(self.curriculum_levels) - 1))
        self.curr_level = level

        cfg = self.curriculum_levels[level]

        # Command ranges (stored in command_cfg dict used by _resample_commands)
        if "lin_vel_x_range" in cfg:
            self.command_cfg["lin_vel_x_range"] = list(cfg["lin_vel_x_range"])
        if "lin_vel_y_range" in cfg:
            self.command_cfg["lin_vel_y_range"] = list(cfg["lin_vel_y_range"])
        if "ang_vel_range" in cfg:
            self.command_cfg["ang_vel_range"] = list(cfg["ang_vel_range"])

        # DR ranges + toggles
        if "friction_range" in cfg:
            self.friction_min, self.friction_max = cfg["friction_range"]
        if "kp_scale_range" in cfg:
            self.kp_scale_min, self.kp_scale_max = cfg["kp_scale_range"]
        if "kd_scale_range" in cfg:
            self.kd_scale_min, self.kd_scale_max = cfg["kd_scale_range"]

        if "rand_friction_enable" in cfg:
            self.rand_friction_enable = bool(cfg["rand_friction_enable"])
        if "rand_pd_enable" in cfg:
            self.rand_pd_enable = bool(cfg["rand_pd_enable"])

        # Push knobs
        if "push_enable" in cfg:
            self.push_enable = bool(cfg["push_enable"])
        if "push_interval_s" in cfg:
            self.push_interval_s = float(cfg["push_interval_s"])
        if "push_prob" in cfg:
            self.push_prob = float(cfg["push_prob"])
        if "push_force_range" in cfg:
            self.push_force_range = tuple(cfg["push_force_range"])
        if "push_z_scale" in cfg:
            self.push_z_scale = float(cfg["push_z_scale"])
        if "push_duration_s" in cfg:
            self.push_duration_steps = int(float(cfg["push_duration_s"]) / self.dt)
        if "push_direction_mode" in cfg:
            self.push_direction_mode = cfg["push_direction_mode"]

        # Latency + action scale
        if "simulate_action_latency" in cfg:
            self.simulate_action_latency = bool(cfg["simulate_action_latency"])
        if "action_scale" in cfg:
            self.action_scale = float(cfg["action_scale"])

        # Init joint noise
        if "init_dof_pos_noise" in cfg:
            self.init_dof_pos_noise = float(cfg["init_dof_pos_noise"])

        # Episode length (optional)
        if "episode_length_s" in cfg:
            self.episode_length_s = float(cfg["episode_length_s"])
            self.max_episode_length = math.ceil(self.episode_length_s / self.dt)

    def _maybe_update_curriculum_from_stats(self):
        """Called after accumulating enough finished episodes (done count)."""
        if not self.curriculum_enabled:
            return

        if self._curr_done < self.curr_window_episodes:
            return

        success_rate = float(self._curr_success) / float(max(1, self._curr_done))
        self._curr_last_success_rate = success_rate

        new_level = self.curr_level
        if success_rate >= self.curr_up_threshold and self.curr_level < len(self.curriculum_levels) - 1:
            new_level += 1
        elif success_rate <= self.curr_down_threshold and self.curr_level > 0:
            new_level -= 1

        # reset window
        self._curr_done = 0
        self._curr_success = 0

        if new_level != self.curr_level:
            self.apply_curriculum_level(new_level)
            # optional: re-DR everyone after level switch (keeps things consistent)
            all_envs = torch.arange(self.num_envs, device=gs.device)
            self._apply_dr_on_reset(all_envs)

    # ------------------ DR helpers ------------------

    def _apply_dr_on_reset(self, envs_idx):
        """Apply all enabled DR components at reset."""
        if envs_idx.numel() == 0:
            return
        if self.rand_friction_enable:
            self.randomize_friction(envs_idx)
        if self.rand_pd_enable:
            self.randomize_pd_gains(envs_idx)

    def randomize_friction(self, envs_idx):
        """Randomize CONTACT friction scaling (per env) and broadcast to all links."""
        if envs_idx.numel() == 0:
            return
        num = int(envs_idx.numel())
        mu = gs_rand_float(self.friction_min, self.friction_max, (num, 1), gs.device).to(gs.tc_float)
        self.friction_last[envs_idx] = mu

        friction_ratio = mu.repeat(1, self.robot.n_links)  # (num_envs_reset, n_links)
        self.robot.set_friction_ratio(
            friction_ratio=friction_ratio,
            links_idx_local=range(self.robot.n_links),
            envs_idx=envs_idx,
        )

    def randomize_pd_gains(self, envs_idx):
        """Randomize KP/KD per env for controlled dofs."""
        if envs_idx.numel() == 0:
            return
        num = int(envs_idx.numel())

        kp_scale = gs_rand_float(self.kp_scale_min, self.kp_scale_max, (num, 1), gs.device).to(gs.tc_float)
        kd_scale = gs_rand_float(self.kd_scale_min, self.kd_scale_max, (num, 1), gs.device).to(gs.tc_float)

        kp_vals = kp_scale * self.kp_nom
        kd_vals = kd_scale * self.kd_nom

        self.kp_last[envs_idx] = kp_vals
        self.kd_last[envs_idx] = kd_vals

        kp_matrix = kp_vals.repeat(1, self.num_actions)
        kd_matrix = kd_vals.repeat(1, self.num_actions)

        self.robot.set_dofs_kp(kp_matrix, self.motors_dof_idx, envs_idx=envs_idx)
        self.robot.set_dofs_kv(kd_matrix, self.motors_dof_idx, envs_idx=envs_idx)

    def maybe_apply_push(self):
        """Multi-step push window."""
        if not self.push_enable:
            return

        interval_steps = max(1, int(self.push_interval_s / self.dt))

        start_mask = (
            (self._push_timer == 0)
            & (self.episode_length_buf % interval_steps == 0)
            & (torch.rand(self.num_envs, device=gs.device) < self.push_prob)
        )
        start_envs = start_mask.nonzero(as_tuple=False).squeeze(-1)

        if start_envs.numel() > 0:
            n = int(start_envs.numel())

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
                    links_idx=[self._base_link_idx],
                    envs_idx=active_envs.detach().cpu().numpy(),
                )

            self._push_timer[active_envs] -= 1

    # -----------------------------------------------

    def _resample_commands(self, envs_idx):
        if envs_idx.numel() == 0:
            return
        n = int(envs_idx.numel())
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (n,), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (n,), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (n,), gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        target_dof_pos = exec_actions * self.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        # push before stepping
        self.maybe_apply_push()

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
        resample_every = int(self.env_cfg["resampling_time_s"] / self.dt)
        envs_idx = (self.episode_length_buf % resample_every == 0).nonzero(as_tuple=False).reshape((-1,))
        self._resample_commands(envs_idx)

        # termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        if "termination_if_z_vel_greater_than" in self.env_cfg:
            self.reset_buf |= torch.abs(self.base_lin_vel[:, 2]) > self.env_cfg["termination_if_z_vel_greater_than"]
        if "termination_if_y_vel_greater_than" in self.env_cfg:
            self.reset_buf |= torch.abs(self.base_lin_vel[:, 1]) > self.env_cfg["termination_if_y_vel_greater_than"]

        # Timeouts (used as "success" signal)
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        if time_out_idx.numel() > 0:
            self.extras["time_outs"][time_out_idx] = 1.0

        # ---- curriculum stats collection (before reset) ----
        done_envs = self.reset_buf.nonzero(as_tuple=False).reshape((-1,))
        if done_envs.numel() > 0:
            # success if timeout, failure otherwise
            success_envs = self.extras["time_outs"][done_envs] > 0.5
            self._curr_done += int(done_envs.numel())
            self._curr_success += int(success_envs.sum().item())
            self._maybe_update_curriculum_from_stats()

        # Reset envs
        self.reset_idx(done_envs)

        # reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # obs
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        # handy logs
        self.extras["friction_mean"] = float(self.friction_last.mean().item())
        self.extras["kp_mean"] = float(self.kp_last.mean().item())
        self.extras["kd_mean"] = float(self.kd_last.mean().item())

        # curriculum logs
        self.extras["curriculum_level"] = int(self.curr_level)
        self.extras["curriculum_success_rate_window"] = float(self._curr_last_success_rate)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if envs_idx.numel() == 0:
            return

        # DR at reset
        self._apply_dr_on_reset(envs_idx)

        # reset dofs (with optional init noise)
        n = int(envs_idx.numel())
        dof_pos = self.default_dof_pos.repeat(n, 1)

        if self.init_dof_pos_noise > 0.0:
            noise = gs_rand_float(-self.init_dof_pos_noise, self.init_dof_pos_noise, (n, self.num_actions), gs.device).to(gs.tc_float)
            dof_pos = dof_pos + noise

        self.dof_pos[envs_idx] = dof_pos
        self.dof_vel[envs_idx] = 0.0

        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # episode logging
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / max(1e-6, self.episode_length_s)
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # ---------------- reward functions ----------------

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
