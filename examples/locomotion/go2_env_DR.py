import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        # control
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # ------------------ Domain Randomization Config ------------------
        # Contact friction (Coulomb friction scaling at contacts)
        self.friction_min, self.friction_max = env_cfg.get("friction_range", (0.4, 1.2))
        self.friction_last = torch.ones((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)

        # PD gain randomization (around nominal)
        self.kp_nom = float(env_cfg["kp"])
        self.kd_nom = float(env_cfg["kd"])
        self.kp_scale_min, self.kp_scale_max = env_cfg.get("kp_scale_range", (1.0, 1.0))
        self.kd_scale_min, self.kd_scale_max = env_cfg.get("kd_scale_range", (1.0, 1.0))

        self.kp_last = torch.ones((self.num_envs, 1), device=gs.device, dtype=gs.tc_float) * self.kp_nom
        self.kd_last = torch.ones((self.num_envs, 1), device=gs.device, dtype=gs.tc_float) * self.kd_nom

        # External push disturbances
        self.push_enable = bool(env_cfg.get("push_enable", True))
        self.push_interval_s = float(env_cfg.get("push_interval_s", 2.0))
        self.push_prob = float(env_cfg.get("push_prob", 0.15))
        self.push_force_range = env_cfg.get("push_force_range", (50.0, 150.0))  # N
        self.push_z_scale = float(env_cfg.get("push_z_scale", 0.0))
        # -----------------------------------------------------------------

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

                # REQUIRED for batched DR APIs (friction, kp/kd, mass/com shifts, etc.)
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

        # ---------------- 
        # From your link dump: base link solver_idx = 1
        self._base_link_idx = 1

        # Multi-step push (not a 1-tick impulse)
        self.push_duration_steps = int(env_cfg.get("push_duration_s", 0.15) / self.dt)

        self._push_timer = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_int)
        self._push_force_cache = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        # "random" = random horizontal direction
        # "fixed"  = always +Y (debug mode)
        self.push_direction_mode = env_cfg.get("push_direction_mode", "random")
        # --------------------------------------------------------------------

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # buffers
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

        # Apply initial DR once after build (best practice)
        all_envs = torch.arange(self.num_envs, device=gs.device)
        self.randomize_friction(all_envs)
        self.randomize_pd_gains(all_envs)

        # set initial kp/kd (batched) as well
        # (randomize_pd_gains already calls set_dofs_kp/kv)

    # ------------------ DR helpers ------------------

    def randomize_friction(self, envs_idx):
        """
        Randomize CONTACT friction scaling (per env) and broadcast to all links.
        """
        if len(envs_idx) == 0:
            return

        num = len(envs_idx)
        mu = gs_rand_float(self.friction_min, self.friction_max, (num, 1), gs.device)
        self.friction_last[envs_idx] = mu

        friction_ratio = mu.repeat(1, self.robot.n_links)  # (num_envs_reset, n_links)

        self.robot.set_friction_ratio(
            friction_ratio=friction_ratio,
            links_idx_local=range(self.robot.n_links),
            envs_idx=envs_idx,
        )

    def randomize_pd_gains(self, envs_idx):
        """
        Randomize KP/KD per env for the controlled dofs.
        """
        if len(envs_idx) == 0:
            return

        num = len(envs_idx)

        kp_scale = gs_rand_float(self.kp_scale_min, self.kp_scale_max, (num, 1), gs.device)
        kd_scale = gs_rand_float(self.kd_scale_min, self.kd_scale_max, (num, 1), gs.device)

        kp_vals = kp_scale * self.kp_nom  # (num,1)
        kd_vals = kd_scale * self.kd_nom  # (num,1)

        self.kp_last[envs_idx] = kp_vals
        self.kd_last[envs_idx] = kd_vals

        kp_matrix = kp_vals.repeat(1, self.num_actions)  # (num, 12)
        kd_matrix = kd_vals.repeat(1, self.num_actions)  # (num, 12)

        self.robot.set_dofs_kp(kp_matrix, self.motors_dof_idx, envs_idx=envs_idx)
        self.robot.set_dofs_kv(kd_matrix, self.motors_dof_idx, envs_idx=envs_idx)

    def maybe_apply_push(self):
        """
        - Applies force at BASE link center-of-mass (solver_idx=1) via rigid_solver API
        - Multi-step push window for realistic impulse
        - Random or fixed direction (set env_cfg["push_direction_mode"])
        """
        if not self.push_enable:
            return

        interval_steps = max(1, int(self.push_interval_s / self.dt))

        # Start new push windows only at interval boundaries, and only if no active push is running
        start_mask = (
            (self._push_timer == 0)
            & (self.episode_length_buf % interval_steps == 0)
            & (torch.rand(self.num_envs, device=gs.device) < self.push_prob)
        )
        start_envs = start_mask.nonzero(as_tuple=False).squeeze(-1)

        if start_envs.numel() > 0:
            n = start_envs.numel()

            # Magnitude
            fmin, fmax = self.push_force_range
            mag = gs_rand_float(fmin, fmax, (n, 1), gs.device).to(gs.tc_float)

            # Direction
            if self.push_direction_mode == "fixed":
                # Always +Y (debug)
                force_dir = torch.zeros((n, 3), device=gs.device, dtype=gs.tc_float)
                force_dir[:, 1] = 1.0
            else:
                # Random horizontal direction
                theta = gs_rand_float(0.0, 2.0 * math.pi, (n, 1), gs.device)
                force_dir = torch.zeros((n, 3), device=gs.device, dtype=gs.tc_float)
                force_dir[:, 0] = torch.cos(theta[:, 0])  # X
                force_dir[:, 1] = torch.sin(theta[:, 0])  # Y

            # Optional vertical component (usually 0)
            force_dir[:, 2] = self.push_z_scale

            force = force_dir * mag  # (n,3)

            self._push_force_cache[start_envs] = force
            self._push_timer[start_envs] = self.push_duration_steps

        # Apply active pushes
        active_envs = (self._push_timer > 0).nonzero(as_tuple=False).squeeze(-1)
        if active_envs.numel() > 0:
            try:
                self.scene.sim.rigid_solver.apply_links_external_force(
                    force=self._push_force_cache[active_envs],
                    links_idx=[self._base_link_idx],
                    envs_idx=active_envs,
                )
            except TypeError:
                # Some builds want numpy on CPU
                self.scene.sim.rigid_solver.apply_links_external_force(
                    force=self._push_force_cache[active_envs].detach().cpu().numpy(),
                    links_idx=[self._base_link_idx],
                    envs_idx=active_envs.detach().cpu().numpy(),
                )

            self._push_timer[active_envs] -= 1

    # -----------------------------------------------

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
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
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        if "termination_if_z_vel_greater_than" in self.env_cfg:
            self.reset_buf |= torch.abs(self.base_lin_vel[:, 2]) > self.env_cfg["termination_if_z_vel_greater_than"]
        if "termination_if_y_vel_greater_than" in self.env_cfg:
            self.reset_buf |= torch.abs(self.base_lin_vel[:, 1]) > self.env_cfg["termination_if_y_vel_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

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

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # DR at reset (best practice)
        self.randomize_friction(envs_idx)
        self.randomize_pd_gains(envs_idx)

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
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
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
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

################## reward written for crouch task
    ## I am adding another as the another crouch is working fine for jumping and I do not want to disturb that

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
        return torch.exp(-((z - z_t) / sigma)**2)   

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)


    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_no_fall(self):
        vz = self.base_lin_vel[:, 2]
        thresh = 0.5
        downward = torch.clamp(-vz - thresh, min=0.0)
        return -(downward ** 2)

    def _reward_y_stability(self):
        v = self.robot.get_vel()
        return -(v[:,1]**2)

    def _reward_torque_load(self):
        tau = self.robot.get_dofs_control_force(self.motors_dof_idx)
        return -0.001*torch.sum(torch.abs(tau), dim=1)

    def _reward_crouch_progress(self):
        z = self.base_pos[:, 2]
        return torch.clamp(0.35 - z, min=0.0)

    def _reward_crouch_speed(self):
        vz = self.base_lin_vel[:, 2]
        return -(vz ** 2)


################# reward written for jump task

    def _reward_jump_impulse(self):
        # Reward upward speed ONLY during push-off (when we're not already high)
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
        return -(v[:,0]**2 + v[:,1]**2)


    def _reward_orientation(self):
        return -self.projected_gravity[:, 2]

    def _reward_no_shake(self):
        return -torch.sum(self.base_ang_vel**2, dim=1)/1.0

    def _reward_crouch(self):
        z = self.base_pos[:, 2]
        return (z < 0.25).float()