# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    wrap_to_pi,
    matrix_from_quat,
    quat_apply,  # << MOD: Added import
    normalize,   # << MOD: Added import
)

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # --- Load the pre-defined optimal raceline ---
        # >> MOD: This is no longer used for the 'progress' reward, but is
        #         kept in case other parts of your code rely on it.
        #         If not, you can remove this section.
        raceline_file = "/home/selinawan/ESE651/DroneRacer/src/isaac_quad_sim2real/tasks/race/config/crazyflie/raceline.npy" # Make sure this file is in the same directory
        if not os.path.exists(raceline_file):
            raise FileNotFoundError(
                f"Raceline file not found: {raceline_file}. "
                "Please run `visualize_racing_line.py` first to generate it."
            )
        self.raceline_points = torch.from_numpy(np.load(raceline_file)).to(self.device)
        self.num_raceline_points = self.raceline_points.shape[0]

        # --- MOD: State variable for 'progress' reward ---
        # This stores the drone's position from the previous step.
        self._prev_drone_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # State variables for gate-passing
        self._gate_side_half = self.cfg.gate_model.gate_side / 2.0
        self._gate_pass_threshold = self._gate_side_half + 0.2 # Allow 20cm margin
        
        # --- MOD: Hyperparameter for 'lookat_next_gate' reward ---
        # This 'std' controls the sharpness of the exponential reward.
        # You should tune this value. It can be added to your config file.
        self.lookat_std = getattr(self.cfg, "lookat_reward_std", 1.0)


        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, "rew"):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in keys
            }
            # Add any new keys that might not be in the config yet
            # >> MOD: Removed 'progress_raceline' and 'dist_from_gate'
            #         Added 'progress_to_gate'
            new_keys = [
                "progress_to_gate", "pass_gate", "time", "action_rate", 
                "ang_vel", "alignment", "lap_complete"
            ]
            for key in new_keys:
                if key not in self._episode_sums:
                    self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Initialize fixed parameters once (no domain randomization)
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value
        self.env._tau_m[:] = self.env._tau_m_value
        self.env._thrust_to_weight[:] = self.env._twr_value
        # Assume a gate radius for passing logic
        self.gate_radius = 0.5 # m, reasonable guess for inner radius/half-width

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b
        num_gates = self.env._waypoints.shape[0]

        # Get position of the *current* target gate
        target_gate_pos_w = self.env._waypoints[self.env._idx_wp, :3]

        # compute crashed environments if contact detected for 100 timesteps
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 0.1).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # --- 3. Progress Reward ---
        # This rewards the drone for *reducing its distance* to the *current* target gate.
        prev_distance_to_gate = torch.norm(self._prev_drone_pos_w - target_gate_pos_w, dim=1)
        current_distance_to_gate = torch.norm(drone_pos_w - target_gate_pos_w, dim=1)
        
        reward_progress_to_gate = prev_distance_to_gate - current_distance_to_gate

        # --- Gate Passing Logic ---
        x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0]
        alignment_drone_wrt_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate[:, 1:], dim=1)

        gate_passed = (x_drone_wrt_gate > 0.0) & \
                    (self.env._prev_x_drone_wrt_gate <= 0.0) & \
                    (alignment_drone_wrt_gate < self._gate_pass_threshold)
        
        ids_gate_passed = torch.where(gate_passed)[0]
        num_gates = self.env._waypoints.shape[0]

        gate_pass_reward = torch.zeros(self.num_envs, device=self.device)
        lap_complete_reward = torch.zeros(self.num_envs, device=self.device)

        if len(ids_gate_passed) > 0:
            gate_pass_reward[ids_gate_passed] = 1.0 # self.env.rew['gate_pass_reward_scale']
            self.env._n_gates_passed[ids_gate_passed] += 1
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % num_gates
            self.env._desired_pos_w[ids_gate_passed, :] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]

            # Reward for partial completion: percentage of current lap
            # This promotes sequential gate passing by rewarding progress through laps
            # After incrementing, this represents the new progress
            current_lap_progress = (self.env._n_gates_passed[ids_gate_passed]).float() / num_gates
            lap_complete_reward[ids_gate_passed] = current_lap_progress

            self.env._desired_pos_w[ids_gate_passed, :] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]            
            target_gate_pos_w = self.env._waypoints[self.env._idx_wp, :3]

        # Store current X-pos for next step's check
        self.env._prev_x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0].detach()

        # --- 4. Time Penalty (Unchanged) ---
        reward_time = torch.ones(self.num_envs, device=self.device) # 1.0, to be scaled by negative value in config

        # --- 5. Stability / Action Penalties ---
        # MOD: Use L2-squared norm for angular velocity (from `ang_vel_l2`)
        penalty_ang_vel = torch.sum(torch.square(drone_ang_vel_b), dim=1)
        
        # (Unchanged) Penalize jerky actions
        action_rate = torch.linalg.norm(self.env._actions - self.env._previous_actions, dim=1)
        penalty_action_rate = action_rate

        # --- 6. MOD: Alignment Reward (from `lookat_next_gate` function) ---
        # Replaces the simpler cosine similarity reward.
        
        # Get position of next gate (handles wraparound)
        # next_wp_idx = (self.env._idx_wp + 1) % num_gates
        # next_gate_pos_w = self.env._waypoints[next_wp_idx, :3]
        
        # Get vector from drone to next gate
        vec_to_next_gate_w = target_gate_pos_w - drone_pos_w
        vec_to_next_gate_w = normalize(vec_to_next_gate_w)
        
        # Get drone's forward-facing vector (X-axis in body frame)
        x_axis_b = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        drone_forward_vec_w = quat_apply(drone_quat_w, x_axis_b)
        # Normalization is technically redundant if quat is unit, but good practice
        drone_forward_vec_w = normalize(drone_forward_vec_w) 
        
        # Calculate alignment (dot product)
        dot = torch.sum(drone_forward_vec_w * vec_to_next_gate_w, dim=1).clamp(-1.0, 1.0)
        
        # Use exponential kernel from new script
        angle = torch.acos(dot)
        reward_alignment = torch.exp(-angle / self.lookat_std)

        # --- 7. Sum all rewards ---
        if self.cfg.is_train:
            # >> MOD: Updated reward dictionary
            rewards = {
                "progress_to_gate": reward_progress_to_gate * self.env.rew["progress_to_gate_reward_scale"],
                "pass_gate": gate_pass_reward * self.env.rew["pass_gate_reward_scale"],
                "lap_complete": lap_complete_reward * self.env.rew["lap_complete_reward_scale"],
                "time": reward_time * self.env.rew["time_reward_scale"],
                "crash": crashed * self.env.rew["crash_reward_scale"],
                "action_rate": penalty_action_rate * self.env.rew["action_rate_reward_scale"],
                "ang_vel": penalty_ang_vel * self.env.rew["ang_vel_reward_scale"],
                "alignment": reward_alignment * self.env.rew["alignment_reward_scale"],
            }
            # NOTE: You will need to add `progress_to_gate_reward_scale`
            # to your environment config file.
            
            # You might want to remove `dist_from_gate_reward_scale` as the
            # new `progress_to_gate` reward serves a similar purpose.

            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

            # Apply death cost
            reward = torch.where(
                self.env.reset_terminated, torch.ones_like(reward) * self.env.rew["death_cost"], reward
            )

            # Logging
            for key, value in rewards.items():
                if key in self._episode_sums:
                    self._episode_sums[key] += value
                
        else:  # eval mode
            reward = torch.zeros(self.num_envs, device=self.device)

        # --- MOD: Update state variable for next step ---
        # Store the current drone position to be used in the next step's progress calculation
        self._prev_drone_pos_w[:] = drone_pos_w.clone()

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for the racing policy. (Unchanged)"""

        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b
        
        num_gates = self.env._waypoints.shape[0]

        # --- Current Target Gate ---
        drone_pos_curr_gate_frame = self.env._pose_drone_wrt_gate

        # --- Next Target Gate ---
        next_wp_idx = (self.env._idx_wp + 1) % num_gates
        next_gate_pos_w = self.env._waypoints[next_wp_idx, :3]
        next_gate_quat_w = self.env._waypoints_quat[next_wp_idx, :]
        
        drone_pos_next_gate_frame, _ = subtract_frame_transforms(
            next_gate_pos_w, next_gate_quat_w, drone_pos_w
        )

        obs = torch.cat(
            [
                drone_lin_vel_b,            # Drone's linear velocity in its own frame (3)
                drone_ang_vel_b,            # Drone's angular velocity in its own frame (3)
                drone_pos_curr_gate_frame,  # Position relative to current target gate (3)
                drone_pos_next_gate_frame,  # Position relative to next target gate (3)
                self.env._previous_actions,  # Previous action (4)
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            # >> MOD: Update logging keys
            for key in self._episode_sums.keys():
                if key in self.env.rew or key in [
                    "progress_to_gate", "pass_gate", "lap_complete", "time", "action_rate", 
                    "ang_vel", "alignment", "crash",
                ]:
                    episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                    extras["Episode_Reward/" + key] = episodic_sum_avg
                    self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            extras["Episode_Stats/gates_passed"] = torch.mean(self.env._n_gates_passed[env_ids].float())
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # (Rest of the method is unchanged...)
        
        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # --- Reset drone to a position behind the first gate ---
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
      
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, 0.0)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)
        z_local = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.5)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp.clamp(min=0.2) # Ensure min altitude

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-10.0 * D2R, 10.0 * D2R)
        roll_noise = torch.empty(n_reset, device=self.device).uniform_(-5.0 * D2R, 5.0 * D2R)
        pitch_noise = torch.empty(n_reset, device=self.device).uniform_(-5.0 * D2R, 5.0 * D2R)

        quat = quat_from_euler_xyz(roll_noise, pitch_noise, initial_yaw + yaw_noise)
        default_root_state[:, 3:7] = quat

        # (Handle play mode... unchanged)
        if not self.cfg.is_train:
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)
            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)
            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0
            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices
        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        
        # --- MOD: Reset the 'previous position' state variable ---
        self._prev_drone_pos_w[env_ids] = default_root_state[:, :3].clone()

        # --- MOD: Remove raceline progress reset ---
        # This is no longer needed for the new reward.
        # dist_to_raceline_pts = torch.linalg.norm(self.raceline_points.unsqueeze(0) - drone_pos_w.unsqueeze(1), dim=2)
        # closest_raceline_idx = torch.argmin(dist_to_raceline_pts, dim=1)
        # self._closest_raceline_idx[env_ids] = closest_raceline_idx
        # self._last_raceline_progress[env_ids] = closest_raceline_idx

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0
        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )
        # self.env._prev_x_drone_wrt_gate = torch.ones(self.num_envs, device=self.device)
        self.env._crashed[env_ids] = 0