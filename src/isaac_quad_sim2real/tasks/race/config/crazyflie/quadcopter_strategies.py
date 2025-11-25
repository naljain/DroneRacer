# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

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

        # TWR
        self._twr_min = self.cfg.thrust_to_weight * 0.95
        self._twr_max = self.cfg.thrust_to_weight * 1.05
        # Aerodynamics
        self._k_aero_xy_min = self.cfg.k_aero_xy * 0.5
        self._k_aero_xy_max = self.cfg.k_aero_xy * 2.0
        self._k_aero_z_min = self.cfg.k_aero_z * 0.5
        self._k_aero_z_max = self.cfg.k_aero_z * 2.0
        # PID gains
        self._kp_omega_rp_min = self.cfg.kp_omega_rp * 0.85
        self._kp_omega_rp_max = self.cfg.kp_omega_rp * 1.15
        self._ki_omega_rp_min = self.cfg.ki_omega_rp * 0.85
        self._ki_omega_rp_max = self.cfg.ki_omega_rp * 1.15
        self._kd_omega_rp_min = self.cfg.kd_omega_rp * 0.7
        self._kd_omega_rp_max = self.cfg.kd_omega_rp * 1.3
        self._kp_omega_y_min = self.cfg.kp_omega_y * 0.85
        self._kp_omega_y_max = self.cfg.kp_omega_y * 1.15
        self._ki_omega_y_min = self.cfg.ki_omega_y * 0.85
        self._ki_omega_y_max = self.cfg.ki_omega_y * 1.15
        self._kd_omega_y_min = self.cfg.kd_omega_y * 0.7
        self._kd_omega_y_max = self.cfg.kd_omega_y * 1.3

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

        # Previous distancce to goal
        self.prev_distance_to_goal = torch.linalg.norm(
                self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
            )
        self.n_laps_completed = torch.zeros(self.num_envs, device=self.device)
    # def get_rewards(self) -> torch.Tensor:
    #     """get_rewards() is called per timestep. This is where you define your reward structure and compute them
    #     according to the reward scales you tune in train_race.py. The following is an example reward structure that
    #     causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
    #     if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

    #     # TODO ----- START ----- Define the tensors required for your custom reward structure
    #     # check to change waypoint
    #     dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        
    #     #gate_passed = dist_to_gate < 0.1
    #     gate_passed = (self.env._pose_drone_wrt_gate[:, 0] <= 0.0) & \
    #                 (self.env._prev_x_drone_wrt_gate > 0.0) & \
    #                 (dist_to_gate < 0.45)

    #     # Debug prints for play mode
    #     if not self.cfg.is_train:
    #         # Print every N steps to avoid spam (e.g., every 50 steps)
    #         if not hasattr(self, '_debug_step_counter'):
    #             self._debug_step_counter = 0
    #         self._debug_step_counter += 1
            
    #         if self._debug_step_counter % 50 == 0:  # Print every 50 steps
    #             print(f"[DEBUG] Step {self._debug_step_counter}:")
    #             print(f"  dist_to_gate: {dist_to_gate[0].item():.3f} m")
    #             print(f"  gate_passed: {gate_passed[0].item()}")
    #             print(f"  x_drone_wrt_gate: {self.env._pose_drone_wrt_gate[0, 0].item():.3f}")
    #             print(f"  prev_x_drone_wrt_gate: {self.env._prev_x_drone_wrt_gate[0].item():.3f}")
    #             print(f"  current_gate_idx: {self.env._idx_wp[0].item()}")
    #             print(f"  gates_passed_total: {self.env._n_gates_passed[0].item()}")
            
    #         # Always print when a gate is passed
    #         if gate_passed[0].item():
    #             print(f"[GATE PASSED!] Step {self._debug_step_counter}: Gate {self.env._idx_wp[0].item()} passed!")
    #             print(f"  Distance: {dist_to_gate[0].item():.3f} m")
    #             print(f"  Total gates passed: {self.env._n_gates_passed[0].item() + 1}")

    #     ids_gate_passed = torch.where(gate_passed)[0]
        
    #     self.env._n_gates_passed[ids_gate_passed] += 1
    #     self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]

    #     # lap complete reward (passed through the zeroth gate again)
    #     lap_complete_reward = torch.zeros(self.num_envs, device=self.device)
    #     lap_complete_reward[ids_gate_passed] = ((self.env._n_gates_passed[ids_gate_passed] - 1) // self.env._waypoints.shape[0]).float()

    #     # accumualate number of gates passed
    #     # gates_passed_reward = self.env._n_gates_passed.float()
    #     gates_passed_reward = torch.zeros(self.num_envs, device=self.device)
    #     gates_passed_reward[ids_gate_passed] = 1.0

    #     # set desired positions in the world frame
    #     self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
    #     self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

        
    #     # calculate progress via distance to goal
    #     distance_to_goal = torch.linalg.norm(self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1)
    #     distance_to_goal = torch.tanh(distance_to_goal/3.0)
    #     progress = 1 - distance_to_goal  # distance_to_goal is between 0 and 1 where 0 means the drone reached the goal
        
        
    #     # new dense progress to try
    #     distance_to_goal = torch.linalg.norm(self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1)
    #     delta_progress = self.prev_distance_to_goal - distance_to_goal
    #     # update previous distance to goal
    #     self.prev_distance_to_goal = distance_to_goal

    #     """
    #     delta_progress = self.prev_distance_to_goal - distance_to_goal
    #     progress_penalty = torch.where(delta_progress < 0.0, 1.0, 0.0)

    #     self.prev_distance_to_goal = distance_to_goal
    #     """
        
    #     # Alignment reward to next gate
    #     # --- Perception Reward (r_t^perc) --- (this is the alignment reward to the next gate)
    #     # Get vector from drone to gate center in world frame
    #     vec_to_gate_w = self.env._desired_pos_w - self.env._robot.data.root_link_pos_w
        
    #     # Normalize the vector to the gate
    #     # Add epsilon for numerical stability in case of zero vector (at goal)
    #     norm_vec_to_gate = torch.linalg.norm(vec_to_gate_w, dim=1, keepdim=True)
    #     vec_to_gate_dir = vec_to_gate_w / (norm_vec_to_gate + 1e-8)
        
    #     # Get the drone's forward vector (X-axis of the body frame) in world frame
    #     # Extract the rotation matrix from the quaternion
    #     # The forward vector is the first column of the rotation matrix R @ [1, 0, 0]^T
    #     rot_mat = matrix_from_quat(self.env._robot.data.root_quat_w)
    #     forward_vec_w = rot_mat[:, :3, 0] # First column of the rotation matrix
        
    #     # Calculate the angle (delta_cam) using the dot product formula: cos(angle) = (A . B) / (|A| |B|)
    #     # Since both vectors are unit vectors, the denominator is 1.
    #     cos_delta_cam = torch.sum(forward_vec_w * vec_to_gate_dir, dim=1)
        
    #     # Clamp to avoid NaN from arccos due to floating point inaccuracies (should be between -1 and 1)
    #     cos_delta_cam = torch.clamp(cos_delta_cam, -1.0, 1.0)
        
    #     # delta_cam is the angle in radians [0, pi]
    #     delta_cam = torch.acos(cos_delta_cam)
    #     lambda_3 = -5.0
    #     alignment_reward = torch.exp(lambda_3 * torch.pow(delta_cam, 4))


    #     # compute crashed environments if contact detected for 100 timesteps
    #     contact_forces = self.env._contact_sensor.data.net_forces_w
    #     crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
    #     mask = (self.env.episode_length_buf > 100).int()
    #     self.env._crashed = self.env._crashed + crashed * mask

    #     # time penalty
    #     reward_time = torch.ones(self.num_envs, device=self.device) # 1.0, to be scaled by negative value in config

    #     # TODO ----- END -----

    #     if self.cfg.is_train:
    #         # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
    #         rewards = {
    #             "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
    #             "crash": crashed * self.env.rew['crash_reward_scale'],
    #             "time": reward_time * self.env.rew['time_reward_scale'],
    #             "gates_passed": gates_passed_reward * self.env.rew['gates_passed_reward_scale'],
    #             "lap_complete": lap_complete_reward * self.env.rew['lap_complete_reward_scale'],
    #             # "alignment": alignment_reward * self.env.rew['alignment_reward_scale'],
    #             "delta_progress": delta_progress * self.env.rew['delta_progress_reward_scale'],
    #         }
    #         reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
    #         reward = torch.where(self.env.reset_terminated,
    #                             torch.ones_like(reward) * self.env.rew['death_cost'], reward)

    #         # Logging
    #         for key, value in rewards.items():
    #             self._episode_sums[key] += value
    #     else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
    #         reward = torch.zeros(self.num_envs, device=self.device)
    #         # TODO ----- END -----

    #     return reward
    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        # check to change waypoint
        dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        num_gates = self.env._waypoints.shape[0]
        last_gate_idx = num_gates - 1

        # 1. Gate Passing Reward
        # check if gate passesd
        gate_passed = (self.env._pose_drone_wrt_gate[:, 0] <= 0.0) & \
                      (self.env._prev_x_drone_wrt_gate > 0.0) & \
                      (dist_to_gate < 0.45) # can increase this to speed up passing

        ids_gate_passed = torch.where(gate_passed)[0] # checks which envs gate passed in
        # prev_wp_ids = self.env._idx_wp[ids_gate_passed]
        # accumualate number of gates passed
        # gates_passed_reward = self.env._n_gates_passed.float()
        gates_passed_reward = torch.zeros(self.num_envs, device=self.device)
        gates_passed_reward[ids_gate_passed] = 1.0
        # gates_passed_reward[ids_gate_passed] = self.env._n_gates_passed[ids_gate_passed].float() + 1.0

        # 2. Lap Completion Reward
        # mask within the subset
        # lap_mask_in_subset = prev_wp_ids == last_gate_idx  # shape: (len(ids_gate_passed),)
        # # env indices that completed a lap
        # lap_env_ids = ids_gate_passed[lap_mask_in_subset]  # map back to global env indices

        is_lap_complete = (self.env._idx_wp[ids_gate_passed] == 0) & (self.env._n_gates_passed[ids_gate_passed] > 0)
        self.n_laps_completed[ids_gate_passed[is_lap_complete]] += 1

        lap_complete_reward = torch.zeros(self.num_envs, device=self.device)
        lap_complete_reward[ids_gate_passed[is_lap_complete]] = self.n_laps_completed[ids_gate_passed[is_lap_complete]]

        # IMPORTANT: reset baseline distance for those envs to avoid huge negative delta on next step
        # if ids_gate_passed.numel() > 0:
        #     new_dist = torch.linalg.norm(
        #         self.env._desired_pos_w[ids_gate_passed] -
        #         self.env._robot.data.root_link_pos_w[ids_gate_passed],
        #         dim=1,
        #     )
        #     self.prev_distance_to_goal[ids_gate_passed] = new_dist

        # calculate progress via distance to goal
        distance_to_goal = torch.linalg.norm(
            self.env._desired_pos_w - self.env._robot.data.root_link_pos_w,
            dim=1)
        # tanh_distance = torch.tanh(distance_to_goal / 3.0)
        # progress = 1 - tanh_distance  # distance_to_goal is between 0 and 1 where 0 means the drone reached the goal

        # new dense progress to try
        delta_progress = self.prev_distance_to_goal - distance_to_goal
        # update previous distance to goal
        self.prev_distance_to_goal = distance_to_goal

        if len(ids_gate_passed) > 0:
            # 4. Update Variables
            # add to n_gates passed
            self.env._n_gates_passed[ids_gate_passed] += 1
            # get new waypoint target to go to
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % num_gates

            # set desired positions in the world frame
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[
                                                        self.env._idx_wp[
                                                            ids_gate_passed],
                                                        :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[
                self.env._idx_wp[ids_gate_passed], 2]
            
            # calculate the distance to the new goal only for environments that passed gates
            distance_to_new_goal = torch.linalg.norm(
                self.env._desired_pos_w[ids_gate_passed] - self.env._robot.data.root_link_pos_w[ids_gate_passed],
                dim=1)
            self.prev_distance_to_goal[ids_gate_passed] = distance_to_new_goal + 0.05

        # compute crashed environments if contact detected for 100 timesteps
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # time penalty
        reward_time = torch.ones(self.num_envs,
                                 device=self.device)  # 1.0, to be scaled by negative value in config

        # TODO ----- END -----
        # Debug prints for play mode
        if not self.cfg.is_train:
            # Print every N steps to avoid spam (e.g., every 50 steps)
            if not hasattr(self, '_debug_step_counter'):
                self._debug_step_counter = 0
            self._debug_step_counter += 1

            # if self._debug_step_counter % 50 == 0:  # Print every 50 steps
            #     print(f"[DEBUG] Step {self._debug_step_counter}:")
            #     print(f"  dist_to_gate: {dist_to_gate[0].item():.3f} m")
            #     print(f"  gate_passed: {gate_passed[0].item()}")
            #     print(
            #         f"  x_drone_wrt_gate: {self.env._pose_drone_wrt_gate[0, 0].item():.3f}")
            #     print(
            #         f"  prev_x_drone_wrt_gate: {self.env._prev_x_drone_wrt_gate[0].item():.3f}")
            #     print(f"  current_gate_idx: {self.env._idx_wp[0].item()}")
            #     print(
            #         f"  gates_passed_total: {self.env._n_gates_passed[0].item()}")

            # Always print when a gate is passed
            if gate_passed[0].item():
                print(
                    f"[GATE PASSED!] Step {self._debug_step_counter}")
                print(f"Gate {self.env._idx_wp[0].item()} is the next gate")
                print(f"  Distance: {dist_to_gate[0].item():.3f} m")
                print(
                    f"  Total gates passed: {self.env._n_gates_passed[0].item()}")

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                # "progress_goal": progress * self.env.rew[
                #     'progress_goal_reward_scale'],
                "crash": crashed * self.env.rew['crash_reward_scale'],
                "time": reward_time * self.env.rew['time_reward_scale'],
                "gates_passed": gates_passed_reward * self.env.rew[
                    'gates_passed_reward_scale'],
                "lap_complete": lap_complete_reward * self.env.rew[
                    'lap_complete_reward_scale'],
                # "alignment": alignment_reward * self.env.rew['alignment_reward_scale'],
                "delta_progress": delta_progress * self.env.rew[
                    'delta_progress_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                 torch.ones_like(reward) * self.env.rew[
                                     'death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:  # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for the racing policy. (Unchanged)"""

        drone_pose_w = self.env._robot.data.root_link_pos_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_quat_w = self.env._robot.data.root_quat_w

        ##### Some example observations you may want to explore using
        # Angular velocities (referred to as body rates)
        # drone_ang_vel_b = self.env._robot.data.root_ang_vel_b  # [roll_rate, pitch_rate, yaw_rate]

        # Current target gate information
        # current_gate_idx = self.env._idx_wp
        # current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]  # World position of current gate
        # current_gate_yaw = self.env._waypoints[current_gate_idx, -1]    # Yaw orientation of current gate

        # Relative position to current gate in gate frame
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate

        # Relative position to current gate in body frame
        # gate_pos_b, _ = subtract_frame_transforms(
        #     self.env._robot.data.root_link_pos_w,
        #     self.env._robot.data.root_quat_w,
        #     current_gate_pos_w
        # )

        # Previous actions
        # prev_actions = self.env._previous_actions  # Shape: (num_envs, 4)

        # Number of gates passed
        gates_passed = self.env._n_gates_passed.unsqueeze(1).float()

        num_gates = self.env._waypoints.shape[0]

        # --- Previous Target Gate ---
        prev_wp_idx = (self.env._idx_wp - 1) % num_gates
        prev_gate_pos_w = self.env._waypoints[prev_wp_idx, :3]
        prev_gate_quat_w = self.env._waypoints_quat[prev_wp_idx, :]
        
        drone_pos_prev_gate_frame, _ = subtract_frame_transforms(
            prev_gate_pos_w, prev_gate_quat_w, drone_pose_w
        )

        # --- Next Target Gate ---
        next_wp_idx = (self.env._idx_wp + 1) % num_gates
        next_gate_pos_w = self.env._waypoints[next_wp_idx, :3]
        next_gate_quat_w = self.env._waypoints_quat[next_wp_idx, :]
        
        drone_pos_next_gate_frame, _ = subtract_frame_transforms(
            next_gate_pos_w, next_gate_quat_w, drone_pose_w
        )

        obs = torch.cat(
            [
                drone_pose_w,       # position in the world frame (3 dims)
                drone_lin_vel_b,    # velocity in the body frame (3 dims)
                drone_quat_w,       # quaternion in the world frame (4 dims)
                gates_passed,
                drone_pos_prev_gate_frame,
                drone_pos_gate_frame,
                drone_pos_next_gate_frame,
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
            for key in self._episode_sums.keys():
                if key in self.env.rew or key in [
                    "progress_goal", "crash", "time", "gates_passed", "lap_complete", "delta_progress",
                ]:
                    episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                    extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                    self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # domain randomization - sample from defined bounds
        # Aerodynamic drag coefficients
        self.env._K_aero[env_ids, :2] = torch.empty(
            (len(env_ids), 2), device=self.device
        ).uniform_(self._k_aero_xy_min, self._k_aero_xy_max)
        self.env._K_aero[env_ids, 2] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(self._k_aero_z_min, self._k_aero_z_max)

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[env_ids, :2] = torch.empty(
            (len(env_ids), 2), device=self.device
        ).uniform_(self._kp_omega_rp_min, self._kp_omega_rp_max)
        self.env._ki_omega[env_ids, :2] = torch.empty(
            (len(env_ids), 2), device=self.device
        ).uniform_(self._ki_omega_rp_min, self._ki_omega_rp_max)
        self.env._kd_omega[env_ids, :2] = torch.empty(
            (len(env_ids), 2), device=self.device
        ).uniform_(self._kd_omega_rp_min, self._kd_omega_rp_max)

        # Yaw has different gains
        self.env._kp_omega[env_ids, 2] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(self._kp_omega_y_min, self._kp_omega_y_max)
        self.env._ki_omega[env_ids, 2] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(self._ki_omega_y_min, self._ki_omega_y_max)
        self.env._kd_omega[env_ids, 2] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(self._kd_omega_y_min, self._kd_omega_y_max)

        # Motor time constants (same for all 4 motors) - keep fixed for now
        # self.env._tau_m[env_ids] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[env_ids] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(self._twr_min, self._twr_max)

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

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        # Randomly restart at any valid gate index with equal probability
        num_gates = self.env._waypoints.shape[0]
        # waypoint_indices = torch.randint(0, num_gates, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
        n_last_gate_envs = int(n_reset * 0.2) # 20% of envs start at the last gate
        waypoint_indices[:n_last_gate_envs] = num_gates - 1

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        # adding variations in initial positons
        # x_local = -2.0 * torch.ones(n_reset, device=self.device)
        # y_local = torch.zeros(n_reset, device=self.device)
        # z_local = torch.zeros(n_reset, device=self.device)

        x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)
        z_local = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.5)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards the zeroth gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        # adding some variation in yaw as noise, so it doesn't point directly at the gate
        yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-10.0 * D2R, 10.0 * D2R)
        roll_noise = torch.empty(n_reset, device=self.device).uniform_(-5.0 * D2R, 5.0 * D2R)
        pitch_noise = torch.empty(n_reset, device=self.device).uniform_(-5.0 * D2R, 5.0 * D2R)

        # quat = quat_from_euler_xyz(
        #     torch.zeros(1, device=self.device),
        #     torch.zeros(1, device=self.device),
        #     initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        # )

        quat = quat_from_euler_xyz(roll_noise, pitch_noise, initial_yaw + yaw_noise)

        default_root_state[:, 3:7] = quat
        self.n_laps_completed[env_ids] = 0.0

        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)
            # x_local = torch.tensor([-1.75], device=self.device) # Fixed distance behind gate
            # y_local = torch.tensor([0.0], device=self.device)    # Fixed center alignment

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
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

        self.prev_distance_to_goal = torch.linalg.norm(
                self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
            )
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0