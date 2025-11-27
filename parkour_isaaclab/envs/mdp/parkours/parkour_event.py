
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np 
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import wrap_to_pi
from parkour_isaaclab.managers import ParkourTerm
from parkour_isaaclab.terrains import ParkourTerrainGeneratorCfg, ParkourTerrainImporter, ParkourTerrainGenerator

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from .parkour_events_cfg import ParkourEventsCfg


class ParkourEvent(ParkourTerm):
    cfg: ParkourEventsCfg

    def __init__(
        self, 
        cfg: ParkourEventsCfg, 
        env: ParkourManagerBasedRLEnv
        ):
        super().__init__(cfg, env)

        self.episode_length_s = env.cfg.episode_length_s
        self.reach_goal_delay = cfg.reach_goal_delay
        self.num_future_goal_obs = cfg.num_future_goal_obs
        self.next_goal_threshold = cfg.next_goal_threshold
        self.simulation_time = env.step_dt
        self.arrow_num = cfg.arrow_num
        self.env = env 
        self.debug_vis = cfg.debug_vis
        self.promotion_goal_threshold = cfg.promotion_goal_threshold
        self.demotion_goal_threshold = cfg.demotion_goal_threshold
        self.promotion_distance_ratio = cfg.promotion_distance_ratio
        self.demotion_distance_ratio = cfg.demotion_distance_ratio
        self.distance_progress_cap = cfg.distance_progress_cap
               
        self.robot: Articulation = env.scene[cfg.asset_name]
        # -- metrics
        self.metrics["far_from_current_goal"] = torch.zeros(self.num_envs, device='cpu')
        self.metrics["how_far_from_start_point"] = torch.zeros(self.num_envs, device='cpu')
        self.metrics["terrain_levels"] = torch.zeros(self.num_envs, device='cpu')
        self.metrics["current_goal_idx"] = torch.zeros(self.num_envs, device='cpu')
        self.dis_to_start_pos = torch.zeros(self.num_envs, device=self.device)
        self.terrain: ParkourTerrainImporter = self.env.scene.terrain
        terrain_generator: ParkourTerrainGenerator = self.terrain.terrain_generator_class
        parkour_terrain_cfg :ParkourTerrainGeneratorCfg = self.terrain.cfg.terrain_generator
        # stage-based curriculum state (P0~P4)
        self.stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stage_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stage_attempts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stage_cooldown = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stage_eval_window = 8
        self.stage_min_stay = 6
        self.max_stage = 100  # allow unbounded progression
        self.recall_prob = 0.15
        # linear mapping stage->terrain row (spacing 2), will be clamped by available rows
        self.stage_to_level = torch.arange(0, 200, 2, device=self.device, dtype=torch.long)
        self.reached_goal_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # expose stage for other components (e.g., hurdle placement)
        self.env.curriculum_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.num_goals = parkour_terrain_cfg.num_goals
        self.env_class = torch.zeros(self.num_envs, device=self.device)
        self.env_origins = self.terrain.env_origins
        self.terrain_type = terrain_generator.terrain_type
        self.terrain_class = torch.from_numpy(self.terrain_type).to(self.device).to(torch.float)
        self.env_class[:] = self.terrain_class[self.terrain.terrain_levels, self.terrain.terrain_types]
        
        terrain_goals = terrain_generator.goals
        self.terrain_goals = torch.from_numpy(terrain_goals).to(self.device).to(torch.float)
        self.env_goals = torch.zeros(self.num_envs,  self.terrain_goals.shape[2] + self.num_future_goal_obs, 3, device=self.device, requires_grad=False)
        self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        temp = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float).to(device = self.device)
        
        if self.debug_vis:
            self.total_heights = torch.from_numpy(terrain_generator.goal_heights).to(device = self.device)
            self.future_goal_idx = torch.ones(self.num_goals, device=self.device, dtype=torch.bool).repeat(self.num_envs, 1)
            self.future_goal_idx[:, 0] = False
            self.env_per_heights = self.total_heights[self.terrain.terrain_levels, self.terrain.terrain_types]
       
        self.total_terrain_names = terrain_generator.terrain_names
        numpy_terrain_levels = self.terrain.terrain_levels.detach().cpu().numpy() ## string type can't convert to torch
        numpy_terrain_types = self.terrain.terrain_types.detach().cpu().numpy()
        self.env_per_terrain_name = self.total_terrain_names[numpy_terrain_levels, numpy_terrain_types]
        self._reset_offset = self.env.event_manager.get_term_cfg('reset_root_state').params['offset']

        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos_w
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos_w
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])


    def __call__(self):
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    def _gather_cur_goals(self, future=0):
        idx = torch.clamp(self.cur_goal_idx + future, max=self.env_goals.shape[1] - 1)
        return self.env_goals.gather(1, idx[:, None, None].expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def __str__(self) -> str:
        msg = "ParkourCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg
    
    def _update_command(self):
        """Re-target the current goal position to the current root state."""
        next_flag = self.reach_goal_timer > self.reach_goal_delay / self.simulation_time
        # 避免索引越界，仅对未完成的索引做自增
        next_flag = next_flag & (self.cur_goal_idx < self.num_goals)
        if self.debug_vis:
            tmp_mask = torch.nonzero(self.cur_goal_idx>0).squeeze(-1)
            if tmp_mask.numel() > 0:
                self.future_goal_idx[tmp_mask, self.cur_goal_idx[tmp_mask]] = False
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0
        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.reached_goal_ids = torch.norm(robot_root_pos_w - self.cur_goals[:, :2], dim=1) < self.next_goal_threshold
        reached_goal_idx = self.reached_goal_ids.nonzero(as_tuple=False).squeeze(-1)
        if reached_goal_idx.numel() > 0:
            self.reach_goal_timer[reached_goal_idx] += 1
        # force completion when the last goal is reached once so higher-level logic can terminate
        last_goal_mask = (self.cur_goal_idx == (self.num_goals - 1)) & self.reached_goal_ids
        if last_goal_mask.any():
            self.cur_goal_idx[last_goal_mask] = self.num_goals

        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos_w
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos_w
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        
        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        start_pos = self.env_origins[:,:2] - \
                    torch.tensor((self.terrain.cfg.terrain_generator.size[1] + \
                                  self._reset_offset, 0)).to(self.device)

        self.dis_to_start_pos = torch.norm(start_pos - self.robot.data.root_pos_w[:, :2], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        # 如果关闭课程（如 Play），跳过阶段机，保持原 terrain level/origin
        curriculum_on = getattr(self.terrain.cfg.terrain_generator, "curriculum", True)
        if not curriculum_on:
            return

        # record outcome of last episode for these envs
        success_mask, has_outcome = self._get_last_outcome(env_ids)
        self._update_stage(env_ids, success_mask, has_outcome)

        # curriculum recall: small prob to revisit previous stage for retention
        stage_now = self.stage.clone()
        stage_used = stage_now[env_ids]
        recall_mask = (stage_used > 0) & (torch.rand_like(stage_used.float()) < self.recall_prob)
        stage_used = torch.where(recall_mask, stage_used - 1, stage_used)
        self.env.curriculum_stage[env_ids] = stage_used

        # map stage to terrain level row; clamp by available rows
        max_rows = self.terrain.terrain_origins.shape[0]
        level_values = self.stage_to_level[torch.clamp(stage_used, max=self.stage_to_level.shape[0] - 1)]
        level_values = torch.clamp(level_values, max=max_rows - 1)
        self.terrain.terrain_levels[env_ids] = level_values
        self.env_origins[env_ids] = self.terrain.terrain_origins[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]]

        temp = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.num_future_goal_obs, 1)), dim=1)[:]
        # 将航向点置于跑道中心（y=0），避免偏移
        self.env_goals[:, :, 1] = 0.0
        # 将航点从栏杆处偏移开，避免目标落在栏杆上
        try:
            place_cfg = self.env.event_manager.get_term_cfg("place_hurdles")
            spacing = place_cfg.params.get("spacing", 2.0)
            start = place_cfg.params.get("start", 2.0)
        except Exception:
            spacing, start = 2.0, 2.0
        hurdle_slots = start + spacing * torch.arange(4, device=self.device).view(1, -1)
        goal_x = self.env_goals[:, :, 0]
        env_hurdle_x = self.env_origins[:, 0:1] + hurdle_slots  # (N,4)
        # 找到与任意栏杆距离过近的航点并向前推开 margin
        margin = 0.5
        dist = goal_x.unsqueeze(-1) - env_hurdle_x.unsqueeze(1)  # (N,G,4)
        nearest = torch.abs(dist).min(dim=-1).values  # (N,G)
        near_mask = nearest < margin
        # 对过近的目标，推到最近栏杆前方 margin
        # 取最近栏杆坐标
        nearest_idx = torch.abs(dist).argmin(dim=-1)
        nearest_hurdle_x = env_hurdle_x.gather(1, nearest_idx)
        adjusted_x = torch.where(goal_x < nearest_hurdle_x, nearest_hurdle_x + margin, goal_x)
        self.env_goals[:, :, 0] = torch.where(near_mask, adjusted_x, goal_x)
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos_w
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos_w
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        
        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        numpy_terrain_levels = self.terrain.terrain_levels.detach().cpu().numpy()
        numpy_terrain_types = self.terrain.terrain_types.detach().cpu().numpy()
        self.env_per_terrain_name = self.total_terrain_names[numpy_terrain_levels, numpy_terrain_types]

        self.reach_goal_timer[env_ids] = 0
        self.cur_goal_idx[env_ids] = 0

        if self.debug_vis:
            self.future_goal_idx[env_ids, 0] = False
            self.future_goal_idx[env_ids, 1:] = True
            self.env_per_heights = self.total_heights[self.terrain.terrain_levels, self.terrain.terrain_types]

    def _update_metrics(self):
        # logs data
        levels = self.terrain.terrain_levels.float()
        self.metrics["terrain_levels"] = levels.to(device='cpu')
        self.metrics["terrain_level_mean"] = torch.mean(levels).to(device='cpu')
        self.metrics["terrain_level_max"] = torch.max(levels).to(device='cpu')
        self.metrics["curriculum_stage"] = self.stage.to(device="cpu", dtype=torch.float)
        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.metrics["far_from_current_goal"] = (torch.norm(self.cur_goals[:, :2] - robot_root_pos_w,dim =-1) - self.next_goal_threshold).to(device = 'cpu')
        self.metrics["current_goal_idx"] = self.cur_goal_idx.to(device='cpu', dtype=float)
        self.metrics["how_far_from_start_point"] = self.dis_to_start_pos.to(device = 'cpu')

    def _get_last_outcome(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine episode success per env with a conservative definition."""
        if not hasattr(self.env, "reset_time_outs") or self.env.reset_time_outs.numel() == 0:
            return torch.zeros(len(env_ids), device=self.device, dtype=torch.bool), torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
        try:
            timeout_mask = self.env.reset_time_outs[env_ids].to(torch.bool)
        except Exception:
            return torch.zeros(len(env_ids), device=self.device, dtype=torch.bool), torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
        reached_goal = self.reached_goal_ids[env_ids] if hasattr(self, "reached_goal_ids") else torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
        # 额外距离判定：跑过半程也记为成功
        start_pos = self.env_origins[env_ids, :2] - torch.tensor((self.terrain.cfg.terrain_generator.size[1] + self._reset_offset, 0)).to(self.device)
        dist = torch.norm(start_pos - self.robot.data.root_pos_w[env_ids, :2], dim=1)
        dist_thresh = 0.5 * (self.terrain.cfg.terrain_generator.size[1])
        dist_success = dist > dist_thresh
        success = reached_goal | dist_success
        # 超时不再视为成功，避免“站桩升级”
        return success, torch.ones(len(env_ids), device=self.device, dtype=torch.bool)

    def _update_stage(self, env_ids: Sequence[int], success_mask: torch.Tensor, valid_mask: torch.Tensor):
        """Stage state machine: accumulate windowed success rate; promote/demote after cooldown."""
        if len(env_ids) == 0:
            return
        # 对无效的样本跳过计数
        if not torch.any(valid_mask):
            return
        # update counters
        self.stage_attempts[env_ids] += valid_mask.to(torch.long)
        self.stage_success[env_ids] += (success_mask & valid_mask).to(torch.long)
        self.stage_cooldown[env_ids] += 1

        # evaluate when enough attempts and cooldown satisfied
        eval_mask = (self.stage_attempts >= self.stage_eval_window) & (self.stage_cooldown >= self.stage_min_stay)
        success_rate = torch.where(self.stage_attempts > 0, self.stage_success.float() / self.stage_attempts.float(), torch.zeros_like(self.stage_success, dtype=torch.float))

        promote = eval_mask & (success_rate >= 0.7) & (self.stage < self.max_stage)
        demote = eval_mask & (success_rate <= 0.3) & (self.stage > 0)

        self.stage = torch.clamp(self.stage + promote.to(torch.long) - demote.to(torch.long), 0, self.max_stage)

        # reset counters for envs that changed or were evaluated
        reset_mask = promote | demote
        self.stage_success[reset_mask] = 0
        self.stage_attempts[reset_mask] = 0
        self.stage_cooldown[reset_mask] = 0

        # avoid counter explosion; keep attempts bounded
        self.stage_attempts = torch.clamp(self.stage_attempts, max=self.stage_eval_window * 2)
        self.stage_success = torch.clamp(self.stage_success, max=self.stage_eval_window * 2)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "current_goal_pose_visualizer"):
                self.current_goal_pose_visualizer = VisualizationMarkers(self.cfg.current_goal_pose_visualizer_cfg)
            # set their visibility to true
            self.current_goal_pose_visualizer.set_visibility(True)
            if not hasattr(self, "future_goal_poses_visualizer"):
                self.future_goal_poses_visualizer = VisualizationMarkers(self.cfg.future_goal_poses_visualizer_cfg)
            self.future_goal_poses_visualizer.set_visibility(True)


            if not hasattr(self, "current_arrow_visualizer"):
                self.current_arrow_visualizer = VisualizationMarkers(self.cfg.current_arrow_visualizer_cfg)
            # set their visibility to true
            self.current_arrow_visualizer.set_visibility(True)
            if not hasattr(self, "future_arrow_visualizer"):
                self.future_arrow_visualizer = VisualizationMarkers(self.cfg.future_arrow_visualizer_cfg)
            self.future_arrow_visualizer.set_visibility(True)

        else:
            if hasattr(self, "current_goal_pose_visualizer"):
                self.current_goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "future_goal_poses_visualizer"):
                self.future_goal_poses_visualizer.set_visibility(False)

            if hasattr(self, "current_arrow_visualizer"):
                self.current_arrow_visualizer.set_visibility(False)
            if hasattr(self, "future_arrow_visualizer"):
                self.future_arrow_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        env_per_goals = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types] 
        env_per_xy_goals = env_per_goals[:,:,:2].reshape(self.num_envs, -1,2) ## (env_num, 8, 2 )
        env_per_xy_goals = env_per_xy_goals + self.env_origins[:, :2].unsqueeze(1)
        goal_height = self.env_per_heights.unsqueeze(-1)*self.terrain.cfg.terrain_generator.vertical_scale
        env_per_goal_pos = torch.concat([env_per_xy_goals, goal_height],dim=-1)
        env_per_current_goal_pos = env_per_goal_pos[~self.future_goal_idx, :]
        env_per_future_goal_pos = env_per_goal_pos[self.future_goal_idx, :] .reshape(-1,3)
        self.current_goal_pose_visualizer.visualize(
            translations=env_per_current_goal_pos,
        )
        if len(env_per_future_goal_pos) > 0:
            self.future_goal_poses_visualizer.visualize(
                translations=env_per_future_goal_pos ,
            )
        current_arrow_list = []
        future_arrow_list = []
        for i in range(self.arrow_num):
            norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            current_pose_arrow = self.robot.data.root_pos_w[:, :2] + 0.1*(i+3) * target_vec_norm[:, :2]
            current_arrow_list.append(torch.concat([
                current_pose_arrow[:,0][:,None], 
                current_pose_arrow[:,1][:,None], 
                self.robot.data.root_pos_w[:, 2][:,None]
                ], dim = 1))
            if len(env_per_future_goal_pos) > 0:
                    
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                future_pose_arrow = self.robot.data.root_pos_w[:, :2] + 0.2*(i+3) * target_vec_norm[:, :2]
                future_arrow_list.append(torch.concat([
                    future_pose_arrow[:,0][:,None], 
                    future_pose_arrow[:,1][:,None], 
                    self.robot.data.root_pos_w[:, 2][:,None]
                    ], dim = 1))
            else:
                future_arrow_list.append(torch.concat([
                    current_pose_arrow[:,0][:,None], 
                    current_pose_arrow[:,1][:,None], 
                    self.robot.data.root_pos_w[:, 2][:,None]
                    ], dim = 1))

        current_arrow_positions = torch.cat(current_arrow_list, dim=0)
        future_arrow_positions = torch.cat(future_arrow_list, dim=0)
        self.current_arrow_visualizer.visualize(
            translations=current_arrow_positions,
        )

        self.future_arrow_visualizer.visualize(
            translations=future_arrow_positions,
        )

    @property
    def command(self):
        """Null command.

        Raises:
            RuntimeError: No command is generated. Always raises this error.
        """
        raise RuntimeError("NullCommandTerm does not generate any commands.")
