import torch

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg
import isaaclab.sim as sim_utils

from parkour_isaaclab.terrains.extreme_parkour.config.parkour import EXTREME_PARKOUR_TERRAINS_CFG
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from parkour_tasks.default_cfg import ParkourDefaultSceneCfg, VIEWER
from .parkour_mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ParkourEventsCfg,
    TeacherObservationsCfg,
    TeacherRewardsCfg,
    TerminationsCfg,
)

try:
    from galileo_parkour.assets.galileo import GALILEO_CFG
except ImportError:
    GALILEO_CFG = None  # 使用内置 USD 备份机器人，方便在缺少扩展时调试

GALILEO_USD_PATH = "/home/lz/Project/IsaacLab/source/extensions/galileo_parkour/galileo_parkour/assets/usd/robot/galileo_v2d3.usd"
# 可用固定栏杆高度（厘米）与颜色，用于训练/竞赛布局
# 栏杆高度档位（米），全局统一供课程、固定赛道等模式复用
HURDLE_HEIGHTS_CM = (5, 10, 20, 30, 40, 50)
HURDLE_HEIGHTS_M = tuple(h / 100.0 for h in HURDLE_HEIGHTS_CM)
HURDLE_BAR_LENGTH = 1.6
HURDLE_BAR_THICKNESS = 0.07
HURDLE_BAR_DEPTH = 0.06
HURDLE_BASE_THICKNESS = 0.04
HURDLE_BASE_DEPTH = 0.18
HURDLE_POST_RADIUS = 0.05
HURDLE_POST_Y_OFFSET = HURDLE_BAR_LENGTH * 0.5 - HURDLE_POST_RADIUS


def _hurdle_asset_name(height_cm: int, component: str) -> str:
    return f"hurdle_{height_cm}cm_{component}"  # 组件命名统一，便于场景索引


def _hurdle_color(height_cm: int) -> tuple[float, float, float]:
    palette = {
        20: (0.20, 0.52, 0.85),
        30: (0.27, 0.68, 0.52),
        40: (0.86, 0.62, 0.26),
        50: (0.83, 0.32, 0.32),
        10: (0.55, 0.45, 0.80),
        5: (0.35, 0.70, 0.88),
    }
    return palette.get(height_cm, (0.6, 0.6, 0.6))


def _galileo_robot_cfg():
    if GALILEO_CFG is not None:
        return GALILEO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GALILEO_USD_PATH,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),
            joint_pos={
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            },
        ),
        actuators={},
    )


def place_galileo_hurdles(
    env,
    env_ids,
    spacing: float = 2.0,
    start: float = 2.0,
    layout: str = "auto",
    jump_to_mix_level: int = 6,
    mix_refresh_prob: float = 0.1,
):
    """Mixed curriculum: col 0-1 侧为“跳跃”，col 2-3 侧为“钻爬”。

    - 难度依据 terrain_level 归一化（level/8.0），逐步提高栏杆高度。
    - 跳跃列：0.05m -> 0.35m 递增；钻爬列：0.60m -> 0.35m 递减。
    - num_visible = level//2 + 1，逐步增加栏杆数量（1~4）。
    - 栏杆使用静态长方体+圆柱组合，不再依赖 USD 资产。
    """
    terrain = env.scene.terrain
    terrain_levels = terrain.terrain_levels[env_ids]
    terrain_types = terrain.terrain_types[env_ids]  # 列索引
    env_origins = terrain.env_origins[env_ids]
    curriculum_on = getattr(terrain.cfg.terrain_generator, "curriculum", True)
    # 自动课程：先 jump-only，达标后切混合，保留少量 jump-only 防遗忘
    layout_mode = layout
    if layout == "auto":
        mean_level = terrain.terrain_levels.float().mean()
        if mean_level >= jump_to_mix_level:
            # 大概率使用竞赛布局，小概率回到 jump-only 保持记忆
            layout_mode = "competition"
            if torch.rand(1, device=env.device) < mix_refresh_prob:
                layout_mode = "jump_train"
        else:
            layout_mode = "jump_train"

    # 统一的栏杆高度增量（米），供各布局模式复用
    increments = torch.tensor([0.00, 0.05, 0.10, 0.15], device=env.device)

    # 固定比赛布局：直接使用 20/30/40/50cm 四根栏杆（有序递增）
    if layout_mode == "competition":
        num_visible = torch.full_like(terrain_levels, 4)
        difficulty = torch.ones_like(terrain_levels, dtype=torch.float)
        target_h = torch.tensor([0.20, 0.30, 0.40, 0.50], device=env.device).repeat(len(env_ids), 1)
    elif layout_mode == "fixed":
        num_visible = torch.full_like(terrain_levels, 4)
        difficulty = torch.ones_like(terrain_levels, dtype=torch.float)
        fixed_sequences = torch.tensor(
            [
                [0.05, 0.10, 0.20, 0.30],
                [0.10, 0.20, 0.30, 0.40],
                [0.20, 0.30, 0.40, 0.50],
            ],
            device=env.device,
        )
        seq_idx = torch.remainder(terrain_levels, fixed_sequences.shape[0])
        target_h = fixed_sequences[seq_idx]
    elif layout_mode == "jump_train":
        # 只在 jump 列练习跳跃，crawl 列隐藏
        num_visible = torch.clamp(terrain_levels // 2 + 1, min=1, max=4)
        difficulty = torch.clamp(terrain_levels.float() / 8.0, 0.0, 1.0)
        target_h = torch.clamp(
            (0.05 + 0.30 * difficulty).unsqueeze(1) + increments, 0.05, 0.50
        )
        # 隐藏 crawl 列
        num_visible = torch.where(terrain_types < 2, num_visible, torch.zeros_like(num_visible))
    elif curriculum_on:
        num_visible = torch.clamp(terrain_levels // 2 + 1, min=1, max=4)
        difficulty = torch.clamp(terrain_levels.float() / 8.0, 0.0, 1.0)
        target_h = None  # lazy compute
    else:
        # Play/评估：固定四根栏杆，全难度
        num_visible = torch.full_like(terrain_levels, 4)
        difficulty = torch.ones_like(terrain_levels, dtype=torch.float)
        target_h = None

    base_heights = torch.zeros_like(difficulty, dtype=torch.float)
    # col 0,1 -> jump; col 2,3 -> crawl
    jump_mask = terrain_types < 2
    crawl_mask = ~jump_mask
    base_heights[jump_mask] = 0.05 + 0.30 * difficulty[jump_mask]  # 5cm -> 35cm
    base_heights[crawl_mask] = 0.60 - 0.25 * difficulty[crawl_mask]  # 60cm -> 35cm

    target_x = env_origins[:, 0].unsqueeze(1) + start + spacing * torch.arange(4, device=env.device)
    target_y = env_origins[:, 1].unsqueeze(1)
    # 目标高度矩阵 shape (num_envs, 4)（仅用于选择最接近的资产）
    if target_h is None:
        target_h = torch.clamp(base_heights.unsqueeze(1) + increments, 0.05, 0.60)

    avail_heights = torch.tensor(HURDLE_HEIGHTS_M, device=env.device)
    # 缓存每个 env 的实际栏杆高度，供特权观测/奖励使用
    if not hasattr(env.scene, "hurdle_heights"):
        env.scene.hurdle_heights = torch.full((env.num_envs, 4), -1.0, device=env.device)
    env.scene.hurdle_heights[env_ids] = -1.0  # 默认无栏杆

    # 先隐藏全部组件
    for height_cm in HURDLE_HEIGHTS_CM:
        for component in ("base", "bar", "post_left", "post_right"):
            asset_name = _hurdle_asset_name(height_cm, component)
            try:
                asset = env.scene[asset_name]
            except KeyError:
                continue
            hide_state = asset.data.default_root_state[env_ids].clone()
            hide_state[:, 0:3] = -1000.0
            asset.write_root_pose_to_sim(hide_state[:, :7], env_ids)
            asset.write_root_velocity_to_sim(torch.zeros_like(hide_state[:, 7:]), env_ids)

    # 按栏杆槽位逐一放置
    for idx in range(4):
        active_mask = num_visible > idx
        if not active_mask.any():
            continue
        desired_h = target_h[:, idx]
        # 选择最近高度的资产
        diff = torch.abs(desired_h.unsqueeze(1) - avail_heights.unsqueeze(0))
        chosen_idx = torch.argmin(diff, dim=1)

        for asset_choice, height_cm in enumerate(HURDLE_HEIGHTS_CM):
            asset_h = avail_heights[asset_choice]
            env_sel = active_mask & (chosen_idx == asset_choice)
            active_ids = env_ids[env_sel]
            if len(active_ids) == 0:
                continue

            env.scene.hurdle_heights[active_ids, idx] = asset_h
            _place_static_hurdle(
                env=env,
                height_cm=height_cm,
                bar_height=asset_h,
                x_positions=target_x[env_sel, idx],
                y_positions=target_y[env_sel, 0],
                ground_heights=env_origins[env_sel, 2],
                active_ids=active_ids,
            )


def _place_static_hurdle(env, height_cm, bar_height, x_positions, y_positions, ground_heights, active_ids):
    """Place the static primitive hurdle (base + 2 posts + bar) for the selected env ids."""
    base_z = ground_heights + HURDLE_BASE_THICKNESS * 0.5
    bar_z = ground_heights + HURDLE_BASE_THICKNESS + bar_height + HURDLE_BAR_THICKNESS * 0.5
    post_height = HURDLE_BASE_THICKNESS + bar_height + HURDLE_BAR_THICKNESS
    post_z = ground_heights + post_height * 0.5
    zeros_vel = torch.zeros((active_ids.numel(), 6), device=env.device)

    def _set_pose(asset_name: str, pos_x: torch.Tensor, pos_y: torch.Tensor, pos_z: torch.Tensor):
        asset = env.scene[asset_name]
        root_state = asset.data.default_root_state[active_ids].clone()
        root_state[:, 0] = pos_x
        root_state[:, 1] = pos_y
        root_state[:, 2] = pos_z
        asset.write_root_pose_to_sim(root_state[:, :7], active_ids)
        asset.write_root_velocity_to_sim(zeros_vel, active_ids)

    _set_pose(_hurdle_asset_name(height_cm, "base"), x_positions, y_positions, base_z)
    _set_pose(_hurdle_asset_name(height_cm, "bar"), x_positions, y_positions, bar_z)
    _set_pose(_hurdle_asset_name(height_cm, "post_left"), x_positions, y_positions - HURDLE_POST_Y_OFFSET, post_z)
    _set_pose(_hurdle_asset_name(height_cm, "post_right"), x_positions, y_positions + HURDLE_POST_Y_OFFSET, post_z)


@configclass
class GalileoParkourSceneCfg(ParkourDefaultSceneCfg):
    robot: ArticulationCfg = _galileo_robot_cfg()
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=2,
        track_air_time=True,
        debug_vis=False,
        force_threshold=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        # Ensure robot config uses Galileo (override go2 defaults from base scene).
        self.robot = _galileo_robot_cfg()
        # Flat terrain; curriculum drives hurdle count和列分配（跳跃/钻爬）。
        self.terrain.terrain_generator = EXTREME_PARKOUR_TERRAINS_CFG
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.terrain_generator.num_cols = 4  # 0-1 列 jump，2-3 列 crawl
        self.terrain.terrain_generator.horizontal_scale = 0.1
        for name, sub in self.terrain.terrain_generator.sub_terrains.items():
            sub.use_simplified = True
            sub.proportion = 1.0 if name == "parkour_flat" else 0.0
            sub.apply_roughness = False

        # Hurdle assets (static primitives: base + posts + bar).
        base_size = (HURDLE_BASE_DEPTH, HURDLE_BAR_LENGTH + 0.2, HURDLE_BASE_THICKNESS)
        bar_size = (HURDLE_BAR_DEPTH, HURDLE_BAR_LENGTH, HURDLE_BAR_THICKNESS)
        for h_cm, bar_height in zip(HURDLE_HEIGHTS_CM, HURDLE_HEIGHTS_M):
            color = _hurdle_color(h_cm)
            base_spawn = sim_utils.CuboidCfg(
                size=base_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            bar_spawn = sim_utils.CuboidCfg(
                size=bar_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            post_spawn = sim_utils.CylinderCfg(
                radius=HURDLE_POST_RADIUS,
                height=HURDLE_BASE_THICKNESS + bar_height + HURDLE_BAR_THICKNESS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            component_cfgs = {
                "base": RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{_hurdle_asset_name(h_cm, 'base')}",
                    spawn=base_spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(-1000.0, -1000.0, -1000.0),
                        lin_vel=(0.0, 0.0, 0.0),
                        ang_vel=(0.0, 0.0, 0.0),
                    ),
                ),
                "bar": RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{_hurdle_asset_name(h_cm, 'bar')}",
                    spawn=bar_spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(-1000.0, -1000.0, -1000.0),
                        lin_vel=(0.0, 0.0, 0.0),
                        ang_vel=(0.0, 0.0, 0.0),
                    ),
                ),
                "post_left": RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{_hurdle_asset_name(h_cm, 'post_left')}",
                    spawn=post_spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(-1000.0, -1000.0, -1000.0),
                        lin_vel=(0.0, 0.0, 0.0),
                        ang_vel=(0.0, 0.0, 0.0),
                    ),
                ),
                "post_right": RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{_hurdle_asset_name(h_cm, 'post_right')}",
                    spawn=post_spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(-1000.0, -1000.0, -1000.0),
                        lin_vel=(0.0, 0.0, 0.0),
                        ang_vel=(0.0, 0.0, 0.0),
                    ),
                ),
            }
            for comp, cfg in component_cfgs.items():
                setattr(self, _hurdle_asset_name(h_cm, comp), cfg)


@configclass
class GalileoTeacherParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: GalileoParkourSceneCfg = GalileoParkourSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18

        # keep curriculum to grow hurdle count
        self.scene.terrain.terrain_generator.curriculum = True
        self.events.random_camera_position = None
        self.events.push_by_setting_velocity.interval_range_s = (6.0, 6.0)
        # place hurdles on reset
        self.events.place_hurdles = EventTermCfg(
            func=place_galileo_hurdles,
            mode="reset",
            params={
                "spacing": 2.0,
                "start": 2.0,
                "layout": "auto",
                "jump_to_mix_level": 6,
                "mix_refresh_prob": 0.1,
            },
        )
        # sensor update periods
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        # ensure mass/com events target base_link
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = "base_link"
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"


@configclass
class GalileoTeacherParkourEnvCfg_PLAY(GalileoTeacherParkourEnvCfg):
    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 60.0
        # Play 时默认展示固定比赛布局（20/30/40/50）并开启可视化箭头/航向点
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        self.events.place_hurdles.params["start"] = 2.0  # type: ignore[attr-defined]
        self.events.place_hurdles.params["spacing"] = 2.0  # type: ignore[attr-defined]
        self.events.place_hurdles.params["layout"] = "competition"  # type: ignore[attr-defined] fixed/competition
        self.commands.base_velocity.debug_vis = True
        self.parkours.base_parkour.debug_vis = True
