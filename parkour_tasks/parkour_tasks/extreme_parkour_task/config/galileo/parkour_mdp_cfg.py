from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab.envs.mdp.events import (
    randomize_rigid_body_mass,
    apply_external_force_torque,
    reset_joints_by_scale,
)
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg
from parkour_isaaclab.envs.mdp import terminations, rewards, parkours, events, observations, parkour_commands


@configclass
class CommandsCfg:
    """前进速度指令（针对直线跑道栏杆）"""

    base_velocity = parkour_commands.ParkourCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 4.0),
        heading_control_stiffness=1.0,
        ranges=parkour_commands.ParkourCommandCfg.Ranges(
            lin_vel_x=(0.6, 1.2),
            heading=(-0.2, 0.2),
        ),
        clips=parkour_commands.ParkourCommandCfg.Clips(
            lin_vel_clip=0.2,
            ang_vel_clip=0.3,
        ),
    )


@configclass
class ParkourEventsCfg:
    base_parkour = parkours.ParkourEventsCfg(
        asset_name="robot",
        promotion_goal_threshold=0.9,
        demotion_goal_threshold=0.35,
        promotion_distance_ratio=0.8,
        demotion_distance_ratio=0.4,
        distance_progress_cap=12.0,
    )


@configclass
class TeacherObservationsCfg:
    """教师（教师网络）观测定义：只包含状态/激光，不含摄像头。"""

    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "parkour_name": "base_parkour",
                "history_length": 10,
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class StudentObservationsCfg:
    """学生（蒸馏/学生网络）观测：额外包含深度相机与航向判定。"""
    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "parkour_name": "base_parkour",
                "history_length": 10,
            },
            clip=(-100, 100),
        )

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        depth_cam = ObsTerm(
            func=observations.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "resize": (58, 87),
                "buffer_len": 2,
                "debug_vis": False,
            },
        )

    @configclass
    class DeltaYawOkPolicyCfg(ObsGroup):
        deta_yaw_ok = ObsTerm(
            func=observations.obervation_delta_yaw_ok,
            params={"parkour_name": "base_parkour", "threshold": 0.6},
        )

    policy: PolicyCfg = PolicyCfg()
    depth_camera: DepthCameraPolicyCfg = DepthCameraPolicyCfg()
    delta_yaw_ok: DeltaYawOkPolicyCfg = DeltaYawOkPolicyCfg()


@configclass
class StudentRewardsCfg:
    """学生奖励（轻量，避免过拟合特权）"""
    reward_alive = RewTerm(
        func=rewards.reward_alive,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-6.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_calf", ".*_thigh"])},
    )
    reward_height_guidance = RewTerm(
        func=rewards.reward_height_guidance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "target_height": 0.4,
            "speed_gate": 0.12,
        },
    )
    reward_jump_clearance = RewTerm(
        func=rewards.reward_jump_clearance,
        weight=7.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "jump_window_front": 0.65,
            "jump_window_back": -0.25,
            "safety_margin": 0.08,
        },
    )
    reward_crawl_clearance = RewTerm(
        func=rewards.reward_crawl_clearance,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
        },
    )
    reward_mode_mismatch = RewTerm(
        func=rewards.reward_mode_mismatch,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "jump_margin": 0.08,
            "crawl_margin": 0.05,
        },
    )
    reward_feet_clearance = RewTerm(
        func=rewards.reward_feet_clearance,
        weight=-1.2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "check_margin_x": 0.28,
            "check_margin_y": 0.85,
            "safety_margin": 0.05,
        },
    )
    reward_foot_symmetry = RewTerm(
        func=rewards.reward_foot_symmetry,
        weight=0.6,
        params={"asset_cfg": SceneEntityCfg("robot"), "height_scale": 0.12},
    )
    reward_successful_traversal = RewTerm(
        func=rewards.reward_successful_traversal,
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "traversal_window": 0.55,
            "lateral_threshold": 0.35,
        },
    )
    reward_torques = RewTerm(
        func=rewards.reward_torques,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.06,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )


@configclass
class TeacherRewardsCfg:
    """教师奖励：包含更丰富的约束/引导。"""
    reward_alive = RewTerm(
        func=rewards.reward_alive,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-15.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_calf", ".*_thigh"])},
    )
    reward_height_guidance = RewTerm(
        func=rewards.reward_height_guidance,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "target_height": 0.4,
            "speed_gate": 0.12,
        },
    )
    reward_jump_clearance = RewTerm(
        func=rewards.reward_jump_clearance,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "jump_window_front": 0.65,
            "jump_window_back": -0.25,
            "safety_margin": 0.08,
        },
    )
    reward_crawl_clearance = RewTerm(
        func=rewards.reward_crawl_clearance,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
        },
    )
    reward_mode_mismatch = RewTerm(
        func=rewards.reward_mode_mismatch,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "jump_margin": 0.08,
            "crawl_margin": 0.05,
        },
    )
    reward_feet_clearance = RewTerm(
        func=rewards.reward_feet_clearance,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "check_margin_x": 0.28,
            "check_margin_y": 0.85,
            "safety_margin": 0.05,
        },
    )
    reward_foot_symmetry = RewTerm(
        func=rewards.reward_foot_symmetry,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "height_scale": 0.12},
    )
    reward_successful_traversal = RewTerm(
        func=rewards.reward_successful_traversal,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "traversal_window": 0.55,
            "lateral_threshold": 0.35,
        },
    )
    reward_torques = RewTerm(
        func=rewards.reward_torques,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_error = RewTerm(
        func=rewards.reward_dof_error,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_hip_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel,
        weight = 4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_delta_torques = RewTerm(
        func=rewards.reward_delta_torques,
        weight=-5.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    total_terminates = DoneTerm(
        func=terminations.terminate_episode,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventCfg:
    """通用事件配置（复位、随机化等）。"""
    reset_root_state = EventTerm(
        func=events.reset_root_state,
        params={"offset": 3.0},
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=reset_joints_by_scale,
        params={"position_range": (0.95, 1.05), "velocity_range": (0.0, 0.0)},
        mode="reset",
    )
    physics_material = EventTerm(
        func=events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": (0.6, 2.0),
            "num_buckets": 64,
        },
    )
    randomize_rigid_body_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)}
        },
    )
    random_camera_position = EventTerm(
        func=events.random_camera_position,
        mode="startup",
        params={"sensor_cfg": SceneEntityCfg("depth_camera"), "rot_noise_range": {"pitch": (-5, 5)}, "convention": "ros"},
    )
    push_by_setting_velocity = EventTerm(
        func=events.push_by_setting_velocity,
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        interval_range_s=(8.0, 8.0),
        is_global_time=True,
        mode="interval",
    )
    base_external_force_torque = EventTerm(
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


@configclass
class ActionsCfg:
    """动作配置（延迟的关节位置控制）。"""
    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=8,
        use_delay=True,
        clip={".*": (-4.8, 4.8)},
    )
