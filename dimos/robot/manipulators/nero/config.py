# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgileX NERO hardware and planning model configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dimos.control.components import HardwareComponent, HardwareType
from dimos.manipulation.planning.groups.models import PlanningGroupDefinition
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.robot.manipulators._modeling import base_pose, coordinator_joint_mapping
from dimos.utils.data import LfsPath

NERO_DIR = Path(__file__).parent
NERO_ASSETS_DIR = NERO_DIR / "assets"
NERO_LOCAL_AGX_ARM_DESCRIPTION = NERO_ASSETS_DIR / "agx_arm_description"
NERO_LOCAL_WBCD_URDF = NERO_ASSETS_DIR / "wbcd_urdf"
NERO_LOCAL_REALSENSE2_DESCRIPTION = NERO_ASSETS_DIR / "realsense2_description"

NERO_DOF = 7
NERO_LEFT_CAN = "can0"
NERO_RIGHT_CAN = "can1"
NERO_FIRMWARE_VERSION = "v120"
NERO_AGX_GRIPPER = "agx_gripper"

# Plain single-arm URDF (no gripper) for the control-side Pinocchio IK used by
# the continuous ``cartesian_ik`` control task. Unlike the planning model, this
# is loaded with a raw ``buildModelFromUrdf`` (no xacro / ``package://``
# expansion; meshes are not loaded for the kinematic model), so it must be a
# flat URDF with exactly the seven arm joints (``nq == NERO_DOF``). Verified:
# joints joint1..joint7, ee_joint_id 7 == joint7 (link7 tip), matching the
# planning tip_link.
NERO_FK_MODEL = (
    NERO_ASSETS_DIR / "agx_arm_description/agx_arm_urdf/nero/urdf/nero_description.urdf"
)
NERO_EE_JOINT_ID = NERO_DOF  # 7 (Pinocchio joint id of the end-effector joint)

# Single-arm URDF/Xacro used by Pinocchio IK and planning per-arm.
# Each arm gets its own RobotModelConfig with this URDF, placed at
# different base_pose offsets (same pattern as dual_xarm6_planner).
# The selected model includes the AGX gripper for visualization/collision,
# while planning still controls only the seven arm joints.
AGX_ARM_DESCRIPTION_PKG = LfsPath("agx_arm_description")
AGX_ARM_URDF_PKG = LfsPath("agx_arm_urdf")
NERO_MODEL_RELATIVE_PATH = Path("agx_arm_urdf/nero/urdf/nero_with_gripper_description.xacro")
NERO_MODEL_PATH = AGX_ARM_DESCRIPTION_PKG / NERO_MODEL_RELATIVE_PATH
NERO_MODEL_PATH_FALLBACKS = (
    NERO_LOCAL_AGX_ARM_DESCRIPTION / NERO_MODEL_RELATIVE_PATH,
    Path.home()
    / "wbcd_extracted/agx_arm_sim/agx_arm_description"
    / NERO_MODEL_RELATIVE_PATH,
    Path.home() / "agx_arm_urdf/nero/urdf/nero_with_gripper_description.xacro",
)
NERO_MODEL_LFS_FALLBACKS = (
    NERO_MODEL_PATH,
    AGX_ARM_URDF_PKG / "nero/urdf/nero_with_gripper_description.xacro",
)
# Plain single-arm URDF (no gripper, no xacro macros) used by the teleop
# TeleopIKTask, whose Pinocchio solver calls buildModelFromUrdf directly and
# therefore cannot expand the .xacro above. Same 7-DOF chain (joint1..joint7,
# tip link7 -> ee_joint_id=7); the gripper is driven separately via
# gripper_joint, so the arm-only URDF is exactly what IK needs.
NERO_FK_MODEL_RELATIVE_PATH = Path("agx_arm_urdf/nero/urdf/nero_description.urdf")
NERO_FK_MODEL = NERO_LOCAL_AGX_ARM_DESCRIPTION / NERO_FK_MODEL_RELATIVE_PATH
NERO_EE_JOINT_ID = NERO_DOF  # tip link7 is the 7th joint in the chain

NERO_BODY_MODEL_RELATIVE_PATH = Path("urdf/wbcd_urdf.urdf")
NERO_BODY_MODEL_PATH = NERO_LOCAL_WBCD_URDF / NERO_BODY_MODEL_RELATIVE_PATH
NERO_PACKAGE_PATHS: dict[str, Path] = {
    "agx_arm_description": NERO_LOCAL_AGX_ARM_DESCRIPTION,
    "wbcd_urdf": NERO_LOCAL_WBCD_URDF,
    "realsense2_description": NERO_LOCAL_REALSENSE2_DESCRIPTION,
}
NERO_HOME_JOINTS = [0.0] * NERO_DOF
NERO_GRIPPER_COLLISION_EXCLUSIONS: list[tuple[str, str]] = [
    ("link7", "gripper_flange"),
    ("gripper_flange", "gripper_base"),
    ("gripper_base", "gripper_link1"),
    ("gripper_base", "gripper_link2"),
    ("gripper_link1", "gripper_link2"),
    # Wrist/flange collision meshes overlap in every configuration (including the
    # all-zeros home pose), so Drake flags them as permanent self-collisions even
    # though they never physically interfere. These pairs are non-adjacent (not
    # auto-filtered) and must be excluded explicitly. Penetration depths measured
    # at home: link6<->gripper_flange ~61mm, link5<->gripper_flange ~28mm,
    # link5<->link7 ~3mm (see scripts/demo_collision_inspect.py).
    ("link6", "gripper_flange"),
    ("link5", "gripper_flange"),
    ("link5", "link7"),
]

# Physical placement of each arm on the massage robot chassis, copied from
# wbcd_urdf/urdf/dual_nero.xacro left_arm_joint/right_arm_joint.
NERO_BASE_X = -0.002
NERO_LEFT_BASE_Y = 0.1
NERO_RIGHT_BASE_Y = -0.1
NERO_BASE_Z = 0.59
NERO_LEFT_BASE_RPY = (-1.57, -1.57, 0.0)
NERO_RIGHT_BASE_RPY = (1.57, -1.57, 0.0)
NERO_TABLE_OBSTACLE_NAME = "nero_table"
NERO_TABLE_CENTER = (0.51, -0.055, -0.01)
NERO_TABLE_SIZE = (1.2, 1.17, 0.08)


def nero_joints(hardware_id: str) -> list[str]:
    return [f"{hardware_id}/joint{i}" for i in range(1, NERO_DOF + 1)]


def nero_hardware(
    hardware_id: str,
    *,
    adapter_type: str = "mock",
    address: str | None = None,
    adapter_kwargs: dict[str, Any] | None = None,
    auto_enable: bool = True,
) -> HardwareComponent:
    return HardwareComponent(
        hardware_id=hardware_id,
        hardware_type=HardwareType.MANIPULATOR,
        joints=nero_joints(hardware_id),
        adapter_type=adapter_type,
        address=address,
        auto_enable=auto_enable,
        adapter_kwargs=dict(adapter_kwargs or {}),
    )


def nero_real_hardware(
    hardware_id: str,
    *,
    address: str,
    firmware_version: str = NERO_FIRMWARE_VERSION,
    interface: str = "socketcan",
    effector_type: str | None = None,
    gripper_force: float = 1.0,
) -> HardwareComponent:
    adapter_kwargs: dict[str, Any] = {
        "firmware_version": firmware_version,
        "interface": interface,
    }
    if effector_type is not None:
        adapter_kwargs.update(
            {
                "effector_type": effector_type,
                "gripper_force": gripper_force,
            }
        )
    return nero_hardware(
        hardware_id,
        adapter_type="nero",
        address=address,
        adapter_kwargs=adapter_kwargs,
    )


def _nero_urdf_path() -> Path:
    """Resolve the single-arm Nero URDF from LFS or a local AgileX checkout."""
    for path in NERO_MODEL_PATH_FALLBACKS:
        if path.exists():
            return path
    # Return the preferred LFS path without touching .exists(); LfsPath.exists()
    # triggers git-lfs, but local sim mode should work without git-lfs installed.
    return NERO_MODEL_PATH


def _nero_package_paths() -> dict[str, Path]:
    """Resolve package:// paths used by the NERO, gripper, and camera models."""
    if (NERO_LOCAL_AGX_ARM_DESCRIPTION / NERO_MODEL_RELATIVE_PATH).exists():
        return dict(NERO_PACKAGE_PATHS)

    extracted_root = Path.home() / "wbcd_extracted"
    extracted_packages = {
        "agx_arm_description": extracted_root / "agx_arm_sim/agx_arm_description",
        "wbcd_urdf": extracted_root / "wbcd_urdf",
        "realsense2_description": extracted_root / "agx_arm_sim/realsense2_description",
    }
    if (extracted_packages["agx_arm_description"] / NERO_MODEL_RELATIVE_PATH).exists():
        return extracted_packages

    return {"agx_arm_description": AGX_ARM_DESCRIPTION_PKG}


def nero_body_static_model() -> dict[str, Any]:
    """Return Viser static-model config for the WBCD body/base."""
    return {
        "name": "nero_body",
        "model_path": NERO_BODY_MODEL_PATH,
        "package_paths": _nero_package_paths(),
        "load_meshes": True,
        "load_collision_meshes": False,
        "wxyz": (0.0, 0.0, 0.0, 1.0),  # 180° yaw rotation
    }


def nero_default_table_obstacle() -> dict[str, Any]:
    """Return startup planning obstacle for the measured table."""
    return {
        "name": NERO_TABLE_OBSTACLE_NAME,
        "shape": "box",
        "position": NERO_TABLE_CENTER,
        "dimensions": NERO_TABLE_SIZE,
    }


def nero_model_config(
    name: str,
    *,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    joint_prefix: str | None = None,
    coordinator_task_name: str | None = None,
) -> RobotModelConfig:
    """Return a per-arm RobotModelConfig for the NERO 7-DOF arm.

    Each arm uses the single-arm URDF placed at the given base_pose offset.
    For dual-arm setups pass y_offset / z_offset to position each arm
    relative to the world origin (same pattern as dual_xarm6_planner).

    Args:
        name: Robot name, e.g. "left_arm" or "right_arm".
        x_offset: X translation of the arm base in metres.
        y_offset: Y translation of the arm base in metres.
        z_offset: Z translation of the arm base in metres.
        roll: Roll rotation of the arm base in radians.
        pitch: Pitch rotation of the arm base in radians.
        yaw: Yaw rotation of the arm base in radians.
        joint_prefix: Override the joint-name prefix used in the coordinator
            joint mapping. Defaults to ``"<name>/"``.
        coordinator_task_name: Override the trajectory task name. Defaults to
            ``"traj_<name>"``.
    """
    local_joint_names = [f"joint{i}" for i in range(1, NERO_DOF + 1)]
    return RobotModelConfig(
        name=name,
        model_path=_nero_urdf_path(),
        base_pose=base_pose(
            x=x_offset,
            y=y_offset,
            z=z_offset,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        ),
        joint_names=local_joint_names,
        base_link="base_link",
        planning_groups=[
            PlanningGroupDefinition(
                name="manipulator",
                joint_names=tuple(local_joint_names),
                base_link="base_link",
                tip_link="link7",
            )
        ],
        package_paths=_nero_package_paths(),
        auto_convert_meshes=True,
        collision_exclusion_pairs=NERO_GRIPPER_COLLISION_EXCLUSIONS,
        joint_name_mapping=coordinator_joint_mapping(
            name,
            NERO_DOF,
            joint_prefix=joint_prefix,
        ),
        coordinator_task_name=coordinator_task_name or f"traj_{name}",
        home_joints=list(NERO_HOME_JOINTS),
    )
