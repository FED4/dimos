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

NERO_DOF = 7
NERO_LEFT_CAN = "can0"
NERO_RIGHT_CAN = "can1"
NERO_FIRMWARE_VERSION = "v120"
NERO_AGX_GRIPPER = "agx_gripper"

AGX_ARM_URDF_PKG = LfsPath("agx_arm_urdf")
NERO_MODEL_PATH = AGX_ARM_URDF_PKG / "nero/urdf/nero_description.urdf"
NERO_PACKAGE_PATHS: dict[str, Path] = {"agx_arm_urdf": AGX_ARM_URDF_PKG}
NERO_HOME_JOINTS = [0.0] * NERO_DOF


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


def nero_model_config(
    name: str,
    *,
    joint_prefix: str | None = None,
    coordinator_task_name: str | None = None,
) -> RobotModelConfig:
    local_joint_names = [f"joint{i}" for i in range(1, NERO_DOF + 1)]
    return RobotModelConfig(
        name=name,
        model_path=NERO_MODEL_PATH,
        base_pose=base_pose(),
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
        package_paths=NERO_PACKAGE_PATHS,
        auto_convert_meshes=True,
        joint_name_mapping=coordinator_joint_mapping(
            name,
            NERO_DOF,
            joint_prefix=joint_prefix,
        ),
        coordinator_task_name=coordinator_task_name or f"traj_{name}",
        home_joints=list(NERO_HOME_JOINTS),
    )
