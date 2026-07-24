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

from __future__ import annotations

from dimos.robot.manipulators.nero.config import NERO_DOF, nero_model_config


def test_nero_model_uses_repo_assets_with_gripper_and_camera_dependencies() -> None:
    config = nero_model_config("left_arm")

    assert config.model_path.name == "nero_with_gripper_description.xacro"
    assert config.model_path.exists()
    assert config.joint_names == [f"joint{i}" for i in range(1, NERO_DOF + 1)]
    assert config.end_effector_link == "link7"

    package_paths = config.package_paths
    assert set(package_paths) == {
        "agx_arm_description",
        "wbcd_urdf",
        "realsense2_description",
    }
    assert (
        package_paths["agx_arm_description"]
        / "agx_arm_urdf/nero/meshes/dae/gripper_base.dae"
    ).exists()
    assert (package_paths["agx_arm_description"] / "meshes/realsense_mid_stand.dae").exists()
    assert (package_paths["realsense2_description"] / "meshes/d435.dae").exists()
    assert (package_paths["wbcd_urdf"] / "urdf/dual_nero.xacro").exists()
