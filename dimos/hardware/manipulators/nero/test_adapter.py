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

from types import SimpleNamespace

from dimos.hardware.manipulators.nero.adapter import NERO_DOF, NeroAdapter
from dimos.hardware.manipulators.spec import ControlMode


class FakeNeroSdk:
    def __init__(self) -> None:
        self.motion_modes: list[str] = []
        self.move_j_calls: list[list[float]] = []
        self.move_cpv_pos_calls: list[tuple[int, float]] = []
        self.move_cpv_vel_calls: list[tuple[int, float]] = []
        self.speed_percent: list[int] = []
        self.events: list[str] = []
        self.disable_calls = 0
        self.disconnect_calls = 0
        self.effector = FakeAgxGripper()
        self.OPTIONS = SimpleNamespace(
            EFFECTOR=SimpleNamespace(AGX_GRIPPER="agx_gripper")
        )

    def init_effector(self, effector_type: str) -> "FakeAgxGripper":
        self.events.append(f"init_effector:{effector_type}")
        return self.effector

    def connect(self) -> None:
        self.events.append("connect")

    def disable(self) -> bool:
        self.disable_calls += 1
        return True

    def disconnect(self) -> None:
        self.disconnect_calls += 1

    def set_motion_mode(self, motion_mode: str) -> None:
        self.motion_modes.append(motion_mode)

    def set_speed_percent(self, speed_percent: int) -> None:
        self.speed_percent.append(speed_percent)

    def move_j(self, joints: list[float]) -> None:
        self.move_j_calls.append(joints)

    def move_cpv_pos(self, joint_index: int, pos: float) -> None:
        self.move_cpv_pos_calls.append((joint_index, pos))

    def move_cpv_vel(self, joint_index: int, vel: float) -> None:
        self.move_cpv_vel_calls.append((joint_index, vel))

    def get_motor_states(self, joint_index: int) -> SimpleNamespace:
        return SimpleNamespace(
            msg=SimpleNamespace(
                velocity=float(joint_index),
                torque=float(joint_index) * 0.1,
            )
        )


class FakeAgxGripper:
    def __init__(self) -> None:
        self.move_gripper_m_calls: list[tuple[float, float]] = []
        self.status = SimpleNamespace(msg=SimpleNamespace(value=0.035, force=1.2))

    def get_gripper_status(self) -> SimpleNamespace:
        return self.status

    def move_gripper_m(self, value: float, force: float) -> None:
        self.move_gripper_m_calls.append((value, force))


def test_position_mode_uses_move_j() -> None:
    sdk = FakeNeroSdk()
    adapter = NeroAdapter()
    adapter._sdk = sdk

    positions = [0.1 * i for i in range(NERO_DOF)]

    assert adapter.set_control_mode(ControlMode.POSITION)
    assert adapter.write_joint_positions(positions, velocity=0.42)

    assert sdk.motion_modes == ["j"]
    assert sdk.speed_percent == [42]
    assert sdk.move_j_calls == [positions]
    assert sdk.move_cpv_pos_calls == []


def test_servo_position_mode_uses_cpv_position_per_joint() -> None:
    sdk = FakeNeroSdk()
    adapter = NeroAdapter()
    adapter._sdk = sdk

    positions = [0.1 * i for i in range(NERO_DOF)]

    assert adapter.set_control_mode(ControlMode.SERVO_POSITION)
    assert adapter.write_joint_positions(positions)

    assert sdk.motion_modes == ["cpv", "cpv"]
    assert sdk.move_j_calls == []
    assert sdk.move_cpv_pos_calls == [
        (joint_index, positions[joint_index - 1])
        for joint_index in range(1, NERO_DOF + 1)
    ]


def test_velocity_mode_uses_cpv_velocity_per_joint() -> None:
    sdk = FakeNeroSdk()
    adapter = NeroAdapter()
    adapter._sdk = sdk

    velocities = [0.2 * i for i in range(NERO_DOF)]

    assert adapter.write_joint_velocities(velocities)

    assert sdk.motion_modes == ["cpv", "cpv"]
    assert sdk.move_cpv_vel_calls == [
        (joint_index, velocities[joint_index - 1])
        for joint_index in range(1, NERO_DOF + 1)
    ]


def test_cpv_modes_fail_when_sdk_lacks_cpv_methods() -> None:
    adapter = NeroAdapter()
    adapter._sdk = SimpleNamespace(set_motion_mode=lambda _: None)

    assert not adapter.set_control_mode(ControlMode.SERVO_POSITION)
    assert not adapter.write_joint_velocities([0.0] * NERO_DOF)


def test_motor_feedback_maps_to_velocity_and_effort() -> None:
    adapter = NeroAdapter()
    adapter._sdk = FakeNeroSdk()

    assert adapter.read_joint_velocities() == [float(i) for i in range(1, NERO_DOF + 1)]
    assert adapter.read_joint_efforts() == [float(i) * 0.1 for i in range(1, NERO_DOF + 1)]


def test_no_configured_effector_has_no_gripper() -> None:
    adapter = NeroAdapter()
    adapter._sdk = FakeNeroSdk()

    assert adapter.read_gripper_position() is None
    assert not adapter.write_gripper_position(0.03)


def test_agx_gripper_init_and_read_write_mapping() -> None:
    sdk = FakeNeroSdk()
    adapter = NeroAdapter(effector_type="agx_gripper", gripper_force=1.5)

    adapter._effector = adapter._init_effector(sdk)
    sdk.connect()

    assert sdk.events == ["init_effector:agx_gripper", "connect"]
    assert adapter.read_gripper_position() == 0.035
    assert adapter.write_gripper_position(0.04)
    assert sdk.effector.move_gripper_m_calls == [(0.04, 1.5)]


def test_disconnect_does_not_disable_by_default() -> None:
    sdk = FakeNeroSdk()
    adapter = NeroAdapter()
    adapter._sdk = sdk
    adapter._connected = True
    adapter._enabled = True

    adapter.disconnect()

    assert sdk.disable_calls == 0
    assert sdk.disconnect_calls == 1
    assert not adapter.is_connected()


def test_disconnect_can_disable_when_explicitly_configured() -> None:
    sdk = FakeNeroSdk()
    adapter = NeroAdapter(disable_on_disconnect=True)
    adapter._sdk = sdk
    adapter._connected = True
    adapter._enabled = True

    adapter.disconnect()

    assert sdk.disable_calls == 1
    assert sdk.disconnect_calls == 1
    assert not adapter.is_connected()
