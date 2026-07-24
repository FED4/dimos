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

"""AgileX NERO adapter - implements ManipulatorAdapter protocol.

SDK Units: angles=radians, distance=meters, angular velocity=rad/s.
DimOS Units: angles=radians, distance=meters, angular velocity=rad/s.
"""

from __future__ import annotations

import time
from typing import Any

from dimos.hardware.manipulators.spec import (
    ControlMode,
    JointLimits,
    ManipulatorAdapter,
    ManipulatorInfo,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

NERO_DOF = 7
ENABLE_RETRY_COUNT = 50
ENABLE_RETRY_INTERVAL = 0.01
DEFAULT_ENABLE_CALL_TIMEOUT = 0.2
CONNECT_READ_TIMEOUT_S = 2.0
CONNECT_READ_POLL_INTERVAL_S = 0.05
DEFAULT_SPEED_PERCENT = 50
DEFAULT_FIRMWARE_VERSION = "v120"
DEFAULT_INTERFACE = "socketcan"
DEFAULT_BITRATE = 1_000_000
CPV_MOTION_MODE = "cpv"
JOINT_MOTION_MODE = "j"
AGX_GRIPPER_EFFECTOR = "agx_gripper"
DEFAULT_GRIPPER_FORCE = 1.0
DEFAULT_DISABLE_ON_DISCONNECT = False

DEFAULT_JOINT_LIMITS = JointLimits(
    position_lower=[
        -2.705261,
        -1.745330,
        -2.757621,
        -1.012291,
        -2.757621,
        -0.733039,
        -1.570797,
    ],
    position_upper=[
        2.705261,
        1.745330,
        2.757621,
        2.146755,
        2.757621,
        0.959932,
        1.570797,
    ],
    velocity_max=[1.0] * NERO_DOF,
)


class NeroAdapter(ManipulatorAdapter):
    """AgileX NERO 7-DOF hardware adapter using pyAgxArm over CAN."""

    def __init__(
        self,
        address: str = "can0",
        *,
        dof: int = NERO_DOF,
        firmware_version: str = DEFAULT_FIRMWARE_VERSION,
        interface: str = DEFAULT_INTERFACE,
        bitrate: int = DEFAULT_BITRATE,
        speed_percent: int = DEFAULT_SPEED_PERCENT,
        effector_type: str | None = None,
        gripper_force: float = DEFAULT_GRIPPER_FORCE,
        disable_on_disconnect: bool = DEFAULT_DISABLE_ON_DISCONNECT,
        enable_retry_count: int = ENABLE_RETRY_COUNT,
        enable_call_timeout: float = DEFAULT_ENABLE_CALL_TIMEOUT,
        **sdk_kwargs: object,
    ) -> None:
        if dof != NERO_DOF:
            raise ValueError(f"NeroAdapter only supports {NERO_DOF} DOF (got {dof})")
        if effector_type not in (None, AGX_GRIPPER_EFFECTOR):
            raise ValueError(f"Unsupported NERO effector: {effector_type}")
        self._channel = address
        self._firmware_version = firmware_version
        self._interface = interface
        self._bitrate = bitrate
        self._speed_percent = speed_percent
        self._effector_type = effector_type
        self._gripper_force = gripper_force
        self._disable_on_disconnect = disable_on_disconnect
        self._enable_retry_count = enable_retry_count
        self._enable_call_timeout = enable_call_timeout
        self._sdk_kwargs = sdk_kwargs
        self._sdk: Any | None = None
        self._effector: Any | None = None
        self._connected = False
        self._enabled = False
        self._control_mode = ControlMode.POSITION

    def connect(self) -> bool:
        """Connect to NERO via pyAgxArm."""
        sdk: Any | None = None
        try:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config

            cfg = create_agx_arm_config(
                robot="nero",
                firmeware_version=self._firmware_version,
                interface=self._interface,
                channel=self._channel,
                bitrate=self._bitrate,
                **self._sdk_kwargs,
            )
            sdk = AgxArmFactory.create_arm(cfg)
            # Establish the CAN bus first: pyAgxArm effector init talks to the
            # gripper over the bus, so it must run after connect().
            sdk.connect()
            self._sdk = sdk
            self._effector = self._init_effector(sdk)

            if self._has_comm_error():
                logger.error(
                    "Failed to connect to NERO: communication error",
                    can_port=self._channel,
                    error=self._get_comm_error(),
                )
                self._disconnect_sdk(sdk)
                self._clear_connection_state()
                return False

            # Poll for the first CAN feedback frame. The arm broadcasts at ~10 Hz
            # so the read thread may not have received any data yet immediately
            # after sdk.connect(). Retry for up to CONNECT_READ_TIMEOUT_S.
            deadline = time.monotonic() + CONNECT_READ_TIMEOUT_S
            positions = None
            while time.monotonic() < deadline:
                try:
                    positions = self.read_joint_positions()
                    break
                except RuntimeError:
                    time.sleep(CONNECT_READ_POLL_INTERVAL_S)
            if positions is not None:
                self._connected = True
                self._set_speed_percent(self._speed_percent)
                logger.info("NERO connected", can_port=self._channel)
                return True
        except ImportError:
            logger.exception("pyAgxArm is required for NeroAdapter")
        except Exception:
            logger.exception("Failed to connect to NERO", can_port=self._channel)

        if sdk is not None:
            self._disconnect_sdk(sdk)
        self._clear_connection_state()
        return False

    def disconnect(self) -> None:
        """Disconnect SDK communication resources.

        NERO joints are not disabled by default because disabling removes servo
        holding torque and can make a raised arm drop.
        """
        sdk = self._sdk
        if sdk is None:
            self._clear_connection_state()
            return

        if self._enabled and self._disable_on_disconnect:
            try:
                sdk.disable()
            except Exception:
                logger.exception("Failed to disable NERO during disconnect")

        self._disconnect_sdk(sdk)
        self._clear_connection_state()

    def is_connected(self) -> bool:
        if not self._connected or self._sdk is None:
            return False
        return not self._has_comm_error()

    def activate(self) -> bool:
        return self.write_enable(True)

    def deactivate(self) -> bool:
        return self.write_stop()

    def get_info(self) -> ManipulatorInfo:
        firmware_version = self._firmware_version
        if self._sdk is not None:
            try:
                firmware = self._sdk.get_firmware()
                if firmware is not None:
                    firmware_version = str(firmware.msg)
            except Exception:
                pass
        return ManipulatorInfo(
            vendor="AgileX",
            model="NERO",
            dof=NERO_DOF,
            firmware_version=firmware_version,
        )

    def get_dof(self) -> int:
        return NERO_DOF

    def get_limits(self) -> JointLimits:
        if self._sdk is None:
            return DEFAULT_JOINT_LIMITS

        lower: list[float] = []
        upper: list[float] = []
        velocity: list[float] = []
        try:
            for joint_index in range(1, NERO_DOF + 1):
                limit = self._sdk.get_joint_angle_vel_limits(
                    joint_index,
                    timeout=0.2,
                    min_interval=0.0,
                )
                if limit is None:
                    return DEFAULT_JOINT_LIMITS
                msg = limit.msg
                lower.append(float(msg.min_angle_limit))
                upper.append(float(msg.max_angle_limit))
                velocity.append(float(msg.max_joint_spd))
            return JointLimits(lower, upper, velocity)
        except Exception:
            logger.exception("Failed to read NERO joint limits")
            return DEFAULT_JOINT_LIMITS

    def set_control_mode(self, mode: ControlMode) -> bool:
        if mode not in (
            ControlMode.POSITION,
            ControlMode.SERVO_POSITION,
            ControlMode.VELOCITY,
        ):
            return False
        if self._sdk is not None and mode in (ControlMode.SERVO_POSITION, ControlMode.VELOCITY):
            if not self._has_cpv_support():
                logger.error(
                    "NERO CPV mode requested but pyAgxArm driver does not expose CPV APIs",
                    firmware_version=self._firmware_version,
                )
                return False
            if not self._set_motion_mode(CPV_MOTION_MODE):
                return False
        elif self._sdk is not None and mode == ControlMode.POSITION:
            self._set_motion_mode(JOINT_MOTION_MODE)
        self._control_mode = mode
        return True

    def get_control_mode(self) -> ControlMode:
        return self._control_mode

    def read_joint_positions(self) -> list[float]:
        sdk = self._require_sdk()
        joint_angles = sdk.get_joint_angles()
        if joint_angles is None:
            raise RuntimeError("Failed to read NERO joint positions")
        positions = list(joint_angles.msg)
        if len(positions) != NERO_DOF:
            raise RuntimeError(f"Expected {NERO_DOF} joint positions, got {len(positions)}")
        return [float(position) for position in positions]

    def read_joint_velocities(self) -> list[float]:
        return self._read_motor_field("velocity")

    def read_joint_efforts(self) -> list[float]:
        return self._read_motor_field("torque")

    def read_state(self) -> dict[str, int]:
        sdk = self._sdk
        if sdk is None:
            return {"state": 0, "mode": 0}
        try:
            status = sdk.get_arm_status()
            if status is None:
                return {"state": 0, "mode": 0}
            msg = status.msg
            error_code = int(getattr(msg, "err_code", 0))
            motion_status = int(getattr(msg, "motion_status", 0))
            return {
                "state": 2 if error_code else motion_status,
                "mode": 0,
                "error_code": error_code,
                "motion_status": motion_status,
            }
        except Exception:
            logger.exception("Failed to read NERO state")
            return {"state": 0, "mode": 0}

    def read_error(self) -> tuple[int, str]:
        if self._sdk is None:
            return 0, ""
        try:
            if self._has_comm_error():
                return 1, self._get_comm_error() or "Communication error"
            status = self._sdk.get_arm_status()
            if status is None:
                return 0, ""
            error_code = int(getattr(status.msg, "err_code", 0))
            if error_code == 0:
                return 0, ""
            return error_code, f"NERO controller error {error_code}"
        except Exception:
            logger.exception("Failed to read NERO error state")
            return 0, ""

    def write_joint_positions(
        self,
        positions: list[float],
        velocity: float = 1.0,
    ) -> bool:
        sdk = self._sdk
        if sdk is None or len(positions) != NERO_DOF:
            return False
        try:
            if self._control_mode == ControlMode.SERVO_POSITION:
                return self._write_cpv_positions(positions)
            else:
                self._set_speed_percent(max(1, min(100, round(velocity * 100))))
                sdk.move_j([float(position) for position in positions])
            return True
        except Exception:
            logger.exception("NERO joint command failed")
            return False

    def write_joint_velocities(self, velocities: list[float]) -> bool:
        if self._sdk is None or len(velocities) != NERO_DOF:
            return False
        if self._control_mode != ControlMode.VELOCITY and not self.set_control_mode(
            ControlMode.VELOCITY
        ):
            return False
        try:
            return self._write_cpv_velocities(velocities)
        except Exception:
            logger.exception("NERO velocity command failed")
            return False

    def write_stop(self) -> bool:
        sdk = self._sdk
        if sdk is None:
            return False
        try:
            if hasattr(sdk, "electronic_emergency_stop"):
                sdk.electronic_emergency_stop()
                return True
            # Fallback: hold the current pose with a joint move. The SDK may be
            # in CPV motion mode (servo/velocity control), where move_j is
            # rejected, so switch back to joint motion mode first.
            if not self._set_motion_mode(JOINT_MOTION_MODE):
                logger.error("Failed to switch NERO to joint mode for stop fallback")
                return False
            self._control_mode = ControlMode.POSITION
            sdk.move_j(self.read_joint_positions())
            return True
        except Exception:
            logger.exception("Failed to stop NERO motion")
            return False

    def write_enable(self, enable: bool) -> bool:
        sdk = self._sdk
        if sdk is None:
            return False
        try:
            if enable:
                if self._enabled:
                    return True
                for attempt in range(self._enable_retry_count):
                    if self._enable_sdk():
                        self._enabled = True
                        return True
                    if attempt < self._enable_retry_count - 1:
                        time.sleep(ENABLE_RETRY_INTERVAL)
                return False
            if sdk.disable():
                self._enabled = False
                return True
            return False
        except Exception:
            logger.exception("Failed to change NERO enable state", enable=enable)
            return False

    def read_enabled(self) -> bool:
        sdk = self._sdk
        if sdk is None:
            return self._enabled
        try:
            states = sdk.get_joints_enable_status_list()
            if states is None:
                return self._enabled
            return all(bool(state) for state in states)
        except Exception:
            return self._enabled

    def write_clear_errors(self) -> bool:
        sdk = self._sdk
        if sdk is None:
            return False
        for method_name in ("clear_joint_error", "reset"):
            method = getattr(sdk, method_name, None)
            if method is None:
                continue
            try:
                result = method()
                return True if result is None else bool(result)
            except Exception:
                logger.exception("Failed to clear NERO errors", method=method_name)
        return False

    def read_cartesian_position(self) -> dict[str, float] | None:
        sdk = self._sdk
        if sdk is None:
            return None
        try:
            pose = sdk.get_flange_pose()
            if pose is None:
                return None
            x, y, z, roll, pitch, yaw = pose.msg
            return {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            }
        except Exception:
            logger.exception("Failed to read NERO flange pose")
            return None

    def write_cartesian_position(
        self,
        pose: dict[str, float],
        velocity: float = 1.0,
    ) -> bool:
        sdk = self._sdk
        if sdk is None:
            return False
        try:
            self._set_speed_percent(max(1, min(100, round(velocity * 100))))
            sdk.move_p(
                [
                    pose["x"],
                    pose["y"],
                    pose["z"],
                    pose["roll"],
                    pose["pitch"],
                    pose["yaw"],
                ]
            )
            return True
        except Exception:
            logger.exception("NERO Cartesian command failed")
            return False

    def read_gripper_position(self) -> float | None:
        effector = self._effector
        if effector is None:
            return None
        try:
            status = effector.get_gripper_status()
            if status is None:
                return None
            return float(status.msg.value)
        except Exception:
            logger.exception("Failed to read NERO gripper position")
            return None

    def write_gripper_position(self, position: float) -> bool:
        effector = self._effector
        if effector is None:
            return False
        try:
            effector.move_gripper_m(
                value=max(0.0, float(position)),
                force=max(0.0, float(self._gripper_force)),
            )
            return True
        except Exception:
            logger.exception("Failed to command NERO gripper")
            return False

    def read_force_torque(self) -> list[float] | None:
        return None

    def _require_sdk(self) -> Any:
        if self._sdk is None:
            raise RuntimeError("Not connected")
        return self._sdk

    def _init_effector(self, sdk: Any) -> Any | None:
        if self._effector_type is None:
            return None
        if self._effector_type != AGX_GRIPPER_EFFECTOR:
            logger.error("Unsupported NERO effector", effector_type=self._effector_type)
            return None
        effector_option = getattr(
            getattr(getattr(sdk, "OPTIONS", None), "EFFECTOR", None),
            "AGX_GRIPPER",
            AGX_GRIPPER_EFFECTOR,
        )
        return sdk.init_effector(effector_option)

    def _read_motor_field(self, field_name: str) -> list[float]:
        sdk = self._sdk
        if sdk is None:
            raise RuntimeError("Not connected")
        values: list[float] = []
        for joint_index in range(1, NERO_DOF + 1):
            state = sdk.get_motor_states(joint_index)
            values.append(0.0 if state is None else float(getattr(state.msg, field_name, 0.0)))
        return values

    def _write_cpv_positions(self, positions: list[float]) -> bool:
        sdk = self._sdk
        if sdk is None or not self._has_cpv_support():
            return False
        if not self._set_motion_mode(CPV_MOTION_MODE):
            return False
        for joint_index, position in enumerate(positions, start=1):
            sdk.move_cpv_pos(joint_index, float(position))
        return True

    def _write_cpv_velocities(self, velocities: list[float]) -> bool:
        sdk = self._sdk
        if sdk is None or not self._has_cpv_support():
            return False
        if not self._set_motion_mode(CPV_MOTION_MODE):
            return False
        for joint_index, velocity in enumerate(velocities, start=1):
            sdk.move_cpv_vel(joint_index, float(velocity))
        return True

    def _has_cpv_support(self) -> bool:
        sdk = self._sdk
        return (
            sdk is not None
            and hasattr(sdk, "set_motion_mode")
            and hasattr(sdk, "move_cpv_pos")
            and hasattr(sdk, "move_cpv_vel")
        )

    def _set_motion_mode(self, motion_mode: str) -> bool:
        sdk = self._sdk
        if sdk is None or not hasattr(sdk, "set_motion_mode"):
            return True
        try:
            sdk.set_motion_mode(motion_mode)
            return True
        except Exception:
            logger.exception("Failed to set NERO motion mode", motion_mode=motion_mode)
            return False

    def _set_speed_percent(self, speed_percent: int) -> None:
        sdk = self._sdk
        if sdk is None:
            return
        try:
            sdk.set_speed_percent(max(1, min(100, speed_percent)))
        except Exception:
            logger.exception("Failed to set NERO speed percent", speed_percent=speed_percent)

    def _enable_sdk(self) -> bool:
        sdk = self._sdk
        if sdk is None:
            return False
        try:
            return bool(sdk.enable(timeout=self._enable_call_timeout))
        except TypeError:
            return bool(sdk.enable())

    def _disconnect_sdk(self, sdk: Any) -> None:
        try:
            sdk.disconnect()
        except Exception:
            logger.exception("Failed to disconnect NERO", can_port=self._channel)

    def _has_comm_error(self) -> bool:
        sdk = self._sdk
        if sdk is None or not hasattr(sdk, "has_comm_error"):
            return False
        try:
            return bool(sdk.has_comm_error())
        except Exception:
            logger.exception("Failed to query NERO communication status")
            return True

    def _get_comm_error(self) -> str:
        sdk = self._sdk
        if sdk is None or not hasattr(sdk, "get_comm_error"):
            return ""
        try:
            return str(sdk.get_comm_error())
        except Exception:
            return ""

    def _clear_connection_state(self) -> None:
        self._sdk = None
        self._effector = None
        self._connected = False
        self._enabled = False
