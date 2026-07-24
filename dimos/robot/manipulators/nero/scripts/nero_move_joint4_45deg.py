#!/usr/bin/env python3

"""Use DimOS NeroAdapter to move joint 4 by +45 degrees on can0/can1.

How to run:
1. Bring up the CAN interface, for example:
   sudo ip link set can0 up type can bitrate 1000000
   sudo ip link set can1 up type can bitrate 1000000

2. Make sure DimOS and pyAgxArm are installed in the Python environment you use
   to run this script.

3. Run from the repository root:
   python3 dimos/robot/manipulators/nero/scripts/nero_move_joint4_45deg.py

Notes:
- Change CAN_CHANNELS if your arms use different CAN interfaces.
- Change FIRMWARE_VERSION to match your NERO firmware.
- This script goes through the DimOS NeroAdapter instead of calling pyAgxArm
  motion APIs directly.
- Do not call disable() while the arm is raised. AgileX documents that disabled
  NERO joints can drop immediately because servo holding torque is removed.
- Motion sequence: command zero first, move joint 4 by +45 degrees, wait 5
  seconds, command zero again, then keep the SDK connection open.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time

from dimos.hardware.manipulators.nero.adapter import NERO_DOF, NeroAdapter


CAN_CHANNELS = ["can0", "can1"]
FIRMWARE_VERSION = "v120"
JOINT_INDEX = 3  # Joint 4 in zero-based Python indexing.
DELTA_DEGREES = 45.0
HOLD_SECONDS_AFTER_DELTA = 5.0
ACTIVATE_TIMEOUT_S = 15.0
MOTION_START_GRACE_POLLS = 20
MOTION_POLL_INTERVAL_S = 0.05
ZERO_JOINTS = [0.0] * NERO_DOF

JOINT_LIMITS_RAD = [
    (-2.705261, 2.705261),
    (-1.745330, 1.745330),
    (-2.757621, 2.757621),
    (-1.012291, 2.146755),
    (-2.757621, 2.757621),
    (-0.733039, 0.959932),
    (-1.570797, 1.570797),
]


def wait_for_motion(channel: str, adapter: NeroAdapter, timeout_s: float = 10.0) -> None:
    # read_state() can briefly still report motion_status == 0 right after a
    # command is sent, so first wait for motion to actually begin. Otherwise we
    # would immediately (and wrongly) report "reached target" before the arm
    # has started moving.
    for _ in range(MOTION_START_GRACE_POLLS):
        if adapter.read_state().get("motion_status", 0) != 0:
            break
        time.sleep(MOTION_POLL_INTERVAL_S)
    else:
        print(f"{channel}: no motion detected (already at target?)", flush=True)
        return

    start_t = time.monotonic()
    while time.monotonic() - start_t < timeout_s:
        if adapter.read_state().get("motion_status", 0) == 0:
            print(f"{channel}: reached target position", flush=True)
            return
        time.sleep(MOTION_POLL_INTERVAL_S)
    print(f"{channel}: wait for motion timeout ({timeout_s:.1f}s)", flush=True)


def hold_enabled_until_interrupt(message: str) -> None:
    print(message, flush=True)
    while True:
        time.sleep(1.0)


def _diagnose_can(channels: list[str]) -> None:
    """Print CAN interface state and raw traffic before attempting SDK connect."""
    print("\n--- CAN DIAGNOSTICS ---", flush=True)

    # 1. Interface state via ip link
    for ch in channels:
        try:
            result = subprocess.run(
                ["ip", "-details", "link", "show", ch],
                capture_output=True, text=True, timeout=3,
            )
            state_line = next(
                (l for l in result.stdout.splitlines() if "state" in l.lower()), ""
            )
            print(f"  {ch} ip link: {state_line.strip()}", flush=True)
        except Exception as e:
            print(f"  {ch} ip link failed: {e}", flush=True)

    # 2. python-can open check
    try:
        import can
        for ch in channels:
            try:
                bus = can.Bus(channel=ch, interface="socketcan", bitrate=1_000_000)
                print(f"  {ch} python-can open: OK", flush=True)

                # 3. Listen for up to 1 second for any CAN frame from the arm
                print(f"  {ch} listening 1s for CAN frames from arm...", flush=True)
                deadline = time.monotonic() + 1.0
                frames_seen = 0
                while time.monotonic() < deadline:
                    msg = bus.recv(timeout=max(0, deadline - time.monotonic()))
                    if msg is not None:
                        frames_seen += 1
                        print(
                            f"  {ch} frame: id=0x{msg.arbitration_id:03X} "
                            f"dlc={msg.dlc} data={msg.data.hex()}",
                            flush=True,
                        )
                        if frames_seen >= 5:
                            print(f"  {ch} (stopping after 5 frames)", flush=True)
                            break
                if frames_seen == 0:
                    print(
                        f"  {ch} NO frames received — arm may be powered off, "
                        f"wrong bitrate, or wrong CAN channel.",
                        flush=True,
                    )
                bus.shutdown()
            except Exception as e:
                print(f"  {ch} python-can error: {e}", flush=True)
    except ImportError:
        print("  python-can not installed — skipping raw CAN check", flush=True)

    # 4. pyAgxArm availability
    try:
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
        print(f"  pyAgxArm: OK (AgxArmFactory importable)", flush=True)
    except ImportError as e:
        print(f"  pyAgxArm: MISSING — {e}", flush=True)
        print("  Fix: uv pip install ../pyAgxArm   (from dimos/ directory)", flush=True)
        sys.exit(1)

    print("--- END DIAGNOSTICS ---\n", flush=True)


def connect_and_enable(channel: str) -> NeroAdapter:
    print(f"{channel}: connecting NeroAdapter...", flush=True)
    adapter = NeroAdapter(
        address=channel,
        firmware_version=FIRMWARE_VERSION,
        interface="socketcan",
        bitrate=1_000_000,
        enable_retry_count=1,
        enable_call_timeout=0.2,
    )

    # Verbose connect: patch in extra prints around what the adapter does
    print(f"{channel}: calling sdk.connect()...", flush=True)
    connected = adapter.connect()
    if not connected:
        error_code, error_message = adapter.read_error()
        # Try to get more info directly from the SDK
        sdk = adapter._sdk
        if sdk is not None:
            try:
                has_err = sdk.has_comm_error() if hasattr(sdk, "has_comm_error") else "N/A"
                comm_err = sdk.get_comm_error() if hasattr(sdk, "has_comm_error") else "N/A"
                is_conn = sdk.is_connected() if hasattr(sdk, "is_connected") else "N/A"
                print(f"{channel}: sdk.has_comm_error()={has_err}", flush=True)
                print(f"{channel}: sdk.get_comm_error()={comm_err}", flush=True)
                print(f"{channel}: sdk.is_connected()={is_conn}", flush=True)
                # Try reading joint angles directly to see if it's a timing issue
                try:
                    ja = sdk.get_joint_angles()
                    print(f"{channel}: sdk.get_joint_angles()={ja}", flush=True)
                except Exception as e:
                    print(f"{channel}: sdk.get_joint_angles() raised: {e}", flush=True)
            except Exception as e:
                print(f"{channel}: sdk introspection failed: {e}", flush=True)
        else:
            print(f"{channel}: sdk is None — connect() raised an exception internally", flush=True)
            print(f"  Check the log output above for the root cause.", flush=True)

        raise RuntimeError(
            f"{channel}: failed to connect NERO adapter "
            f"(error_code={error_code}, error={error_message!r})"
        )
    print(f"{channel}: connected; activating servos...", flush=True)

    start_t = time.monotonic()
    last_report_t = 0.0
    while not adapter.activate():
        now = time.monotonic()
        if time.monotonic() - start_t > ACTIVATE_TIMEOUT_S:
            error_code, error_message = adapter.read_error()
            raise RuntimeError(
                f"{channel}: timed out activating NERO adapter after "
                f"{ACTIVATE_TIMEOUT_S:.1f}s "
                f"(enabled={adapter.read_enabled()}, state={adapter.read_state()}, "
                f"error_code={error_code}, error={error_message!r})"
            )
        if now - last_report_t >= 1.0:
            error_code, error_message = adapter.read_error()
            print(
                f"{channel}: waiting for NERO adapter activate "
                f"(enabled={adapter.read_enabled()}, state={adapter.read_state()}, "
                f"error_code={error_code}, error={error_message!r})",
                flush=True,
            )
            last_report_t = now
        time.sleep(0.1)
    print(f"{channel}: activated; enabled={adapter.read_enabled()}", flush=True)
    return adapter


def move_joint4_delta(channel: str, adapter: NeroAdapter) -> None:
    joints = adapter.read_joint_positions()
    print(f"{channel}: current joints:", joints, flush=True)

    joints[JOINT_INDEX] += math.radians(DELTA_DEGREES)
    lower, upper = JOINT_LIMITS_RAD[JOINT_INDEX]
    if not lower <= joints[JOINT_INDEX] <= upper:
        raise RuntimeError(
            f"{channel}: joint 4 target {joints[JOINT_INDEX]:.3f} rad is outside "
            f"NERO limits [{lower:.3f}, {upper:.3f}] rad"
        )
    print(f"{channel}: target joints:", joints, flush=True)

    if not adapter.write_joint_positions(joints):
        raise RuntimeError(f"{channel}: failed to command target joints through adapter")


def command_zero(adapters: dict[str, NeroAdapter]) -> None:
    for channel, adapter in adapters.items():
        print(f"{channel}: sending zero target", flush=True)
        if not adapter.write_joint_positions(ZERO_JOINTS):
            raise RuntimeError(f"{channel}: failed to command zero target through adapter")
    for channel, adapter in adapters.items():
        wait_for_motion(channel, adapter)


def main() -> None:
    adapters: dict[str, NeroAdapter] = {}

    _diagnose_can(CAN_CHANNELS)

    try:
        for channel in CAN_CHANNELS:
            adapters[channel] = connect_and_enable(channel)

        print("Commanding zero before test motion.")
        command_zero(adapters)

        for channel, adapter in adapters.items():
            move_joint4_delta(channel, adapter)

        for channel, adapter in adapters.items():
            wait_for_motion(channel, adapter)

        print(f"Holding +{DELTA_DEGREES:.1f} deg target for {HOLD_SECONDS_AFTER_DELTA:.1f}s.")
        time.sleep(HOLD_SECONDS_AFTER_DELTA)

        print("Commanding zero after test motion.")
        command_zero(adapters)

        hold_enabled_until_interrupt(
            "Zero target sent. Arms are enabled, connected, and holding position. "
            "Press Ctrl+C only when safe to exit this process."
        )

    except KeyboardInterrupt:
        print("Exiting without disable() or disconnect().")


if __name__ == "__main__":
    main()
