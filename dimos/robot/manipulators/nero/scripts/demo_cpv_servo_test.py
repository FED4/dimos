#!/usr/bin/env python3
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

"""CPV 100 Hz servo streaming validation for the AgileX NERO arm.

Tests whether move_cpv_pos can sustain 100 Hz joint-position commands via the
DimOS coordinator servo task, measures achieved rate and command→feedback
latency, and logs pass/fail summary to console.

Usage (two terminals)
---------------------
Terminal 1 - start the coordinator (one arm):
    dimos run coordinator-nero-servo-left

Terminal 2 - run this script:
    python -m dimos.robot.manipulators.nero.scripts.demo_cpv_servo_test

Arguments
---------
--arm           Which arm to test: "left" (default) or "right".
--duration      Test duration in seconds (default: 20).
--amplitude-deg Oscillation amplitude in degrees on joint4 (default: 5).
--freq-hz       Oscillation frequency in Hz (default: 0.25).
--rate-hz       Target command publish rate in Hz (default: 100).
--joint-index   Joint index (1-7) to oscillate (default: 4).

What the script does
--------------------
1.  Checks that pyAgxArm exposes move_cpv_pos, move_cpv_vel, set_motion_mode
    on the Nero v120 driver (Step 1 of the test plan).

2.  Subscribes to /coordinator_joint_state to read current arm position and
    waits until joint state is received (arm is live and coordinator is up).

3.  Publishes a sinusoidal trajectory on the chosen joint at the requested
    rate.  All other joints are held at their initial position.

4.  Every tick it records:
      - t_cmd    : wall time when the command was published
      - q_cmd[j] : commanded position on the oscillating joint
      - q_act[j] : latest measured position from coordinator_joint_state
      - delta_ms : one-sided latency estimate (t_feedback - t_cmd_of_nearest)

5.  Prints a rolling summary every 2 s and a final report:
      - Achieved publish rate (Hz)
      - Mean / max absolute tracking error (rad and deg)
      - Estimated one-sided command→feedback latency (ms)
      - Drop count (ticks where no new feedback arrived)
      - Pass/Fail verdict
"""

from __future__ import annotations

import argparse
import math
import signal
import statistics
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Step 1: CPV availability check (no hardware connection needed)
# ---------------------------------------------------------------------------

def check_cpv_support() -> bool:
    """Verify pyAgxArm v120 driver exposes CPV methods without connecting."""
    try:
        from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v120.driver import (
            Driver as V120Driver,
        )
    except ImportError:
        print("[FAIL] pyAgxArm not installed or v120 driver not found.")
        print("       Install with: pip install pyAgxArm")
        return False

    required = ("set_motion_mode", "move_cpv_pos", "move_cpv_vel")
    missing = [m for m in required if not hasattr(V120Driver, m)]
    if missing:
        print(f"[FAIL] v120 driver missing CPV methods: {missing}")
        print("       CPV is not available; consider MIT mode instead.")
        return False

    print("[OK]  pyAgxArm v120 driver has: set_motion_mode, move_cpv_pos, move_cpv_vel")
    return True


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["left", "right"], default="left",
                   help="Which arm to test (default: left)")
    p.add_argument("--duration", type=float, default=20.0,
                   help="Test duration in seconds (default: 20)")
    p.add_argument("--amplitude-deg", type=float, default=5.0,
                   help="Oscillation amplitude on the test joint in degrees (default: 5)")
    p.add_argument("--freq-hz", type=float, default=0.25,
                   help="Oscillation frequency in Hz (default: 0.25)")
    p.add_argument("--rate-hz", type=float, default=100.0,
                   help="Target publish rate in Hz (default: 100)")
    p.add_argument("--joint-index", type=int, default=4, choices=range(1, 8),
                   metavar="1-7",
                   help="Joint to oscillate (1-indexed, default: 4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("NERO CPV 100 Hz Servo Streaming Test")
    print("=" * 60)

    # -- Step 1: CPV availability ----------------------------------------
    print("\n[Step 1] Checking pyAgxArm CPV support...")
    if not check_cpv_support():
        sys.exit(1)

    # -- Imports ---------------------------------------------------------
    try:
        from dimos.core.transport import LCMTransport
        from dimos.msgs.sensor_msgs.JointState import JointState
    except ImportError as e:
        print(f"[FAIL] DimOS import error: {e}")
        print("       Run from inside the dimos repo with: uv run python -m ...")
        sys.exit(1)

    arm = args.arm
    hardware_id = f"{arm}_arm"
    joint_names = [f"{hardware_id}/joint{i}" for i in range(1, 8)]
    test_joint_idx = args.joint_index - 1  # zero-based index into joint_names

    dt = 1.0 / args.rate_hz
    amplitude = math.radians(args.amplitude_deg)
    duration = args.duration
    freq = args.freq_hz

    print(f"\n[Config]")
    print(f"  Arm:            {hardware_id}")
    print(f"  Test joint:     joint{args.joint_index} ({joint_names[test_joint_idx]})")
    print(f"  Amplitude:      ±{args.amplitude_deg:.1f}° ({amplitude:.4f} rad)")
    print(f"  Frequency:      {freq} Hz")
    print(f"  Publish rate:   {args.rate_hz} Hz")
    print(f"  Duration:       {duration} s")
    print(f"  Expected ticks: {int(duration * args.rate_hz)}")

    # -- Wait for coordinator joint state --------------------------------
    print(f"\n[Step 2] Waiting for coordinator_joint_state on /coordinator_joint_state ...")
    print("         (Make sure 'dimos run coordinator-nero-servo-{arm}' is running)")

    latest_state: dict[str, float] = {}
    state_lock = threading.Lock()
    state_received = threading.Event()
    state_timestamps: list[float] = []

    def on_joint_state(msg: JointState) -> None:
        t = time.perf_counter()
        with state_lock:
            for name, pos in zip(msg.name, msg.position, strict=False):
                latest_state[name] = pos
            state_timestamps.append(t)
            if not state_received.is_set() and any(n in latest_state for n in joint_names):
                state_received.set()

    state_sub: LCMTransport[JointState] = LCMTransport(
        "/coordinator_joint_state", JointState
    )
    state_sub.subscribe(on_joint_state)

    if not state_received.wait(timeout=10.0):
        print("[FAIL] No joint state received after 10 s.")
        print("       Is 'dimos run coordinator-nero-servo-left' running?")
        print("       Is the arm connected and enabled?")
        sys.exit(1)

    with state_lock:
        q_home = [latest_state.get(n, 0.0) for n in joint_names]

    print(f"[OK]  Joint state received. Home positions (rad):")
    for i, (n, q) in enumerate(zip(joint_names, q_home)):
        print(f"        joint{i+1}: {q:+.4f} rad ({math.degrees(q):+.2f}°)")

    # -- Command publisher -----------------------------------------------
    cmd_pub: LCMTransport[JointState] = LCMTransport("/joint_command", JointState)

    # -- Safety guard: SIGINT stops cleanly ------------------------------
    running = threading.Event()
    running.set()

    def _stop(sig: int, frame: object) -> None:
        print("\n[INFO] Interrupted — stopping...")
        running.clear()

    signal.signal(signal.SIGINT, _stop)

    # -- Measurement storage ---------------------------------------------
    cmd_times: list[float] = []
    cmd_positions: list[float] = []
    act_positions: list[float] = []
    latency_estimates_ms: list[float] = []
    last_feedback_t = time.perf_counter()
    dropped_ticks = 0

    # -- Main loop -------------------------------------------------------
    print(f"\n[Step 3] Publishing sinusoidal trajectory at {args.rate_hz} Hz for {duration} s...")
    print("         Ctrl-C to stop early.\n")

    t_start = time.perf_counter()
    last_summary_t = t_start
    tick = 0

    while running.is_set():
        t_now = time.perf_counter()
        t_elapsed = t_now - t_start

        if t_elapsed >= duration:
            break

        # Sinusoidal position on test joint, hold home on all others
        q_target = list(q_home)
        q_target[test_joint_idx] = (
            q_home[test_joint_idx] + amplitude * math.sin(2.0 * math.pi * freq * t_elapsed)
        )

        # Publish command
        msg = JointState(
            name=joint_names,
            position=q_target,
            velocity=[0.0] * 7,
            effort=[0.0] * 7,
        )
        cmd_pub.publish(msg)
        t_cmd = time.perf_counter()

        cmd_times.append(t_cmd)
        cmd_positions.append(q_target[test_joint_idx])

        # Sample feedback
        with state_lock:
            q_act = latest_state.get(joint_names[test_joint_idx])
            latest_fb_t = state_timestamps[-1] if state_timestamps else 0.0

        if q_act is not None:
            act_positions.append(q_act)
            # Latency: time from our command to the feedback timestamp
            # (one-sided estimate — ignores CAN round-trip asymmetry)
            lat = (latest_fb_t - t_cmd) * 1000.0  # ms, will be negative if feedback lags
            latency_estimates_ms.append(lat)
            last_feedback_t = latest_fb_t
        else:
            dropped_ticks += 1

        tick += 1

        # Rolling summary every 2 s
        if t_now - last_summary_t >= 2.0:
            elapsed = t_now - t_start
            actual_rate = tick / elapsed if elapsed > 0 else 0.0
            if act_positions and cmd_positions[:len(act_positions)]:
                paired = min(len(cmd_positions), len(act_positions))
                errors = [abs(cmd_positions[i] - act_positions[i]) for i in range(paired)]
                mean_err_deg = math.degrees(statistics.mean(errors))
                max_err_deg = math.degrees(max(errors))
                err_str = f"mean={mean_err_deg:.2f}° max={max_err_deg:.2f}°"
            else:
                err_str = "no feedback yet"
            print(
                f"  t={elapsed:5.1f}s  rate={actual_rate:5.1f}Hz  "
                f"tracking err: {err_str}  dropped={dropped_ticks}"
            )
            last_summary_t = t_now

        # Precise rate limiting
        t_next = t_start + (tick * dt)
        sleep_remaining = t_next - time.perf_counter()
        if sleep_remaining > 0:
            time.sleep(sleep_remaining)

    # -- Send stop (hold current position) --------------------------------
    with state_lock:
        q_final = [latest_state.get(n, q_home[i]) for i, n in enumerate(joint_names)]
    cmd_pub.publish(JointState(
        name=joint_names,
        position=q_final,
        velocity=[0.0] * 7,
        effort=[0.0] * 7,
    ))

    # -- Final report -----------------------------------------------------
    total_time = time.perf_counter() - t_start
    achieved_rate = tick / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total ticks published:  {tick}")
    print(f"  Duration:               {total_time:.2f} s")
    print(f"  Achieved publish rate:  {achieved_rate:.1f} Hz  (target: {args.rate_hz} Hz)")
    print(f"  Dropped feedback ticks: {dropped_ticks}")

    if act_positions and cmd_positions:
        paired = min(len(cmd_positions), len(act_positions))
        errors_rad = [abs(cmd_positions[i] - act_positions[i]) for i in range(paired)]
        mean_err = statistics.mean(errors_rad)
        max_err = max(errors_rad)
        print(f"  Tracking error (joint{args.joint_index}):")
        print(f"    mean: {mean_err:.4f} rad  ({math.degrees(mean_err):.2f}°)")
        print(f"    max:  {max_err:.4f} rad  ({math.degrees(max_err):.2f}°)")
    else:
        print("  [WARN] No paired feedback samples — was the arm actually moving?")

    if latency_estimates_ms:
        # Positive = feedback arrived after command (normal); negative = stale feedback
        valid_lat = [l for l in latency_estimates_ms if l > -500]
        if valid_lat:
            mean_lat = statistics.mean(valid_lat)
            max_lat = max(valid_lat)
            print(f"  Command→feedback latency (one-sided estimate):")
            print(f"    mean: {mean_lat:.1f} ms")
            print(f"    max:  {max_lat:.1f} ms")

    # -- Pass/fail verdict -----------------------------------------------
    rate_ok = achieved_rate >= args.rate_hz * 0.95  # within 5% of target
    feedback_ok = len(act_positions) >= tick * 0.8  # at least 80% of ticks got feedback
    error_ok = (
        statistics.mean(errors_rad) < math.radians(2.0)  # mean < 2°
        if act_positions and cmd_positions
        else False
    )

    print("\n  Checks:")
    print(f"    Rate ≥ {args.rate_hz*0.95:.0f} Hz:      {'PASS' if rate_ok else 'FAIL'}")
    print(f"    Feedback coverage ≥ 80%:  {'PASS' if feedback_ok else 'FAIL'}")
    print(f"    Mean tracking error < 2°: {'PASS' if error_ok else 'FAIL  (check CPV mode active)'}")

    verdict = all([rate_ok, feedback_ok, error_ok])
    print(f"\n  VERDICT: {'PASS — CPV 100 Hz servo is viable' if verdict else 'FAIL — review output above'}")

    if not verdict and not rate_ok:
        print("\n  Hint: Rate failure usually means the publish loop is blocked.")
        print("        Check if the coordinator is running and LCM is reachable.")

    if not verdict and not error_ok:
        print("\n  Hint: Large tracking error may mean:")
        print("        - CPV mode was not activated (move_cpv_pos falling back to move_j)")
        print("        - Arm is not enabled (check: sdk.enable())")
        print("        - Joint limits clamping the command")
        print("        - Consider MIT mode: pyAgxArm Nero v111/v112 driver.move_mit()")

    print("=" * 60)
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
