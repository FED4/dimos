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

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest


class FakeNeroAdapter:
    instances: list["FakeNeroAdapter"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.address = kwargs["address"]
        self.calls: list[str] = []
        self.written_positions: list[list[float]] = []
        FakeNeroAdapter.instances.append(self)

    def connect(self) -> bool:
        self.calls.append("connect")
        return True

    def activate(self) -> bool:
        self.calls.append("activate")
        return True

    def read_error(self) -> tuple[int, str]:
        self.calls.append("read_error")
        return 0, ""

    def read_joint_positions(self) -> list[float]:
        self.calls.append("read_joint_positions")
        return [0.0] * 7

    def write_joint_positions(self, positions: list[float]) -> bool:
        self.calls.append("write_joint_positions")
        self.written_positions.append(list(positions))
        return True

    def read_state(self) -> dict[str, int]:
        self.calls.append("read_state")
        return {"motion_status": 0}

    def disable(self) -> None:
        raise AssertionError("sample script must not disable NERO")

    def disconnect(self) -> None:
        raise AssertionError("sample script must not disconnect NERO")


@pytest.fixture()
def script_module(monkeypatch: pytest.MonkeyPatch):
    FakeNeroAdapter.instances.clear()
    fake_adapter_module = types.ModuleType("dimos.hardware.manipulators.nero.adapter")
    fake_adapter_module.NERO_DOF = 7
    fake_adapter_module.NeroAdapter = FakeNeroAdapter
    monkeypatch.setitem(
        sys.modules,
        "dimos.hardware.manipulators.nero.adapter",
        fake_adapter_module,
    )

    script_path = (
        Path(__file__).parent
        / "scripts"
        / "nero_move_joint4_45deg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "nero_move_joint4_45deg_under_test",
        script_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_joint4_script_uses_adapter_for_both_can_channels(script_module, monkeypatch) -> None:
    sleep_calls: list[float] = []
    hold_messages: list[str] = []

    monkeypatch.setattr(script_module.time, "sleep", sleep_calls.append)

    def stop_after_final_hold(message: str) -> None:
        hold_messages.append(message)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        script_module,
        "hold_enabled_until_interrupt",
        stop_after_final_hold,
    )

    script_module.main()

    assert [adapter.address for adapter in FakeNeroAdapter.instances] == ["can0", "can1"]
    assert sleep_calls == [5.0]
    assert len(hold_messages) == 1

    zero = [0.0] * 7
    delta = [0.0] * 7
    delta[3] = math.radians(45.0)

    for adapter in FakeNeroAdapter.instances:
        assert adapter.kwargs == {
            "address": adapter.address,
            "firmware_version": "v120",
            "interface": "socketcan",
            "bitrate": 1_000_000,
        }
        assert adapter.written_positions == [zero, delta, zero]
        assert "connect" in adapter.calls
        assert "activate" in adapter.calls
        assert "read_joint_positions" in adapter.calls
        assert "read_state" in adapter.calls
