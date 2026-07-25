from __future__ import annotations

from types import SimpleNamespace

import pytest

from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.control.tasks.teleop_task import teleop_task
from dimos.control.tasks.teleop_task._registry import TASK_EXPOSES


def _make_task(monkeypatch: pytest.MonkeyPatch) -> teleop_task.TeleopIKTask:
    monkeypatch.setattr(
        teleop_task.PinocchioIK,
        "from_model_path",
        staticmethod(lambda *_args, **_kwargs: SimpleNamespace(nq=2)),
    )
    return teleop_task.TeleopIKTask(
        "teleop_left_arm",
        teleop_task.TeleopIKTaskConfig(
            joint_names=["left_arm/joint1", "left_arm/joint2"],
            model_path="fake.urdf",
            ee_joint_id=2,
            hand="left",
            max_joint_delta_deg=5.0,
            timeout=0.5,
        ),
    )


def test_configure_updates_live_params(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _make_task(monkeypatch)

    assert task.configure(max_joint_delta_deg=2.5, timeout=0.2)

    config = task.get_config()
    assert config["max_joint_delta_deg"] == 2.5
    assert config["timeout"] == 0.2


def test_configure_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _make_task(monkeypatch)

    with pytest.raises(ValueError, match="max_joint_delta_deg"):
        task.configure(max_joint_delta_deg=0.0)
    with pytest.raises(ValueError, match="timeout"):
        task.configure(timeout=-0.1)


def test_registry_exposes_live_config_commands() -> None:
    assert set(TASK_EXPOSES["teleop_ik"]) == {"configure", "get_config"}


def test_coordinator_task_invoke_reaches_configure(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _make_task(monkeypatch)
    coordinator = ControlCoordinator(publish_joint_state=False)
    try:
        coordinator.add_task(task, task_type="teleop_ik")

        assert (
            coordinator.task_invoke(
                "teleop_left_arm",
                "configure",
                {"max_joint_delta_deg": 3.0},
            )
            is True
        )

        assert coordinator.task_invoke("teleop_left_arm", "get_config", {})[
            "max_joint_delta_deg"
        ] == 3.0
    finally:
        coordinator.stop()
