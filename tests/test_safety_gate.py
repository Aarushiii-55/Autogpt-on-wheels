"""
Basic unit tests for the DualModeSafetyGate.

Run with: pytest tests/test_safety_gate.py
"""

import pytest
from src.planning_interface.safety_gate import DualModeSafetyGate, ValidationResult


class FakeEnvState:
    """Minimal stand-in for CarlaEnvironment, used in unit tests."""

    def __init__(self, blocked_zones=None, pedestrians=None, current_task_type="idle"):
        self._blocked_zones = blocked_zones or []
        self._pedestrians = pedestrians or []
        self._current_task_type = current_task_type

    def get_blocked_zones(self):
        return self._blocked_zones

    def get_nearby_pedestrians(self):
        return self._pedestrians

    def get_current_task_type(self):
        return self._current_task_type


class FakePedestrian:
    def __init__(self, distance):
        self.distance = distance


@pytest.fixture
def gate():
    env = FakeEnvState()
    config = {"max_speed_kmh": 50, "min_pedestrian_dist_m": 3.0, "map_bounds": None}
    return DualModeSafetyGate(env_state=env, config=config)


def test_valid_navigation_subtask_passes(gate):
    subtask = {"type": "navigation", "target": {"lat": 12.0, "lon": 77.0}}
    result = gate.validate(subtask)
    assert result.passed is True


def test_speed_violation_rejected(gate):
    subtask = {"type": "speed_adjustment", "speed_kmh": 80, "constraints": ["speed_limit"]}
    result = gate.validate(subtask)
    assert result.passed is False
    assert result.rejection_code == "SPEED_VIOLATION"


def test_pedestrian_proximity_rejected():
    env = FakeEnvState(pedestrians=[FakePedestrian(distance=1.2)])
    config = {"max_speed_kmh": 50, "min_pedestrian_dist_m": 3.0, "map_bounds": None}
    gate = DualModeSafetyGate(env_state=env, config=config)

    subtask = {"type": "navigation", "target": {"lat": 12.0, "lon": 77.0}}
    result = gate.validate(subtask)
    assert result.passed is False
    assert result.rejection_code == "PEDESTRIAN_PROXIMITY"


def test_blocked_zone_rejected():
    env = FakeEnvState(blocked_zones=[{"lat": 12.0, "lon": 77.0}])
    config = {"max_speed_kmh": 50, "min_pedestrian_dist_m": 3.0, "map_bounds": None}
    gate = DualModeSafetyGate(env_state=env, config=config)

    subtask = {"type": "navigation", "target": {"lat": 12.0, "lon": 77.0}}
    result = gate.validate(subtask)
    assert result.passed is False
    assert result.rejection_code == "BLOCKED_ZONE"
