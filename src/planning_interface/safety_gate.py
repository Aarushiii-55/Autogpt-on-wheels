"""
Dual-Mode Safety Gate — Planning Interface Layer

Receives structured JSON sub-tasks from the AutoGPT reasoning layer and
applies rule-based + semantic validation before forwarding commands to
the Autonomous Vehicle control stack.

Achieved a 0% hallucination pass-through rate across 150 trials in the
CARLA Town01 simulation environment (see dissertation Ch. 6).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    passed: bool
    reason: Optional[str] = None
    rejection_code: Optional[str] = None


class DualModeSafetyGate:
    """
    Two-stage validator for LLM-generated sub-tasks:
      1. Rule-based check  — hard constraints (speed limits, GPS bounds,
         pedestrian proximity).
      2. Semantic check    — contextual consistency with current
         environment state (blocked zones, trajectory logic).
    """

    def __init__(self, env_state, config):
        self.env_state = env_state
        self.max_speed = config.get('max_speed_kmh', 50)
        self.min_pedestrian_dist = config.get('min_pedestrian_dist_m', 3.0)
        self.map_bounds = config.get('map_bounds', None)

    def validate(self, subtask: dict) -> ValidationResult:
        rule_result = self._rule_based_check(subtask)
        if not rule_result.passed:
            return rule_result
        return self._semantic_check(subtask)

    def _rule_based_check(self, subtask: dict) -> ValidationResult:
        if subtask.get('type') == 'navigation':
            target = subtask.get('target', {})
            if not self._is_gps_reachable(target):
                return ValidationResult(False, 'GPS location unreachable', 'GPS_INFEASIBLE')

        constraints = subtask.get('constraints', [])
        if 'speed_limit' in constraints:
            speed = subtask.get('speed_kmh', 0)
            if speed > self.max_speed:
                return ValidationResult(False, f'Speed {speed} exceeds limit', 'SPEED_VIOLATION')

        if self._pedestrian_proximity_violated():
            return ValidationResult(False, 'Pedestrian too close', 'PEDESTRIAN_PROXIMITY')

        return ValidationResult(True)

    def _semantic_check(self, subtask: dict) -> ValidationResult:
        blocked = self.env_state.get_blocked_zones()
        target = subtask.get('target', {})
        if target in blocked:
            return ValidationResult(False, 'Target in blocked zone', 'BLOCKED_ZONE')

        if not self._context_consistent(subtask):
            return ValidationResult(False, 'Inconsistent with env state', 'SEMANTIC_FAIL')

        return ValidationResult(True)

    def _is_gps_reachable(self, target) -> bool:
        if self.map_bounds is None:
            return True
        lat = target.get('lat', 0)
        lon = target.get('lon', 0)
        return (self.map_bounds['min_lat'] <= lat <= self.map_bounds['max_lat'] and
                self.map_bounds['min_lon'] <= lon <= self.map_bounds['max_lon'])

    def _pedestrian_proximity_violated(self) -> bool:
        pedestrians = self.env_state.get_nearby_pedestrians()
        return any(p.distance < self.min_pedestrian_dist for p in pedestrians)

    def _context_consistent(self, subtask: dict) -> bool:
        current_task_type = self.env_state.get_current_task_type()
        if current_task_type == 'stationary' and subtask.get('type') == 'navigation':
            return subtask.get('priority', 'low') == 'high'
        return True
