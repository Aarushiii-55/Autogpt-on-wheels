"""
Main orchestration loop — ties the Generative Agent Layer, Planning
Interface Layer, and CARLA-based AV Control Stack together.

This runs the high-level asynchronous planning cycle described in the
architecture: goal in -> LLM reasoning -> dual-mode safety gate ->
translated vehicle control -> repeat. The low-level control loop inside
CarlaEnvironment runs independently and is never blocked by planning
latency.

NOTE: This is a reference orchestration script for a CARLA-based
research/demo setup, not a production driving system.
"""

from config.settings import SAFETY_CONFIG, MEMORY_CONFIG
from src.memory.hybrid_memory_manager import HybridMemoryManager
from src.planning_interface.safety_gate import DualModeSafetyGate
from src.agent.prompt_templates import build_prompt
from src.carla_integration.environment import CarlaEnvironment


class AutoGPTOnWheels:
    def __init__(self, llm_client):
        self.env = CarlaEnvironment()
        self.memory = HybridMemoryManager(
            window_size=MEMORY_CONFIG["window_size"],
            embed_dim=MEMORY_CONFIG["embed_dim"],
        )
        self.llm_client = llm_client
        self.safety_gate = None  # initialised after CARLA connects

    def setup(self):
        self.env.connect()
        self.env.spawn_ego_vehicle()
        self.env.attach_sensors()
        self.safety_gate = DualModeSafetyGate(env_state=self.env, config=SAFETY_CONFIG)

    def handle_instruction(self, goal: str):
        """
        Run one planning cycle for a passenger instruction: build the
        prompt, query the LLM, validate the resulting sub-task, and
        (if valid) forward it to the control stack.
        """
        context = self.memory.get_context(goal)
        prompt = build_prompt(goal, context["recent_steps"], context["relevant_summaries"])

        subtask = self.llm_client.plan(prompt)
        result = self.safety_gate.validate(subtask)

        step_record = {
            "instruction": goal,
            "subtask": subtask,
            "status": "completed" if result.passed else "rejected",
            "rejection_reason": result.reason,
        }
        self.memory.add_step(step_record)

        if result.passed:
            self._execute(subtask)
        else:
            # Rejection reason is fed back into the next planning prompt
            # so the agent can generate a corrective alternative.
            print(f"Sub-task rejected: {result.reason} ({result.rejection_code})")

        return result

    def _execute(self, subtask: dict):
        """Translate a validated sub-task into a CARLA control command."""
        if subtask.get("type") == "speed_adjustment":
            target_speed = subtask.get("speed_kmh", 0)
            throttle = min(target_speed / SAFETY_CONFIG["max_speed_kmh"], 1.0)
            self.env.apply_control(throttle=throttle, steer=0.0, brake=0.0)
        # navigation / detection / wait sub-tasks would route to their
        # own handlers here (waypoint follower, perception query, etc.)


if __name__ == "__main__":
    # Example wiring — llm_client is a placeholder for whichever LLM
    # API client you configure (OpenAI, local model, etc.)
    class DummyLLMClient:
        def plan(self, prompt: str) -> dict:
            return {"type": "wait", "constraints": []}

    system = AutoGPTOnWheels(llm_client=DummyLLMClient())
    system.setup()
    system.handle_instruction("Take me to the nearest pharmacy")
