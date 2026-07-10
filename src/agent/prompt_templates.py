"""
Prompt construction for the Generative Agent Layer.

Builds the structured prompt sent to the LLM at each planning step:
system context -> recent history + relevant long-term memory -> current
goal and required output schema. Keeping this as a standalone module
makes prompt iteration independent of the planning/validation logic.
"""

SYSTEM_CONTEXT = """You are a safety-conscious automotive planning assistant.
You translate a passenger's natural language request into one or more
structured sub-tasks for an autonomous vehicle. You do not control the
vehicle directly — every sub-task you produce will be checked by a
safety validator before execution.

Allowed sub-task types: navigation, detection, speed_adjustment, wait.
Prohibited: any action that violates traffic law, exceeds the posted
speed limit, or brings the vehicle within 3 metres of a pedestrian or
cyclist.

Respond only with a JSON object matching the required task schema.
"""

TASK_SCHEMA = {
    "type": "string (navigation | detection | speed_adjustment | wait)",
    "target": {"lat": "float", "lon": "float"},
    "speed_kmh": "float, optional",
    "constraints": "list[string], optional",
    "priority": "string (low | high)",
}


def build_prompt(goal: str, recent_steps: list, relevant_summaries: list) -> str:
    """
    Assemble the full prompt sent to the LLM for a single planning
    iteration, combining system context, short-term history, retrieved
    long-term memory, and the current goal.
    """
    history_block = _format_recent_steps(recent_steps)
    memory_block = _format_summaries(relevant_summaries)

    return (
        f"{SYSTEM_CONTEXT}\n"
        f"Task schema:\n{TASK_SCHEMA}\n\n"
        f"Recent interaction history:\n{history_block}\n\n"
        f"Relevant past context:\n{memory_block}\n\n"
        f"Current passenger goal: \"{goal}\"\n"
        f"Respond with a single JSON sub-task."
    )


def _format_recent_steps(steps: list) -> str:
    if not steps:
        return "(none yet)"
    return "\n".join(f"- {s}" for s in steps)


def _format_summaries(summaries: list) -> str:
    if not summaries:
        return "(no relevant long-term memory retrieved)"
    return "\n".join(f"- {s}" for s in summaries)
