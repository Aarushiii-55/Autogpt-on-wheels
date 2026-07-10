# AutoGPT on Wheels: Deploying Generative Agents in Autonomous Vehicles

A framework that lets passengers give autonomous vehicles natural-language instructions, safely, by inserting an LLM-powered reasoning layer between the passenger and the vehicle's deterministic control stack.

> Presented at **PEIS 2026** | Submitted to **Springer LNEE** (Scopus-indexed) | Related work at **ICAMSF 2025**

---

## The problem

Autonomous vehicles today follow fixed, deterministic plans. A passenger can change the destination — that's about it. There's no way to say *"stop at a pharmacy on the way"* or *"there's an emergency, reroute now"* and have the car reason about it safely.

## The approach

**AutoGPT on Wheels** inserts a generative reasoning layer between the passenger and the vehicle's control stack, decoupled so that LLM latency never touches the real-time control loop:

```
Passenger Instruction
        │
        ▼
┌─────────────────────────┐
│  Generative Agent Layer │  ← LLM-based reasoning, task decomposition
│  (~380ms ± 45ms)        │     hybrid memory (sliding window + FAISS)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Planning Interface     │  ← Dual-mode safety gate:
│  Layer                  │     rule-based + semantic validation
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  AV Control Stack        │  ← Deterministic, real-time
│  (20–50ms per cycle)     │     never blocked by LLM latency
└──────────────────────────┘
```

The safety gate is the core contribution: every LLM-generated action is checked against hard constraints (speed limits, pedestrian distance, GPS feasibility) **and** semantic consistency with the current environment before it's allowed to reach the vehicle.

## Results

Tested in **CARLA 0.9.10** (Town01 map) across 150 trials, 3 scenario types: natural language navigation, emergency replanning, and multi-task sequential intent fulfilment.

| Approach | Task Success Rate |
|---|---|
| Rule-based planner | 74.1% |
| Hybrid heuristic conversational planner | 85.7% |
| **AutoGPT on Wheels** | **92.3%** |

- **0% hallucination pass-through rate** — every unsafe/infeasible LLM action was caught by the safety gate across all 150 trials
- **380ms (±45ms)** average planning latency, fully decoupled from the 20–50ms real-time control loop
- Failure modes (all safely caught pre-execution): validation rejection (7.4%), replanning events (5.6%), fallback triggers (2.3%), hallucinated actions (1.8%)

> Full experimental setup, ablations, and failure mode analysis are documented in the dissertation. The full manuscript is not published in this repo, as the work is currently under review/publication with Springer LNEE — a link will be added here once it's publicly available.

## Architecture components

- **Generative Agent Layer** — LLM-based goal interpretation and task decomposition, with hybrid short/long-term memory
- **Planning Interface Layer** — dual-mode safety gate (rule-based + semantic validation)
- **AV Control Stack** — deterministic, CARLA-based vehicle control, untouched by LLM latency

## Repo structure

```
autogpt-on-wheels/
├── main.py                          # Orchestration: ties all layers together
├── config/
│   └── settings.py                  # Safety, memory, and CARLA config
├── src/
│   ├── agent/
│   │   └── prompt_templates.py      # LLM prompt construction
│   ├── planning_interface/
│   │   └── safety_gate.py           # Dual-mode validation logic
│   ├── memory/
│   │   └── hybrid_memory_manager.py # Sliding window + FAISS semantic memory
│   └── carla_integration/
│       └── environment.py           # CARLA world/sensor/vehicle wrapper
├── tests/
│   └── test_safety_gate.py          # Unit tests for the safety gate
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt

# Run the reference orchestration script (requires a running CARLA server)
python main.py

# Run unit tests (safety gate logic, no CARLA server needed)
pytest tests/
```

## Tech stack

`Python 3.10` · `CARLA 0.9.10` · `AutoGPT / GPT API` · `LangChain` · `FAISS` · `Sentence-Transformers` · `ROS Noetic` · `NumPy` · `OpenCV`

## Deployment strategy

A three-tier deployment model is proposed to make this feasible on real hardware under automotive compute/latency constraints: lightweight on-board inference → edge-server reasoning via V2X → full cloud-based reasoning, with graceful fallback as connectivity degrades.

## Limitations (honest assessment)

- Evaluated in simulation only (CARLA); real-world sim-to-real gap not yet measured
- Single map (Town01) — behavior on highways, multi-lane junctions, and complex pedestrian zones is untested
- Relies on a cloud-hosted LLM; production deployment needs stronger resilience to API unavailability

## Author

**Aarushi Sharma** — M.Sc. Data Science, Chandigarh University
Supervised by Dr. Harpreet Kaur, Department of Mathematics, University Institute of Sciences

## Citation

If you reference this work, please cite:

> Sharma, A. "AutoGPT on Wheels: Deploying Generative Agents in Autonomous Vehicles." Accepted at PEIS 2026; submitted for publication in Springer LNEE (Scopus-indexed).
