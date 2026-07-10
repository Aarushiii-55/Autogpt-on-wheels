"""
Central configuration for AutoGPT on Wheels.

Values here mirror the parameters used in the dissertation's CARLA
Town01 evaluation (Ch. 6), but are exposed as editable defaults so the
system can be re-tuned for a different map or vehicle class.
"""

SAFETY_CONFIG = {
    "max_speed_kmh": 50,
    "min_pedestrian_dist_m": 3.0,
    "map_bounds": None,  # set per-map at runtime, e.g. CARLA Town01 bounds
}

MEMORY_CONFIG = {
    "window_size": 10,          # short-term sliding window length
    "embed_dim": 384,           # matches all-MiniLM-L6-v2 output dim
    "summarise_every": 5,       # steps between long-term memory writes
    "retrieve_top_k": 3,        # relevant summaries pulled per query
}

PLANNING_CONFIG = {
    "target_planning_latency_ms": 380,
    "control_loop_hz_range": (20, 50),  # ms per cycle, deterministic loop
}

CARLA_CONFIG = {
    "town": "Town01",
    "camera_resolution": (1280, 720),
    "lidar_range_m": 100,
    "lidar_channels": 64,
}
