"""
CARLA Integration Layer

Thin wrapper around the CARLA Python API, covering the four
responsibilities described in the system architecture: environment
initialisation, sensor data collection, sub-task-to-control translation,
and simulation state monitoring.

Requires a running CARLA server (carla_simulator) and the `carla`
Python package matching your server version (0.9.10 used in
development/evaluation).
"""

from config.settings import CARLA_CONFIG


class CarlaEnvironment:
    def __init__(self, host="localhost", port=2000):
        self.host = host
        self.port = port
        self.world = None
        self.ego_vehicle = None
        self.sensors = {}

    def connect(self):
        """Connect to the CARLA server and load the configured town."""
        import carla
        client = carla.Client(self.host, self.port)
        client.set_timeout(10.0)
        self.world = client.load_world(CARLA_CONFIG["town"])
        return self.world

    def spawn_ego_vehicle(self):
        """Spawn the ego vehicle at a random valid spawn point."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.*")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        import random
        spawn_point = random.choice(spawn_points)
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        return self.ego_vehicle

    def attach_sensors(self):
        """Attach RGB camera, LiDAR, IMU, and GPS to the ego vehicle."""
        blueprint_library = self.world.get_blueprint_library()

        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CARLA_CONFIG["camera_resolution"][0]))
        cam_bp.set_attribute("image_size_y", str(CARLA_CONFIG["camera_resolution"][1]))
        self.sensors["camera"] = self._spawn_sensor(cam_bp)

        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(CARLA_CONFIG["lidar_channels"]))
        lidar_bp.set_attribute("range", str(CARLA_CONFIG["lidar_range_m"]))
        self.sensors["lidar"] = self._spawn_sensor(lidar_bp)

        self.sensors["imu"] = self._spawn_sensor(blueprint_library.find("sensor.other.imu"))
        self.sensors["gps"] = self._spawn_sensor(blueprint_library.find("sensor.other.gnss"))

        return self.sensors

    def _spawn_sensor(self, blueprint):
        import carla
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        return self.world.spawn_actor(blueprint, transform, attach_to=self.ego_vehicle)

    def apply_control(self, throttle: float, steer: float, brake: float):
        """Send a low-level control command to the ego vehicle."""
        import carla
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego_vehicle.apply_control(control)

    def get_snapshot(self):
        """Return the current world snapshot for state monitoring."""
        return self.world.get_snapshot()
