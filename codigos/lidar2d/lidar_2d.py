#!/usr/bin/env python3
"""
SIMULADOR DE ROBOT HUMANOIDE G1 CON LIDAR 2D, MAPEO Y CÁMARA OpenCV
=============================================================================
Controles:

    ↑/↓ : Velocidad adelante/atrás
    ←/→ : Girar izquierda/derecha
    Q/E : Velocidad lateral
    Z/X : Subir/bajar altura
    J/U : Torso yaw
    K/I : Torso pitch
    L/O : Torso roll
    T   : Toggle control de brazos aleatorio

    V   : Toggle visualización Sensor LIDAR
    M   : Toggle visualización MAPA
    C   : Limpiar mapa
    B   : Imprimir escaneo en consola

    P   : Toggle visualización cámara OpenCV (head_cam)

    ESC : Salir

=============================================================================
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import cv2


# =============================================================================
# KEYCODES PARA TECLAS ESPECIALES (GLFW)
# =============================================================================

KEY_ESCAPE = 256
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262


RUTA_POLITICA = "politicas/politica.pt"
RUTA_POLITICA_ADAPTADORA = "politicas/adapter.pt"
RUTA_PLITICA_ADAPTADORA_ESTADOS = "politicas/adapter_norm_stats.pt"
ESCENARIO_RUTA = "escenas/lydar_2d.xml"



# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qz)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)
    else:
        eulerVec[1] = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)

    return eulerVec


# =============================================================================
# CLASE PARA MANEJAR COMANDOS Y ESTADO DE VISUALIZACIÓN
# =============================================================================

class ViewerState:
    """Clase para manejar el estado de comandos y visualización."""
    def __init__(self):
        self.commands = np.zeros(8, dtype=np.float32)
        self.show_sensor = False
        self.show_map = False
        self.print_sensor_data = False
        self.clear_map = False
        self.show_cv_cam = False
        self.should_exit = False

    def print_status(self):
        print(
            f"vx: {self.commands[0]:<8.2f}"
            f"vy: {self.commands[2]:<8.2f}"
            f"yaw: {self.commands[1]:<8.2f}"
            f"height: {(0.75 + self.commands[3]):<8.2f}"
        )


def create_key_callback(state: ViewerState):
    """Crea el callback de teclado para el viewer nativo de MuJoCo."""
    def key_callback(keycode):
        # ===== TECLAS DE FLECHA PARA MOVIMIENTO =====
        if keycode == KEY_DOWN:  # Flecha abajo - atrás
            state.commands[0] -= 0.05
        elif keycode == KEY_UP:  # Flecha arriba - adelante
            state.commands[0] += 0.05
        elif keycode == KEY_LEFT:  # Flecha izquierda - girar izquierda
            state.commands[1] += 0.1
        elif keycode == KEY_RIGHT:  # Flecha derecha - girar derecha
            state.commands[1] -= 0.1
        # ===== MOVIMIENTO LATERAL =====
        elif keycode == ord('q') or keycode == ord('Q'):
            state.commands[2] += 0.05
        elif keycode == ord('e') or keycode == ord('E'):
            state.commands[2] -= 0.05
        # ===== ALTURA =====
        elif keycode == ord('z') or keycode == ord('Z'):
            state.commands[3] += 0.05
        elif keycode == ord('x') or keycode == ord('X'):
            state.commands[3] -= 0.05
        # ===== TORSO =====
        elif keycode == ord('j') or keycode == ord('J'):
            state.commands[4] += 0.1
        elif keycode == ord('u') or keycode == ord('U'):
            state.commands[4] -= 0.1
        elif keycode == ord('k') or keycode == ord('K'):
            state.commands[5] += 0.05
        elif keycode == ord('i') or keycode == ord('I'):
            state.commands[5] -= 0.05
        elif keycode == ord('l') or keycode == ord('L'):
            state.commands[6] += 0.05
        elif keycode == ord('o') or keycode == ord('O'):
            state.commands[6] -= 0.1
        # ===== BRAZOS ALEATORIOS =====
        elif keycode == ord('t') or keycode == ord('T'):
            state.commands[7] = not state.commands[7]
            print(f"🦾 Control de brazos aleatorio: {'ON' if state.commands[7] else 'OFF'}")
        # ===== VISUALIZACIÓN LIDAR =====
        elif keycode == ord('v') or keycode == ord('V'):
            state.show_sensor = not state.show_sensor
            print(f"📷 Visualización Sensor LIDAR: {'ON' if state.show_sensor else 'OFF'}")
        # ===== VISUALIZACIÓN MAPA =====
        elif keycode == ord('m') or keycode == ord('M'):
            state.show_map = not state.show_map
            print(f"🗺️  Visualización MAPA: {'ON' if state.show_map else 'OFF'}")
        # ===== LIMPIAR MAPA =====
        elif keycode == ord('c') or keycode == ord('C'):
            state.clear_map = True
            print("🧹 Limpiando mapa...")
        # ===== IMPRIMIR DATOS SENSOR =====
        elif keycode == ord('b') or keycode == ord('B'):
            state.print_sensor_data = True
        # ===== CÁMARA OPENCV =====
        elif keycode == ord('p') or keycode == ord('P'):
            state.show_cv_cam = not state.show_cv_cam
            print(f"🎥 Cámara OpenCV: {'ON' if state.show_cv_cam else 'OFF'}")
        # ===== SALIR =====
        elif keycode == KEY_ESCAPE:
            print("Cerrando simulación...")
            state.should_exit = True
            return
        else:
            return  # No imprimir status para teclas no reconocidas

        state.print_status()

    return key_callback


# =============================================================================
# LIDAR 2D BASADO EN RANGEFINDER DEL XML
# =============================================================================

class Lidar2DRangefinder:
    """
    LIDAR 2D 360° usando 32 sensores <rangefinder> definidos en el XML.
    """

    def __init__(self, model, data, prefix="lidar_", n_rays=32, max_range=10.0):
        self.model = model
        self.data = data
        self.prefix = prefix
        self.n_rays = n_rays
        self.max_range = max_range

        self.h_fov = 2 * np.pi

        self.site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{i}")
            for i in range(n_rays)
        ]
        self.sensor_names = [f"{prefix}{i}" for i in range(n_rays)]

        self.ranges = np.full(self.n_rays, self.max_range, dtype=np.float32)
        self.point_cloud = []

        self.visualization_enabled = False
        self.fig = None
        self.ax = None

        self.angles = np.linspace(0.0, 2.0 * np.pi, self.n_rays, endpoint=False)

        print(f"📷 LIDAR 2D inicializado: {n_rays} rayos, FOV 360°, rango {max_range} m")

    def scan(self):
        """Lee los 32 rangefinder y genera nube de puntos en el frame global."""
        self.point_cloud = []
        self.ranges.fill(self.max_range)

        for i in range(self.n_rays):
            sname = self.sensor_names[i]
            sid = self.site_ids[i]

            if sid < 0:
                continue

            dist = float(self.data.sensor(sname).data[0])
            if not np.isfinite(dist) or dist <= 0.0:
                continue

            self.ranges[i] = min(dist, self.max_range)
            if dist >= self.max_range:
                continue

            origin = self.data.site_xpos[sid].copy()
            R = self.data.site_xmat[sid].reshape(3, 3)
            dir_global = R[:, 2]

            end_point = origin + dist * dir_global
            self.point_cloud.append(end_point)

    def get_2d_points(self):
        if not self.point_cloud:
            return np.array([]).reshape(0, 2)
        pts = np.array(self.point_cloud)
        return pts[:, :2]

    @property
    def min_distance(self):
        return float(np.min(self.ranges)) if len(self.ranges) > 0 else self.max_range

    def print_data(self):
        print("\n" + "=" * 60)
        print("📷 LIDAR 2D (32 rayos)")
        print("=" * 60)
        print("  Rangos (m):")
        for i, d in enumerate(self.ranges):
            print(f"    r[{i:02d}] = {d:.2f}")
        print(f"\n  Distancia mínima: {self.min_distance:.2f} m")
        print(f"  Puntos en nube:   {len(self.point_cloud)}")
        print("=" * 60)

    def init_visualization(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        self.fig.canvas.manager.set_window_title('LIDAR 2D - Rangos')
        self.visualization_enabled = True

    def update_visualization(self):
        if not self.visualization_enabled or self.fig is None:
            return
        try:
            self.ax.clear()
            r_display = np.clip(self.ranges, 0, self.max_range)
            self.ax.scatter(self.angles, r_display, c='red', s=20, alpha=0.8)
            self.ax.set_rmax(self.max_range)
            self.ax.set_title(f'LIDAR 2D - min: {self.min_distance:.2f} m', fontsize=10)
            self.ax.grid(True)
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
        except Exception as e:
            print(f"Error en visualización LIDAR: {e}")

    def close_visualization(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.visualization_enabled = False


# =============================================================================
# CLASE MAPA 2D
# =============================================================================

class OccupancyMap:
    """Mapa de ocupación 2D construido con el LIDAR."""

    def __init__(self, resolution=0.05, size=30.0):
        self.resolution = resolution
        self.size = size
        self.grid_size = int(size / resolution)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        self.log_odds = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.l_occ = 1.0
        self.l_free = -0.3
        self.l_max = 10.0
        self.l_min = -10.0

        self.occ_threshold = 0.5
        self.free_threshold = -0.5

        self.robot_trajectory = []
        self.max_trajectory = 10000

        self.obstacle_points = []
        self.max_obstacle_points = 50000

        self.origin = size / 2.0

        self.fig = None
        self.ax = None
        self.visualization_enabled = False

        print(f"🗺️  Mapa: {self.grid_size}x{self.grid_size} celdas ({size}m x {size}m), {resolution}m/celda")

    def world_to_grid(self, x, y):
        gx = int((x + self.origin) / self.resolution)
        gy = int((y + self.origin) / self.resolution)
        return gx, gy

    def is_valid(self, gx, gy):
        return 0 <= gx < self.grid_size and 0 <= gy < self.grid_size

    def update_from_sensor(self, robot_x, robot_y, robot_yaw, sensor):
        if len(self.robot_trajectory) < self.max_trajectory:
            self.robot_trajectory.append((robot_x, robot_y, robot_yaw))

        robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)

        points_2d = sensor.get_2d_points()
        if len(points_2d) == 0:
            return

        for point in points_2d:
            px, py = point
            gx, gy = self.world_to_grid(px, py)

            if self.is_valid(gx, gy):
                if len(self.obstacle_points) < self.max_obstacle_points:
                    self.obstacle_points.append((px, py))

                self.log_odds[gx, gy] = np.clip(
                    self.log_odds[gx, gy] + self.l_occ,
                    self.l_min, self.l_max
                )

                if self.log_odds[gx, gy] > self.occ_threshold:
                    self.grid[gx, gy] = 2

                self._trace_ray_free(robot_gx, robot_gy, gx, gy)

        self._mark_fov_free(robot_x, robot_y, robot_yaw, sensor)

    def _mark_fov_free(self, robot_x, robot_y, robot_yaw, sensor):
        num_rays = 60
        angles = np.linspace(-sensor.h_fov / 2, sensor.h_fov / 2, num_rays) + robot_yaw

        for angle in angles:
            for dist in np.linspace(0.5, sensor.max_range * 0.8, 30):
                test_x = robot_x + dist * np.cos(angle)
                test_y = robot_y + dist * np.sin(angle)
                gx, gy = self.world_to_grid(test_x, test_y)

                if self.is_valid(gx, gy) and self.grid[gx, gy] != 2:
                    self.log_odds[gx, gy] = np.clip(
                        self.log_odds[gx, gy] + self.l_free * 0.5,
                        self.l_min, self.l_max
                    )
                    if self.log_odds[gx, gy] < self.free_threshold:
                        self.grid[gx, gy] = 1

    def _trace_ray_free(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        steps = 0
        max_steps = max(dx, dy) + 1

        while steps < max_steps:
            if (x, y) == (x1, y1):
                break

            if self.is_valid(x, y) and self.grid[x, y] != 2:
                self.log_odds[x, y] = np.clip(
                    self.log_odds[x, y] + self.l_free,
                    self.l_min, self.l_max
                )
                if self.log_odds[x, y] < self.free_threshold and self.grid[x, y] == 0:
                    self.grid[x, y] = 1

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

            steps += 1

    def clear(self):
        self.grid.fill(0)
        self.log_odds.fill(0)
        self.robot_trajectory.clear()
        self.obstacle_points.clear()
        print("🧹 Mapa limpiado")

    def get_stats(self):
        total = self.grid_size ** 2
        free = np.sum(self.grid == 1)
        occupied = np.sum(self.grid == 2)
        return {
            'explored_pct': (free + occupied) / total * 100,
            'free': free,
            'occupied': occupied,
            'trajectory': len(self.robot_trajectory),
            'obstacle_points': len(self.obstacle_points)
        }

    def init_visualization(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.canvas.manager.set_window_title('Mapa 2D - Exploración LIDAR')
        self.visualization_enabled = True

    def update_visualization(self, robot_x, robot_y, robot_yaw, sensor_fov=None):
        if not self.visualization_enabled or self.fig is None:
            return
        try:
            self.ax.clear()

            map_img = np.zeros((self.grid_size, self.grid_size, 3))
            map_img[self.grid == 0] = [0.7, 0.7, 0.7]
            map_img[self.grid == 1] = [1.0, 1.0, 1.0]
            map_img[self.grid == 2] = [0.1, 0.1, 0.1]

            extent = [-self.origin, self.origin, -self.origin, self.origin]
            self.ax.imshow(map_img.transpose(1, 0, 2), extent=extent, origin='lower')

            if self.obstacle_points:
                obs_array = np.array(self.obstacle_points)
                self.ax.scatter(obs_array[:, 0], obs_array[:, 1],
                                c='red', s=2, alpha=0.5)

            if len(self.robot_trajectory) > 1:
                traj = np.array(self.robot_trajectory)
                self.ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, alpha=0.6)

            robot_circle = Circle((robot_x, robot_y), 0.3, color='blue', alpha=0.9, zorder=10)
            self.ax.add_patch(robot_circle)

            if sensor_fov is not None:
                fov_deg = np.degrees(sensor_fov) / 2
                start_angle = np.degrees(robot_yaw) - fov_deg
                end_angle = np.degrees(robot_yaw) + fov_deg
                wedge = Wedge(
                    (robot_x, robot_y), 4.0,
                    start_angle, end_angle,
                    alpha=0.2, color='cyan', zorder=5
                )
                self.ax.add_patch(wedge)

            arrow_len = 0.8
            dx = arrow_len * np.cos(robot_yaw)
            dy = arrow_len * np.sin(robot_yaw)
            self.ax.arrow(robot_x, robot_y, dx, dy,
                          head_width=0.2, head_length=0.15,
                          fc='yellow', ec='black', linewidth=2, zorder=11)

            stats = self.get_stats()
            info = (
                f"📊 ESTADÍSTICAS\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"Explorado: {stats['explored_pct']:.1f}%\n"
                f"Libre: {stats['free']}\n"
                f"Ocupado: {stats['occupied']}\n"
                f"Obstáculos: {stats['obstacle_points']}\n"
                f"Trayectoria: {stats['trajectory']} pts"
            )
            self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                         fontsize=9, va='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

            view_range = 10
            self.ax.set_xlim(robot_x - view_range, robot_x + view_range)
            self.ax.set_ylim(robot_y - view_range, robot_y + view_range)
            self.ax.set_xlabel('X (m)', fontsize=10)
            self.ax.set_ylabel('Y (m)', fontsize=10)
            self.ax.set_title('🗺️ Mapa de Exploración - LIDAR 2D')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
        except Exception as e:
            print(f"Error en visualización del mapa: {e}")

    def close_visualization(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.visualization_enabled = False


# =============================================================================
# ENTORNO DE SIMULACIÓN
# =============================================================================

class HumanoidEnv:
    def __init__(self, policy_jit, robot_type="g1", device="cuda"):
        self.robot_type = robot_type
        self.device = device

        if robot_type == "g1":
            model_path = ESCENARIO_RUTA

            self.stiffness = np.array([
                150, 150, 150, 300, 80, 20,
                150, 150, 150, 300, 80, 20,
                400, 400, 400,
                80, 80, 40, 60,
                80, 80, 40, 60,
            ])
            self.damping = np.array([
                2, 2, 2, 4, 2, 1,
                2, 2, 2, 4, 2, 1,
                15, 15, 15,
                2, 2, 1, 1,
                2, 2, 1, 1,
            ])
            self.num_actions = 15
            self.num_dofs = 23
            self.default_dof_pos = np.array([
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0.2, 0.2, 0.0, 1.28,
                0.2, -0.2, 0.0, 1.28,
            ])
            self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
            self.arm_dof_lower_range = -0.4 * np.ones(8)
            self.arm_dof_upper_range = 0.4 * np.ones(8)
        else:
            raise ValueError(f"Robot type {robot_type} not supported!")

        self.sim_duration = 100 * 20.0
        self.sim_dt = 0.002
        self.sim_decimation = 10
        self.control_dt = self.sim_dt * self.sim_decimation

        print(f"📁 Cargando modelo: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)

        # Estado del viewer (comandos y flags)
        self.viewer_state = ViewerState()

        # Crear callback de teclado
        self.key_callback = create_key_callback(self.viewer_state)

        # Viewer nativo de MuJoCo (pasivo)
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback
        )

        # Configurar cámara
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.elevation = -25
        self.viewer.cam.azimuth = 180

        # Cámara OpenCV (head_cam del XML)
        self.cv_cam_name = "head_cam"
        try:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cv_cam_name)
            self.cv_cam_id = cam_id if cam_id >= 0 else -1
            print(f"🎥 Cámara OpenCV usando '{self.cv_cam_name}' (id={self.cv_cam_id})")
        except Exception as e:
            print(f"⚠️  Error buscando cámara '{self.cv_cam_name}': {e}")
            self.cv_cam_id = -1

        self.cv_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # LIDAR 2D y mapa
        self.rgbd_sensor = Lidar2DRangefinder(
            self.model, self.data,
            prefix="lidar_",
            n_rays=32,
            max_range=10.0
        )
        self.occupancy_map = OccupancyMap(resolution=0.05, size=30.0)
        self.sensor_update_freq = 3
        self.sensor_step_counter = 0

        # Estado robot / política
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.25
        self.arm_action = self.default_dof_pos[15:]
        self.prev_arm_action = self.default_dof_pos[15:]
        self.arm_blend = 0.0
        self.toggle_arm = False

        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05

        self.nj = 23
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8

        self.dof_pos = np.zeros(self.nj, dtype=np.float32)
        self.dof_vel = np.zeros(self.nj, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.nj)

        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3,))
        self.demo_obs_template[:self._n_demo_dof] = self.default_dof_pos[15:]
        self.demo_obs_template[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75

        self.target_yaw = 0.0
        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for _ in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))

        self.policy_jit = policy_jit

        # Adapter
        self.adapter = torch.jit.load(
            RUTA_POLITICA_ADAPTADORA,
            map_location=self.device
        )
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False

        norm_stats = torch.load(
            RUTA_PLITICA_ADAPTADORA_ESTADOS,
            weights_only=False
        )
        self.input_mean = torch.tensor(norm_stats['input_mean'], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=self.device, dtype=torch.float32)

        self.adapter_input = torch.zeros((1, 8 + 4), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

        self._print_instructions()

    def _print_instructions(self):
        print("\n" + "=" * 65)
        print("🤖 ROBOT G1 CON LIDAR 2D, MAPEO Y CÁMARA OpenCV")
        print("=" * 65)
        print("\n--- LOCOMOCIÓN ---")
        print("↑/↓      : Adelante / Atrás")
        print("←/→      : Girar izquierda / derecha")
        print("Q/E      : Movimiento lateral")
        print("Z/X      : Subir / Bajar altura")
        print("\n--- TORSO ---")
        print("J/U      : Yaw")
        print("K/I      : Pitch")
        print("L/O      : Roll")
        print("\n--- SENSOR Y MAPA ---")
        print("V        : Toggle visualización LIDAR")
        print("M        : Toggle visualización MAPA")
        print("B        : Imprimir datos del LIDAR")
        print("C        : Limpiar mapa")
        print("\n--- CÁMARA ---")
        print("P        : Toggle ventana OpenCV de la cámara (head_cam)")
        print("\n--- OTROS ---")
        print("T        : Toggle brazos aleatorios")
        print("ESC      : Salir")
        print("=" * 65, "\n")

    def extract_data(self):
        self.dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        self.dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        self.quat = self.data.sensor('orientation').data.astype(np.float32)
        self.ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)

    def get_observation(self):
        rpy = quatToEuler(self.quat)

        self.target_yaw = self.viewer_state.commands[1]
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand_flag:
            dyaw = 0.0

        obs_dof_vel = self.dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0

        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        self.adapter_input = np.concatenate([np.zeros(4), self.dof_pos[15:]])
        self.adapter_input[0] = 0.75 + self.viewer_state.commands[3]
        self.adapter_input[1] = self.viewer_state.commands[4]
        self.adapter_input[2] = self.viewer_state.commands[5]
        self.adapter_input[3] = self.viewer_state.commands[6]

        self.adapter_input = torch.tensor(self.adapter_input).to(
            self.device, dtype=torch.float32
        ).unsqueeze(0)

        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        obs_prop = np.concatenate([
            self.ang_vel * self.scales_ang_vel,
            rpy[:2],
            (np.sin(dyaw), np.cos(dyaw)),
            (self.dof_pos - self.default_dof_pos),
            self.dof_vel * self.scales_dof_vel,
            self.last_action,
            gait_obs,
            self.adapter_output.cpu().numpy().squeeze(),
        ])

        obs_priv = np.zeros((self.n_priv,))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = self.dof_pos[15:]
        obs_demo[self._n_demo_dof] = self.viewer_state.commands[0]
        obs_demo[self._n_demo_dof + 1] = self.viewer_state.commands[2]
        self._in_place_stand_flag = np.abs(self.viewer_state.commands[0]) < 0.1
        obs_demo[self._n_demo_dof + 3] = self.viewer_state.commands[4]
        obs_demo[self._n_demo_dof + 4] = self.viewer_state.commands[5]
        obs_demo[self._n_demo_dof + 5] = self.viewer_state.commands[6]
        obs_demo[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75 + self.viewer_state.commands[3]

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)

        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))

    def update_sensor_and_map(self):
        robot_pos = self.data.qpos[:3]
        rpy = quatToEuler(self.quat)
        robot_yaw = rpy[2]

        self.rgbd_sensor.scan()

        if self.viewer_state.clear_map:
            self.viewer_state.clear_map = False
            self.occupancy_map.clear()

        self.occupancy_map.update_from_sensor(
            robot_pos[0], robot_pos[1], robot_yaw,
            self.rgbd_sensor
        )

        if self.viewer_state.print_sensor_data:
            self.viewer_state.print_sensor_data = False
            self.rgbd_sensor.print_data()
            stats = self.occupancy_map.get_stats()
            print(f"\n🗺️  MAPA: Explorado {stats['explored_pct']:.1f}%, "
                  f"Libre: {stats['free']}, Ocupado: {stats['occupied']}, "
                  f"Puntos: {stats['obstacle_points']}")

        if self.viewer_state.show_sensor:
            if not self.rgbd_sensor.visualization_enabled:
                self.rgbd_sensor.init_visualization()
            self.rgbd_sensor.update_visualization()
        else:
            if self.rgbd_sensor.visualization_enabled:
                self.rgbd_sensor.close_visualization()

        if self.viewer_state.show_map:
            if not self.occupancy_map.visualization_enabled:
                self.occupancy_map.init_visualization()
            self.occupancy_map.update_visualization(
                robot_pos[0], robot_pos[1], robot_yaw,
                self.rgbd_sensor.h_fov
            )
        else:
            if self.occupancy_map.visualization_enabled:
                self.occupancy_map.close_visualization()

    def render_cv_camera(self):
        """Renderiza la cámara del XML y la muestra con OpenCV."""
        if not self.viewer_state.show_cv_cam:
            return
        if self.cv_cam_id < 0:
            return

        self.cv_renderer.update_scene(self.data, camera=self.cv_cam_id)
        img = self.cv_renderer.render()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)  # <-- ROTACIÓN 90°
        cv2.imshow(self.cv_cam_name, img_bgr)
        cv2.waitKey(1)

    def run(self):
        pd_target = self.default_dof_pos.copy()

        print("\n🚀 Simulación iniciada!")
        print("   Usa las FLECHAS para moverte")
        print("   Presiona 'V' para ver el LIDAR")
        print("   Presiona 'M' para ver el mapa")
        print("   Presiona 'P' para ver la cámara en OpenCV")
        print("   Presiona 'B' para ver estadísticas\n")

        try:
            i = 0
            while self.viewer.is_running() and not self.viewer_state.should_exit:
                self.extract_data()

                if i % self.sim_decimation == 0:
                    obs = self.get_observation()
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        extra_hist = torch.tensor(
                            np.array(self.extra_history_buf).flatten().copy(),
                            dtype=torch.float
                        ).view(1, -1).to(self.device)
                        raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()

                    raw_action = np.clip(raw_action, -40., 40.)
                    self.last_action = np.concatenate([
                        raw_action.copy(),
                        (self.dof_pos - self.default_dof_pos)[15:] / self.action_scale
                    ])
                    scaled_actions = raw_action * self.action_scale

                    if i % 300 == 0 and i > 0 and self.viewer_state.commands[7]:
                        self.arm_blend = 0
                        self.prev_arm_action = self.dof_pos[15:].copy()
                        self.arm_action = (
                            np.random.uniform(0, 1, 8) *
                            (self.arm_dof_upper_range - self.arm_dof_lower_range) +
                            self.arm_dof_lower_range
                        )
                        self.toggle_arm = True
                    elif not self.viewer_state.commands[7]:
                        if self.toggle_arm:
                            self.toggle_arm = False
                            self.arm_blend = 0
                            self.prev_arm_action = self.dof_pos[15:].copy()
                            self.arm_action = self.default_dof_pos[15:]

                    pd_target = np.concatenate([scaled_actions, np.zeros(8)]) + self.default_dof_pos
                    pd_target[15:] = (1 - self.arm_blend) * self.prev_arm_action + self.arm_blend * self.arm_action
                    self.arm_blend = min(1.0, self.arm_blend + 0.01)

                    self.gait_cycle = np.remainder(
                        self.gait_cycle + self.control_dt * self.gait_freq, 1.0
                    )
                    if self._in_place_stand_flag and (
                        (np.abs(self.gait_cycle[0] - 0.25) < 0.05) or
                        (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
                    ):
                        self.gait_cycle = np.array([0.25, 0.25])
                    if (not self._in_place_stand_flag) and (
                        (np.abs(self.gait_cycle[0] - 0.25) < 0.05) and
                        (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
                    ):
                        self.gait_cycle = np.array([0.25, 0.75])

                    self.sensor_step_counter += 1
                    if self.sensor_step_counter >= self.sensor_update_freq:
                        self.sensor_step_counter = 0
                        self.update_sensor_and_map()

                    # Actualizar cámara del viewer para seguir al robot
                    self.viewer.cam.lookat[:] = self.data.qpos[:3]

                    # Sincronizar viewer
                    self.viewer.sync()

                    # Cámara OpenCV
                    self.render_cv_camera()

                torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                self.data.ctrl = torque

                mujoco.mj_step(self.model, self.data)
                i += 1

        finally:
            self.rgbd_sensor.close_visualization()
            self.occupancy_map.close_visualization()
            self.viewer.close()
            cv2.destroyAllWindows()
            print("✅ Simulación finalizada correctamente")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":

    robot = "g1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Dispositivo: {device}")

    policy_pth = RUTA_POLITICA

    print("📦 Cargando política...")
    policy_jit = torch.jit.load(policy_pth, map_location=device)

    print("🤖 Inicializando entorno...")
    env = HumanoidEnv(policy_jit=policy_jit, robot_type=robot, device=device)

    print("▶️  Iniciando simulación...")
    env.run()