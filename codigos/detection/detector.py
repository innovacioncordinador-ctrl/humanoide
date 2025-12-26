#!/usr/bin/env python3
"""
=============================================================================
SIMULADOR DE ROBOT HUMANOIDE G1 CON SECUENCIA COMPLETA DE PICK & PLACE
+ CÁMARA OPENCV INTEGRADA + DETECCIÓN YOLO11
=============================================================================

Controles:
----------
- ↑/↓     : Adelante / Atrás
- ←/→     : Girar izquierda / derecha
- Q/E     : Movimiento lateral
- Z/X     : Subir / Bajar altura
- J/U     : Torso yaw
- K/I     : Torso pitch
- L       : Torso roll

- P       : Iniciar secuencia completa de pick & place
- R       : Resetear a posición inicial
- 1-8     : Poses individuales

- O       : Toggle cámara OpenCV (head_cam)
- Y       : Toggle detección YOLO11
- ESC     : Salir

=============================================================================
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
import torch
import math
import cv2
from ultralytics import YOLO  # <-- AÑADIDO PARA YOLO11


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
YOLO_RUTA = "detection/yolov9e_trained.pt"


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def quatToEuler(quat):
    """
    Convierte un cuaternión a ángulos de Euler (roll, pitch, yaw).
    """
    eulerVec = np.zeros(3)
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
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
# CLASE PARA MANEJAR COMANDOS Y ESTADO
# =============================================================================

class ViewerState:
    """Clase para manejar el estado de comandos y visualización."""
    def __init__(self):
        self.commands = np.zeros(8, dtype=np.float32)
        self.show_cv_cam = False
        self.start_grab_sequence = False
        self.reset_to_default_pose = False
        self.go_to_pose = -1
        self.should_exit = False
        self.yolo_active = False  # <-- AÑADIDO PARA YOLO11

    def print_status(self):
        print(
            f"vx: {self.commands[0]:<8.2f}"
            f"vy: {self.commands[2]:<8.2f}"
            f"yaw: {self.commands[1]:<8.2f}"
            f"altura: {(0.75 + self.commands[3]):<8.2f}"
        )


def create_key_callback(state: ViewerState):
    """Crea el callback de teclado para el viewer nativo de MuJoCo."""
    def key_callback(keycode):
        # ===== TECLAS DE FLECHA PARA MOVIMIENTO =====
        if keycode == KEY_UP:  # Flecha arriba - adelante
            state.commands[0] += 0.02
        elif keycode == KEY_DOWN:  # Flecha abajo - atrás
            state.commands[0] -= 0.02
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
        
        # ===== CÁMARA OPENCV =====
        elif keycode == ord('o') or keycode == ord('O'):
            state.show_cv_cam = not state.show_cv_cam
            print(f"🎥 Cámara OpenCV: {'ON' if state.show_cv_cam else 'OFF'}")
        
        # ===== YOLO DETECTION ===== (AÑADIDO)
        elif keycode == ord('y') or keycode == ord('Y'):
            state.yolo_active = not state.yolo_active
            print(f"🔍 YOLO Detection: {'ON' if state.yolo_active else 'OFF'}")
        
        # ===== SECUENCIA PICK & PLACE =====
        elif keycode == ord('p') or keycode == ord('P'):
            state.start_grab_sequence = True
            print("\n>>> INICIANDO SECUENCIA DE PICK & PLACE <<<")
        
        elif keycode == ord('r') or keycode == ord('R'):
            state.reset_to_default_pose = True
            print("\n>>> RESETEANDO A POSE POR DEFECTO <<<")
        
        # ===== POSES INDIVIDUALES =====
        elif keycode == ord('1'):
            state.go_to_pose = 0
        elif keycode == ord('2'):
            state.go_to_pose = 1
        elif keycode == ord('3'):
            state.go_to_pose = 2
        elif keycode == ord('4'):
            state.go_to_pose = 3
        elif keycode == ord('5'):
            state.go_to_pose = 4
        elif keycode == ord('6'):
            state.go_to_pose = 5
        elif keycode == ord('7'):
            state.go_to_pose = 6
        elif keycode == ord('8'):
            state.go_to_pose = 7
        
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
# CONTROLADOR DE SECUENCIA COMPLETA DE PICK & PLACE
# =============================================================================

class PickAndPlaceController:
    """
    Controlador avanzado para secuencia completa de pick & place.
    """
    
    def __init__(self, num_dofs, default_dof_pos):
        self.num_dofs = num_dofs
        self.default_dof_pos = default_dof_pos.copy()
        
        # MAPEO DE ARTICULACIONES
        self.joint_map = {
            'left_hip_pitch': 0, 'left_hip_roll': 1, 'left_hip_yaw': 2,
            'left_knee': 3, 'left_ankle_pitch': 4, 'left_ankle_roll': 5,
            'right_hip_pitch': 6, 'right_hip_roll': 7, 'right_hip_yaw': 8,
            'right_knee': 9, 'right_ankle_pitch': 10, 'right_ankle_roll': 11,
            'waist_yaw': 12, 'waist_roll': 13, 'waist_pitch': 14,
            'left_shoulder_pitch': 15, 'left_shoulder_roll': 16,
            'left_shoulder_yaw': 17, 'left_elbow': 18,
            'right_shoulder_pitch': 19, 'right_shoulder_roll': 20,
            'right_shoulder_yaw': 21, 'right_elbow': 22,
        }
        
        # POSES DE BRAZOS
        self.arm_poses = {
            'relajado': {
                'left_shoulder_pitch': 0.28,
                'left_shoulder_roll': 0.12,
                'left_shoulder_yaw': 0.0,
                'left_elbow': 0.98,
                'right_shoulder_pitch': 0.28,
                'right_shoulder_roll': -0.12,
                'right_shoulder_yaw': 0.0,
                'right_elbow': 0.98,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.0,
            },
            
            'preparar_agarre': {
                'left_shoulder_pitch': -0.3,
                'left_shoulder_roll': 0.4,
                'left_shoulder_yaw': 0.3,
                'left_elbow': 0.2,
                'right_shoulder_pitch': -0.3,
                'right_shoulder_roll': -0.4,
                'right_shoulder_yaw': -0.3,
                'right_elbow': 0.2,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.25,
            },
            
            'alcanzar': {
                'left_shoulder_pitch': -1.0,
                'left_shoulder_roll': 0.15,
                'left_shoulder_yaw': -0.35,
                'left_elbow': 0.6,
                'right_shoulder_pitch': -1.0,
                'right_shoulder_roll': -0.15,
                'right_shoulder_yaw': 0.35,
                'right_elbow': 0.6,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.40,
            },
            
            'agarrar': {
                'left_shoulder_pitch': -1.0,
                'left_shoulder_roll': 0.05,
                'left_shoulder_yaw': -0.50,
                'left_elbow': 0.7,
                'right_shoulder_pitch': -1.0,
                'right_shoulder_roll': -0.05,
                'right_shoulder_yaw': 0.50,
                'right_elbow': 0.7,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.40,
            },
            
            'levantar': {
                'left_shoulder_pitch': -0.5,
                'left_shoulder_roll': 0.08,
                'left_shoulder_yaw': -0.45,
                'left_elbow': 0.9,
                'right_shoulder_pitch': -0.5,
                'right_shoulder_roll': -0.08,
                'right_shoulder_yaw': 0.45,
                'right_elbow': 0.9,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.15,
            },
            
            'transportar': {
                'left_shoulder_pitch': -0.3,
                'left_shoulder_roll': 0.10,
                'left_shoulder_yaw': -0.40,
                'left_elbow': 1.0,
                'right_shoulder_pitch': -0.3,
                'right_shoulder_roll': -0.10,
                'right_shoulder_yaw': 0.40,
                'right_elbow': 1.0,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.05,
            },
            
            'bajar': {
                'left_shoulder_pitch': -0.8,
                'left_shoulder_roll': 0.08,
                'left_shoulder_yaw': -0.45,
                'left_elbow': 0.75,
                'right_shoulder_pitch': -0.8,
                'right_shoulder_roll': -0.08,
                'right_shoulder_yaw': 0.45,
                'right_elbow': 0.75,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.35,
            },
            
            'soltar': {
                'left_shoulder_pitch': -0.7,
                'left_shoulder_roll': 0.35,
                'left_shoulder_yaw': -0.20,
                'left_elbow': 0.5,
                'right_shoulder_pitch': -0.7,
                'right_shoulder_roll': -0.35,
                'right_shoulder_yaw': 0.20,
                'right_elbow': 0.5,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.30,
            },
            
            'retroceder': {
                'left_shoulder_pitch': -0.2,
                'left_shoulder_roll': 0.25,
                'left_shoulder_yaw': 0.0,
                'left_elbow': 0.6,
                'right_shoulder_pitch': -0.2,
                'right_shoulder_roll': -0.25,
                'right_shoulder_yaw': 0.0,
                'right_elbow': 0.6,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.10,
            },
        }
        
        self.pose_names = [
            'relajado', 'preparar_agarre', 'alcanzar', 'agarrar',
            'levantar', 'transportar', 'bajar', 'soltar', 'retroceder'
        ]
        
        # SECUENCIA COMPLETA
        self.sequence = [
            # === FASE 1: APROXIMACIÓN ===
            {'name': '1. Posición inicial', 'type': 'pose', 'pose': 'relajado', 'duration': 1.0, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '2. Caminar hacia objeto', 'type': 'walk', 'pose': 'relajado', 'duration': 5.0, 'locomotion': {'vx': 0.6, 'vy': 0, 'yaw_rate': 0}},
            {'name': '3. Detenerse frente al objeto', 'type': 'wait', 'pose': 'relajado', 'duration': 0.5, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            
            # === FASE 2: AGARRE ===
            {'name': '4. Preparar brazos para agarre', 'type': 'pose', 'pose': 'preparar_agarre', 'duration': 1.5, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '5. Alcanzar objeto', 'type': 'pose', 'pose': 'alcanzar', 'duration': 1.5, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '6. Cerrar brazos (agarrar)', 'type': 'pose', 'pose': 'agarrar', 'duration': 1.5, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '7. Mantener agarre', 'type': 'wait', 'pose': 'agarrar', 'duration': 1.0, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '8. Levantar objeto', 'type': 'pose', 'pose': 'levantar', 'duration': 2.0, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '9. Preparar para transporte', 'type': 'pose', 'pose': 'transportar', 'duration': 1.0, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            
            # === FASE 3: GIRAR 180° Y TRANSPORTAR ===
            {'name': '10. Girar 180° (parte 1/4)', 'type': 'turn', 'pose': 'transportar', 'duration': 1.5, 'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.4}},
            {'name': '11. Girar 180° (parte 2/4)', 'type': 'turn', 'pose': 'transportar', 'duration': 1.5, 'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.4}},
            {'name': '12. Girar 180° (parte 3/4)', 'type': 'turn', 'pose': 'transportar', 'duration': 1.5, 'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.6}},
            {'name': '13. Girar 180° (parte 4/4)', 'type': 'turn', 'pose': 'transportar', 'duration': 1.5, 'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.7}},
            {'name': '14. Estabilizar después de giro', 'type': 'wait', 'pose': 'transportar', 'duration': 1.0, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '15. Caminar al destino', 'type': 'walk', 'pose': 'transportar', 'duration': 7.0, 'locomotion': {'vx': 0.6, 'vy': 0, 'yaw_rate': 0}},
            {'name': '16. Detenerse en destino', 'type': 'wait', 'pose': 'transportar', 'duration': 0.5, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            
            # === FASE 4: ENTREGA ===
            {'name': '17. Bajar objeto', 'type': 'pose', 'pose': 'bajar', 'duration': 2.0, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '18. Soltar objeto', 'type': 'pose', 'pose': 'soltar', 'duration': 1.5, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '19. Confirmar soltar', 'type': 'wait', 'pose': 'soltar', 'duration': 0.5, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '20. Retroceder brazos', 'type': 'pose', 'pose': 'retroceder', 'duration': 1.0, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            
            # === FASE 5: RETORNO ===
            {'name': '21. Brazos relajados', 'type': 'pose', 'pose': 'relajado', 'duration': 1.0, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '22. Girar hacia origen (parte 1/4)', 'type': 'turn', 'pose': 'relajado', 'duration': 1.5, 'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.4}},
            {'name': '23. Girar hacia origen (parte 2/4)', 'type': 'turn', 'pose': 'relajado', 'duration': 1.5, 'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.4}},
            {'name': '24. Girar hacia origen (parte 3/4)', 'type': 'turn', 'pose': 'relajado', 'duration': 1.5, 'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.6}},
            {'name': '25. Girar hacia origen (parte 4/4)', 'type': 'turn', 'pose': 'relajado', 'duration': 1.5, 'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.7}},
            {'name': '26. Estabilizar', 'type': 'wait', 'pose': 'relajado', 'duration': 1.0, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
            {'name': '27. Caminar de regreso', 'type': 'walk', 'pose': 'relajado', 'duration': 3.0, 'locomotion': {'vx': 0.3, 'vy': 0, 'yaw_rate': 0}},
            {'name': '28. Posición final', 'type': 'pose', 'pose': 'relajado', 'duration': 1.0, 'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}},
        ]
        
        # ESTADO DEL CONTROLADOR
        self.active = False
        self.current_step = 0
        self.step_time = 0.0
        
        self.trajectory_start_pos = np.zeros(num_dofs)
        self.trajectory_target_pos = np.zeros(num_dofs)
        self.trajectory_T = 1.0
        self.trajectory_t = 0.0
        
        self.single_pose_active = False
        self.current_locomotion = {'vx': 0, 'vy': 0, 'yaw_rate': 0}
        self.accumulated_yaw = 0.0
        
    def _name_to_index(self, joint_name):
        return self.joint_map[joint_name]
    
    def _get_pose_target(self, pose_name):
        target = self.default_dof_pos.copy()
        pose = self.arm_poses[pose_name]
        for name, value in pose.items():
            idx = self._name_to_index(name)
            target[idx] = value
        return target
    
    def interpolate_cosine(self, q_init, q_target, t, T):
        if t >= T:
            return q_target
        ratio = (1 - math.cos(math.pi * (t / T))) / 2
        return q_init + (q_target - q_init) * ratio
    
    def start_sequence(self, current_pos):
        print("\n" + "=" * 60)
        print("🤖 INICIANDO SECUENCIA COMPLETA DE PICK & PLACE")
        print("=" * 60)
        
        self.active = True
        self.single_pose_active = False
        self.current_step = 0
        self.step_time = 0.0
        self.accumulated_yaw = 0.0
        
        self._start_step(current_pos)
    
    def start_single_pose(self, current_pos, pose_index, duration=2.0):
        if pose_index < 0 or pose_index >= len(self.pose_names):
            print(f"Pose {pose_index} no válida")
            return
        
        pose_name = self.pose_names[pose_index]
        print(f"\n>> Moviendo a pose: {pose_name}")
        
        self.single_pose_active = True
        self.active = False
        
        self.trajectory_start_pos = current_pos.copy()
        self.trajectory_target_pos = self._get_pose_target(pose_name)
        self.trajectory_T = duration
        self.trajectory_t = 0.0
        
        self.current_locomotion = {'vx': 0, 'vy': 0, 'yaw_rate': 0}
    
    def _start_step(self, current_pos):
        if self.current_step >= len(self.sequence):
            return
        
        step = self.sequence[self.current_step]
        
        print(f"\n>> {step['name']}")
        
        self.trajectory_start_pos = current_pos.copy()
        self.trajectory_target_pos = self._get_pose_target(step['pose'])
        self.trajectory_T = step['duration']
        self.trajectory_t = 0.0
        self.step_time = 0.0
        
        self.current_locomotion = step['locomotion'].copy()
    
    def reset_to_default(self, current_pos):
        print("\n>> Reseteando a pose relajada")
        self.accumulated_yaw = 0.0
        self.start_single_pose(current_pos, 0, duration=1.5)
    
    def stop_sequence(self):
        self.active = False
        self.single_pose_active = False
        self.current_locomotion = {'vx': 0, 'vy': 0, 'yaw_rate': 0}
        print("\n⏹️  Secuencia detenida")
    
    def update(self, current_pos, dt):
        if self.single_pose_active:
            self.trajectory_t += dt
            
            interpolated = np.zeros(self.num_dofs)
            for i in range(self.num_dofs):
                interpolated[i] = self.interpolate_cosine(
                    self.trajectory_start_pos[i],
                    self.trajectory_target_pos[i],
                    self.trajectory_t,
                    self.trajectory_T
                )
            
            if self.trajectory_t >= self.trajectory_T:
                self.single_pose_active = False
                print("   Pose alcanzada")
                return None, None
            
            loco_out = {
                'vx': self.current_locomotion['vx'],
                'vy': self.current_locomotion['vy'],
                'yaw': self.accumulated_yaw
            }
            return interpolated, loco_out
        
        if not self.active:
            return None, None
        
        self.trajectory_t += dt
        self.step_time += dt
        
        step = self.sequence[self.current_step]
        
        yaw_rate = self.current_locomotion.get('yaw_rate', 0)
        self.accumulated_yaw += yaw_rate * dt
        
        interpolated = np.zeros(self.num_dofs)
        for i in range(self.num_dofs):
            interpolated[i] = self.interpolate_cosine(
                self.trajectory_start_pos[i],
                self.trajectory_target_pos[i],
                self.trajectory_t,
                self.trajectory_T
            )
        
        if self.step_time >= step['duration']:
            self.current_step += 1
            
            if self.current_step >= len(self.sequence):
                self.active = False
                self.current_locomotion = {'vx': 0, 'vy': 0, 'yaw_rate': 0}
                print("\n" + "=" * 60)
                print("✅ SECUENCIA COMPLETADA EXITOSAMENTE")
                print("=" * 60 + "\n")
                return None, None
            
            self._start_step(self.trajectory_target_pos.copy())
        
        loco_out = {
            'vx': self.current_locomotion['vx'],
            'vy': self.current_locomotion['vy'],
            'yaw': self.accumulated_yaw
        }
        
        return interpolated, loco_out
    
    def is_active(self):
        return self.active or self.single_pose_active
    
    def get_locomotion_commands(self):
        return {
            'vx': self.current_locomotion['vx'],
            'vy': self.current_locomotion['vy'],
            'yaw': self.accumulated_yaw
        }


# =============================================================================
# ENTORNO DE SIMULACIÓN
# =============================================================================

class HumanoidEnv:
    """
    Entorno de simulación del robot humanoide G1 con capacidad de pick & place,
    cámara OpenCV integrada y detección YOLO11.
    """
    
    def __init__(self, policy_jit, robot_type="g1", device="cuda", yolo_model_path=None):
        self.robot_type = robot_type
        self.device = device

        if robot_type == "g1":
            model_path = "/home/iudc/unitree_robotic/unitree_mujoco/unitree_robots/g1/objetos.xml"
            
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
                0.5, 0.0, 0.2, 0.3,
                0.5, 0.0, -0.2, 0.3,
            ])
            
            self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
        else:
            raise ValueError(f"Robot '{robot_type}' no soportado")

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

        # Configurar cámara del viewer
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.elevation = -25
        self.viewer.cam.azimuth = 180

        # =====================================================================
        # CONFIGURACIÓN DE CÁMARA OPENCV (head_cam del XML)
        # =====================================================================
        self.cv_cam_name = "head_cam"
        try:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cv_cam_name)
            self.cv_cam_id = cam_id if cam_id >= 0 else -1
            print(f"🎥 Cámara OpenCV usando '{self.cv_cam_name}' (id={self.cv_cam_id})")
        except Exception as e:
            print(f"⚠️  Error buscando cámara '{self.cv_cam_name}': {e}")
            self.cv_cam_id = -1

        # Renderer para la cámara OpenCV
        self.cv_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # =====================================================================
        # CONFIGURACIÓN DE YOLO11
        # =====================================================================
        self.yolo_model_path = yolo_model_path
        self.yolo_conf_threshold = 0.5
        self.yolo_model = None
        self.last_detections = []  # Guardar últimas detecciones
        
        if self.yolo_model_path is not None:
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
                print(f"✅ Modelo YOLO11 cargado: {self.yolo_model_path}")
                print(f"   Clases: {self.yolo_model.names}")
            except Exception as e:
                print(f"⚠️  Error cargando modelo YOLO: {e}")
                self.yolo_model = None
        else:
            print("ℹ️  No se especificó modelo YOLO. Presiona 'Y' no tendrá efecto.")
        # =====================================================================

        self.action_scale = 0.25
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

        self.adapter_input = torch.zeros((1, 12), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

        # Controlador de Pick & Place
        self.pick_place_controller = PickAndPlaceController(
            self.num_dofs, 
            self.default_dof_pos
        )
        self.use_pick_place = False
        
        self._print_instructions()
    
    def _print_instructions(self):
        print("\n" + "=" * 65)
        print("🤖 ROBOT G1 - SIMULADOR CON PICK & PLACE + CÁMARA OPENCV + YOLO11")
        print("=" * 65)
        
        print("\n--- LOCOMOCIÓN ---")
        print("↑/↓      : Adelante / Atrás")
        print("←/→      : Girar izquierda / derecha")
        print("Q/E      : Movimiento lateral")
        print("Z/X      : Subir / Bajar altura")
        
        print("\n--- TORSO ---")
        print("J/U      : Yaw")
        print("K/I      : Pitch")
        print("L        : Roll")
        
        print("\n--- SECUENCIA DE PICK & PLACE ---")
        print("P        : Iniciar secuencia completa")
        print("R        : Resetear a pose relajada")
        print("1-8      : Poses individuales")
        
        print("\n--- CÁMARA Y VISIÓN ---")
        print("O        : Toggle ventana OpenCV (head_cam)")
        print("Y        : Toggle detección YOLO11")
        
        print("\n--- SISTEMA ---")
        print("ESC      : Salir")
        print("=" * 65 + "\n")

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
        
        if self._in_place_stand_flag and abs(self.viewer_state.commands[1]) < 0.01:
            dyaw = 0.0

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
        
        self._in_place_stand_flag = (np.abs(self.viewer_state.commands[0]) < 0.1 and 
                                      np.abs(self.viewer_state.commands[1]) < 0.1)
        
        obs_demo[self._n_demo_dof + 3] = self.viewer_state.commands[4]
        obs_demo[self._n_demo_dof + 4] = self.viewer_state.commands[5]
        obs_demo[self._n_demo_dof + 5] = self.viewer_state.commands[6]
        obs_demo[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75 + self.viewer_state.commands[3]

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)

        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))

    def _handle_commands(self):
        if self.viewer_state.start_grab_sequence:
            self.viewer_state.start_grab_sequence = False
            self.use_pick_place = True
            self.pick_place_controller.start_sequence(self.dof_pos)
        
        if self.viewer_state.reset_to_default_pose:
            self.viewer_state.reset_to_default_pose = False
            self.use_pick_place = True
            self.pick_place_controller.reset_to_default(self.dof_pos)
        
        if self.viewer_state.go_to_pose >= 0:
            pose_idx = self.viewer_state.go_to_pose
            self.viewer_state.go_to_pose = -1
            self.use_pick_place = True
            self.pick_place_controller.start_single_pose(self.dof_pos, pose_idx, duration=2.0)

    # =========================================================================
    # MÉTODO PARA RENDERIZAR CÁMARA OPENCV CON YOLO11
    # =========================================================================
    def render_cv_camera(self):
        """Renderiza la cámara del XML y opcionalmente ejecuta YOLO11."""
        if not self.viewer_state.show_cv_cam:
            return
        if self.cv_cam_id < 0:
            return

        # Actualizar escena con la cámara especificada
        self.cv_renderer.update_scene(self.data, camera=self.cv_cam_id)
        img = self.cv_renderer.render()
        
        # Convertir de RGB a BGR para OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Rotación de la imagen
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # =====================================================================
        # INFERENCIA YOLO11 (si está activo)
        # =====================================================================
        if self.viewer_state.yolo_active and self.yolo_model is not None:
            # Ejecutar inferencia
            results = self.yolo_model.predict(
                source=img_bgr,
                conf=self.yolo_conf_threshold,
                device=self.device,
                verbose=False  # No imprimir logs en cada frame
            )
            
            # Procesar y guardar detecciones
            self.last_detections = []
            if len(results) > 0:
                # Dibujar resultados en la imagen
                img_bgr = results[0].plot()
                
                # Guardar detecciones para uso externo
                for r in results:
                    if len(r.boxes) > 0:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            cls_name = self.yolo_model.names[cls_id]
                            xyxy = box.xyxy[0].cpu().numpy()
                            
                            detection = {
                                'class_id': cls_id,
                                'class_name': cls_name,
                                'confidence': conf,
                                'bbox': xyxy,  # [x1, y1, x2, y2]
                                'center': ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                            }
                            self.last_detections.append(detection)
        
        # =====================================================================
        # MOSTRAR INFORMACIÓN EN PANTALLA
        # =====================================================================
        # Estado YOLO
        if self.yolo_model is not None:
            status_text = "YOLO: ON" if self.viewer_state.yolo_active else "YOLO: OFF"
            color = (0, 255, 0) if self.viewer_state.yolo_active else (0, 0, 255)
        else:
            status_text = "YOLO: NO MODEL"
            color = (128, 128, 128)
        
        cv2.putText(img_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2)
        
        # Mostrar número de detecciones
        if self.viewer_state.yolo_active and len(self.last_detections) > 0:
            det_text = f"Detections: {len(self.last_detections)}"
            cv2.putText(img_bgr, det_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)
        
        cv2.imshow(self.cv_cam_name, img_bgr)
        cv2.waitKey(1)

    # =========================================================================
    # MÉTODO PARA OBTENER DETECCIONES (para uso externo)
    # =========================================================================
    def get_detections(self):
        """Retorna las últimas detecciones de YOLO."""
        return self.last_detections
    
    def get_detection_by_class(self, class_name):
        """Retorna detecciones filtradas por nombre de clase."""
        return [d for d in self.last_detections if d['class_name'] == class_name]

    def run(self):
        pd_target = self.default_dof_pos.copy()
        
        print("\n🚀 Simulación iniciada!")
        print("   Usa las FLECHAS para moverte")
        print("   Presiona 'P' para iniciar Pick & Place")
        print("   Presiona 'O' para ver la cámara en OpenCV")
        print("   Presiona 'Y' para activar detección YOLO11\n")

        try:
            i = 0
            while self.viewer.is_running() and not self.viewer_state.should_exit:
                self.extract_data()

                if i % self.sim_decimation == 0:
                    self._handle_commands()
                    
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

                    pd_target = np.concatenate([scaled_actions, np.zeros(8)]) + self.default_dof_pos
                    
                    # CONTROL DE PICK & PLACE
                    if self.use_pick_place:
                        arm_targets, loco_cmds = self.pick_place_controller.update(
                            self.dof_pos, 
                            self.control_dt
                        )
                        
                        if arm_targets is not None:
                            pd_target[12:] = arm_targets[12:]
                            
                            if loco_cmds is not None:
                                self.viewer_state.commands[0] = loco_cmds['vx']
                                self.viewer_state.commands[2] = loco_cmds['vy']
                                self.viewer_state.commands[1] = loco_cmds['yaw']
                        else:
                            if not self.pick_place_controller.is_active():
                                self.use_pick_place = False
                                self.viewer_state.commands[0] = 0
                                self.viewer_state.commands[1] = 0
                                self.viewer_state.commands[2] = 0

                    # Actualizar ciclo de marcha
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

                    # Actualizar cámara del viewer para seguir al robot
                    self.viewer.cam.lookat[:] = self.data.qpos[:3]

                    # Sincronizar viewer
                    self.viewer.sync()

                    # Renderizar cámara OpenCV con YOLO
                    self.render_cv_camera()

                torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                self.data.ctrl = torque

                mujoco.mj_step(self.model, self.data)
                i += 1

        finally:
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
    
    # =========================================================================
    # CONFIGURACIÓN DEL MODELO YOLO11 - CAMBIA ESTA RUTA
    # =========================================================================
    # Opción 1: Usar tu modelo entrenado
    yolo_model_path = YOLO_RUTA
    
    # Opción 2: Usar modelo preentrenado (para probar sin entrenar)
    # yolo_model_path = "yolo11l.pt"
    
    # Opción 3: Sin modelo YOLO (la tecla Y no hará nada)
    # yolo_model_path = None
    # =========================================================================
    
    print("📦 Cargando política...")
    policy_jit = torch.jit.load(policy_pth, map_location=device)

    print("🤖 Inicializando entorno...")
    env = HumanoidEnv(
        policy_jit=policy_jit, 
        robot_type=robot, 
        device=device,
        yolo_model_path=yolo_model_path  # <-- AÑADIDO
    )

    print("▶️  Iniciando simulación...")
    env.run()