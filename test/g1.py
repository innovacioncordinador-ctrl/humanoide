#!/usr/bin/env python3
"""
=============================================================================
SIMULADOR DE ROBOT HUMANOIDE G1 CON SECUENCIA COMPLETA DE PICK & PLACE
=============================================================================

Este programa simula un robot humanoide G1 de Unitree en el entorno MuJoCo.
El robot puede caminar, girar y realizar una secuencia completa de agarre
y colocación de objetos (pick & place).

Secuencia Completa:
-------------------
1. Aproximación: Caminar hacia el objeto
2. Agarre: Preparar, alcanzar, agarrar, levantar
3. Transporte: Girar 180°, caminar al destino
4. Entrega: Bajar, soltar, retroceder
5. Retorno: Girar, caminar de vuelta, pose final

Controles:
----------
- P: Iniciar secuencia completa de pick & place
- R: Resetear a posición inicial
- ESC: Salir

Dependencias:
-------------
- mujoco: Motor de física para simulación
- mujoco_viewer: Visualizador 3D
- torch: Para redes neuronales (política de control)
- numpy: Operaciones numéricas
- glfw: Manejo de ventanas y teclado

=============================================================================
"""

# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================

import types          # Para modificar métodos de objetos en tiempo de ejecución
import numpy as np    # Operaciones matemáticas y arrays
import mujoco         # Motor de física MuJoCo
import mujoco_viewer  # Visualizador para MuJoCo
import glfw           # Biblioteca para manejo de ventanas y eventos de teclado
from collections import deque  # Cola de doble extremo para historial de observaciones
import torch          # PyTorch para redes neuronales
import math           # Funciones matemáticas básicas


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def quatToEuler(quat):
    """
    Convierte un cuaternión a ángulos de Euler (roll, pitch, yaw).
    
    Los cuaterniones son una forma de representar rotaciones en 3D sin 
    problemas de gimbal lock. Esta función los convierte a ángulos más 
    intuitivos:
    - Roll (eulerVec[0]): Rotación alrededor del eje X (inclinación lateral)
    - Pitch (eulerVec[1]): Rotación alrededor del eje Y (inclinación frontal)
    - Yaw (eulerVec[2]): Rotación alrededor del eje Z (orientación/dirección)
    
    Parámetros:
    -----------
    quat : array de 4 elementos [qw, qx, qy, qz]
        Cuaternión que representa la orientación
    
    Retorna:
    --------
    eulerVec : array de 3 elementos [roll, pitch, yaw]
        Ángulos de Euler en radianes
    """
    # Inicializar vector de salida con ceros
    eulerVec = np.zeros(3)
    
    # Extraer componentes del cuaternión
    # qw es la parte escalar, (qx, qy, qz) es la parte vectorial
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    
    # --- Cálculo del ROLL (rotación en X) ---
    # Fórmula: atan2(2(qw*qx + qy*qz), 1 - 2(qx² + qy²))
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # --- Cálculo del PITCH (rotación en Y) ---
    # Fórmula: asin(2(qw*qy - qz*qx))
    # Nota: Se maneja el caso límite cuando sinp >= 1 (gimbal lock)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        # Usar copysign para evitar errores numéricos en arcsin
        eulerVec[1] = np.copysign(np.pi / 2, sinp)
    else:
        eulerVec[1] = np.arcsin(sinp)

    # --- Cálculo del YAW (rotación en Z) ---
    # Fórmula: atan2(2(qw*qz + qx*qy), 1 - 2(qy² + qz²))
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)

    return eulerVec


def _key_callback(self, window, key, scancode, action, mods):
    """
    Callback para manejar eventos de teclado en la ventana de simulación.
    
    Esta función se llama automáticamente cada vez que se presiona una tecla.
    Permite controlar el robot manualmente durante la simulación.
    
    Parámetros:
    -----------
    self : objeto viewer
        El visualizador de MuJoCo (contiene el estado de comandos)
    window : objeto GLFW
        La ventana donde ocurrió el evento
    key : int
        Código de la tecla presionada
    scancode : int
        Código de escaneo específico del sistema
    action : int
        Tipo de acción (PRESS, RELEASE, REPEAT)
    mods : int
        Modificadores (Shift, Ctrl, etc.)
    """
    # Solo procesar cuando se PRESIONA la tecla (no al soltar)
    if action != glfw.PRESS:
        return
    
    # =========================================================================
    # CONTROLES DE LOCOMOCIÓN (Movimiento del robot)
    # =========================================================================
    # self.commands es un array de 8 elementos que controla:
    # [0]: Velocidad hacia adelante/atrás (vx)
    # [1]: Velocidad de giro (yaw)
    # [2]: Velocidad lateral (vy)
    # [3]: Altura del cuerpo
    # [4-6]: Control del torso
    
    if key == glfw.KEY_W:
        # W: Aumentar velocidad hacia adelante
        self.commands[0] += 0.05
    elif key == glfw.KEY_S:
        # S: Aumentar velocidad hacia atrás (disminuir adelante)
        self.commands[0] -= 0.05
    elif key == glfw.KEY_A:
        # A: Girar a la izquierda (yaw positivo)
        self.commands[1] += 0.1
    elif key == glfw.KEY_D:
        # D: Girar a la derecha (yaw negativo)
        self.commands[1] -= 0.1
    elif key == glfw.KEY_Q:
        # Q: Movimiento lateral hacia la izquierda
        self.commands[2] += 0.05
    elif key == glfw.KEY_E:
        # E: Movimiento lateral hacia la derecha
        self.commands[2] -= 0.05
    elif key == glfw.KEY_Z:
        # Z: Subir altura del cuerpo
        self.commands[3] += 0.05
    elif key == glfw.KEY_X:
        # X: Bajar altura del cuerpo
        self.commands[3] -= 0.05
    
    # =========================================================================
    # CONTROLES DEL TORSO (Inclinación del cuerpo)
    # =========================================================================
    elif key == glfw.KEY_J:
        # J: Inclinar torso (yaw del torso)
        self.commands[4] += 0.1
    elif key == glfw.KEY_U:
        self.commands[4] -= 0.1
    elif key == glfw.KEY_K:
        # K: Inclinar torso (roll del torso)
        self.commands[5] += 0.05
    elif key == glfw.KEY_I:
        self.commands[5] -= 0.05
    elif key == glfw.KEY_L:
        # L: Inclinar torso (pitch del torso)
        self.commands[6] += 0.05
    elif key == glfw.KEY_O:
        self.commands[6] -= 0.1
    
    # =========================================================================
    # CONTROLES DE SECUENCIA DE PICK & PLACE
    # =========================================================================
    elif key == glfw.KEY_P:
        # P: Iniciar la secuencia completa de agarrar y colocar
        self.start_grab_sequence = True
        print("\n>>> INICIANDO SECUENCIA DE PICK & PLACE <<<")
    
    elif key == glfw.KEY_R:
        # R: Resetear el robot a su pose por defecto
        self.reset_to_default_pose = True
        print("\n>>> RESETEANDO A POSE POR DEFECTO <<<")
    
    # =========================================================================
    # POSES INDIVIDUALES (Teclas numéricas)
    # =========================================================================
    # Permiten probar poses específicas de los brazos
    elif key == glfw.KEY_1:
        self.go_to_pose = 0  # Pose: relajado
    elif key == glfw.KEY_2:
        self.go_to_pose = 1  # Pose: preparar_agarre
    elif key == glfw.KEY_3:
        self.go_to_pose = 2  # Pose: alcanzar
    elif key == glfw.KEY_4:
        self.go_to_pose = 3  # Pose: agarrar
    elif key == glfw.KEY_5:
        self.go_to_pose = 4  # Pose: levantar
    elif key == glfw.KEY_6:
        self.go_to_pose = 5  # Pose: transportar
    elif key == glfw.KEY_7:
        self.go_to_pose = 6  # Pose: bajar
    elif key == glfw.KEY_8:
        self.go_to_pose = 7  # Pose: soltar
    
    # =========================================================================
    # CONTROL DEL SISTEMA
    # =========================================================================
    elif key == glfw.KEY_ESCAPE:
        # ESC: Cerrar la simulación
        print("Cerrando simulación...")
        glfw.set_window_should_close(self.window, True)
        return
    
    # Imprimir el estado actual de los comandos para debugging
    print(
        f"vx: {self.commands[0]:<8.2f}"      # Velocidad adelante/atrás
        f"vy: {self.commands[2]:<8.2f}"      # Velocidad lateral
        f"yaw: {self.commands[1]:<8.2f}"     # Velocidad de giro
        f"altura: {(0.75 + self.commands[3]):<8.2f}"  # Altura total
    )


# =============================================================================
# CONTROLADOR DE SECUENCIA COMPLETA DE PICK & PLACE
# =============================================================================

class PickAndPlaceController:
    """
    Controlador avanzado para ejecutar una secuencia completa de pick & place.
    
    Este controlador gestiona:
    1. Las poses de los brazos (articulaciones de hombro, codo, muñeca)
    2. Los comandos de locomoción (caminar, girar)
    3. La interpolación suave entre poses
    4. El timing de cada fase de la secuencia
    
    La secuencia completa incluye:
    - Caminar hacia el objeto
    - Preparar brazos y agarrar
    - Levantar el objeto
    - Girar 180° y caminar al destino
    - Bajar y soltar el objeto
    - Regresar a la posición inicial
    
    Atributos principales:
    ----------------------
    num_dofs : int
        Número total de grados de libertad del robot (23 para G1)
    default_dof_pos : array
        Posiciones por defecto de todas las articulaciones
    joint_map : dict
        Mapeo de nombres de articulaciones a sus índices
    arm_poses : dict
        Diccionario con todas las poses predefinidas de brazos
    sequence : list
        Lista de pasos que componen la secuencia completa
    """
    
    def __init__(self, num_dofs, default_dof_pos):
        """
        Inicializa el controlador de pick & place.
        
        Parámetros:
        -----------
        num_dofs : int
            Número de grados de libertad del robot (23 para G1)
        default_dof_pos : array
            Posiciones por defecto de las articulaciones
        """
        self.num_dofs = num_dofs
        self.default_dof_pos = default_dof_pos.copy()
        
        # =====================================================================
        # MAPEO DE ARTICULACIONES
        # =====================================================================
        # Este diccionario mapea el nombre legible de cada articulación
        # a su índice en el array de posiciones del robot.
        # El robot G1 tiene 23 articulaciones en total:
        # - 6 por pierna (12 total para piernas)
        # - 3 para la cintura
        # - 4 por brazo (8 total para brazos)
        self.joint_map = {
            # --- PIERNA IZQUIERDA (índices 0-5) ---
            'left_hip_pitch': 0,    # Cadera: flexión/extensión
            'left_hip_roll': 1,     # Cadera: abducción/aducción
            'left_hip_yaw': 2,      # Cadera: rotación interna/externa
            'left_knee': 3,         # Rodilla: flexión
            'left_ankle_pitch': 4,  # Tobillo: flexión dorsal/plantar
            'left_ankle_roll': 5,   # Tobillo: inversión/eversión
            
            # --- PIERNA DERECHA (índices 6-11) ---
            'right_hip_pitch': 6,
            'right_hip_roll': 7,
            'right_hip_yaw': 8,
            'right_knee': 9,
            'right_ankle_pitch': 10,
            'right_ankle_roll': 11,
            
            # --- CINTURA/TORSO (índices 12-14) ---
            'waist_yaw': 12,    # Rotación de la cintura
            'waist_roll': 13,   # Inclinación lateral
            'waist_pitch': 14,  # Inclinación frontal
            
            # --- BRAZO IZQUIERDO (índices 15-18) ---
            'left_shoulder_pitch': 15,  # Hombro: elevación frontal
            'left_shoulder_roll': 16,   # Hombro: abducción lateral
            'left_shoulder_yaw': 17,    # Hombro: rotación
            'left_elbow': 18,           # Codo: flexión
            
            # --- BRAZO DERECHO (índices 19-22) ---
            'right_shoulder_pitch': 19,
            'right_shoulder_roll': 20,
            'right_shoulder_yaw': 21,
            'right_elbow': 22,
        }
        
        # =====================================================================
        # DEFINICIÓN DE POSES DE BRAZOS
        # =====================================================================
        # Cada pose define los ángulos (en radianes) para las articulaciones
        # de los brazos y la cintura. Estas poses se utilizan en diferentes
        # fases de la secuencia de pick & place.
        self.arm_poses = {
            # -----------------------------------------------------------------
            # POSE: RELAJADO
            # Brazos en posición natural, ligeramente doblados
            # Se usa como pose inicial/final y durante la caminata
            # -----------------------------------------------------------------
            'relajado': {
                'left_shoulder_pitch': 0.28,    # Brazos ligeramente hacia adelante
                'left_shoulder_roll': 0.12,     # Separados del cuerpo
                'left_shoulder_yaw': 0.0,       # Sin rotación
                'left_elbow': 0.98,             # Codos doblados
                'right_shoulder_pitch': 0.28,
                'right_shoulder_roll': -0.12,   # Simétrico (negativo)
                'right_shoulder_yaw': 0.0,
                'right_elbow': 0.98,
                'waist_yaw': 0.0,               # Cintura recta
                'waist_roll': 0.0,
                'waist_pitch': 0.0,
            },
            
            # -----------------------------------------------------------------
            # POSE: PREPARAR AGARRE
            # Brazos extendidos hacia adelante, listos para agarrar
            # Cintura ligeramente inclinada hacia adelante
            # -----------------------------------------------------------------
            'preparar_agarre': {
                'left_shoulder_pitch': -0.3,    # Brazos hacia adelante (negativo = arriba)
                'left_shoulder_roll': 0.4,      # Brazos separados
                'left_shoulder_yaw': 0.3,       # Rotación para orientar manos
                'left_elbow': 0.2,              # Codos casi estirados
                'right_shoulder_pitch': -0.3,
                'right_shoulder_roll': -0.4,
                'right_shoulder_yaw': -0.3,
                'right_elbow': 0.2,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.25,            # Inclinarse hacia adelante
            },
            
            # -----------------------------------------------------------------
            # POSE: ALCANZAR
            # Brazos extendidos hacia el objeto, cuerpo inclinado
            # -----------------------------------------------------------------
            'alcanzar': {
                'left_shoulder_pitch': -1.0,    # Brazos muy elevados
                'left_shoulder_roll': 0.15,     # Brazos más juntos
                'left_shoulder_yaw': -0.35,     # Manos orientadas hacia objeto
                'left_elbow': 0.6,              # Codos parcialmente flexionados
                'right_shoulder_pitch': -1.0,
                'right_shoulder_roll': -0.15,
                'right_shoulder_yaw': 0.35,
                'right_elbow': 0.6,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.40,            # Mayor inclinación
            },
            
            # -----------------------------------------------------------------
            # POSE: AGARRAR
            # Brazos cerrados alrededor del objeto
            # -----------------------------------------------------------------
            'agarrar': {
                'left_shoulder_pitch': -1.0,
                'left_shoulder_roll': 0.05,     # Brazos muy juntos (cerrando)
                'left_shoulder_yaw': -0.50,     # Rotación para envolver objeto
                'left_elbow': 0.7,              # Más flexión para sujetar
                'right_shoulder_pitch': -1.0,
                'right_shoulder_roll': -0.05,
                'right_shoulder_yaw': 0.50,
                'right_elbow': 0.7,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.40,
            },
            
            # -----------------------------------------------------------------
            # POSE: LEVANTAR
            # Brazos subiendo con el objeto agarrado
            # -----------------------------------------------------------------
            'levantar': {
                'left_shoulder_pitch': -0.5,    # Brazos subiendo
                'left_shoulder_roll': 0.08,
                'left_shoulder_yaw': -0.45,
                'left_elbow': 0.9,              # Codos más doblados
                'right_shoulder_pitch': -0.5,
                'right_shoulder_roll': -0.08,
                'right_shoulder_yaw': 0.45,
                'right_elbow': 0.9,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.15,            # Cuerpo más recto
            },
            
            # -----------------------------------------------------------------
            # POSE: TRANSPORTAR
            # Objeto sostenido cerca del pecho para caminar
            # -----------------------------------------------------------------
            'transportar': {
                'left_shoulder_pitch': -0.3,    # Brazos cerca del cuerpo
                'left_shoulder_roll': 0.10,
                'left_shoulder_yaw': -0.40,
                'left_elbow': 1.0,              # Codos muy flexionados
                'right_shoulder_pitch': -0.3,
                'right_shoulder_roll': -0.10,
                'right_shoulder_yaw': 0.40,
                'right_elbow': 1.0,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.05,            # Casi vertical
            },
            
            # -----------------------------------------------------------------
            # POSE: BAJAR
            # Brazos bajando para colocar el objeto
            # -----------------------------------------------------------------
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
                'waist_pitch': 0.35,            # Inclinarse para colocar
            },
            
            # -----------------------------------------------------------------
            # POSE: SOLTAR
            # Brazos abriéndose para liberar el objeto
            # -----------------------------------------------------------------
            'soltar': {
                'left_shoulder_pitch': -0.7,
                'left_shoulder_roll': 0.35,     # Brazos separándose
                'left_shoulder_yaw': -0.20,
                'left_elbow': 0.5,              # Codos más estirados
                'right_shoulder_pitch': -0.7,
                'right_shoulder_roll': -0.35,
                'right_shoulder_yaw': 0.20,
                'right_elbow': 0.5,
                'waist_yaw': 0.0,
                'waist_roll': 0.0,
                'waist_pitch': 0.30,
            },
            
            # -----------------------------------------------------------------
            # POSE: RETROCEDER
            # Brazos apartándose después de soltar
            # -----------------------------------------------------------------
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
        
        # Lista ordenada de nombres de poses (para acceso por índice)
        self.pose_names = [
            'relajado', 'preparar_agarre', 'alcanzar', 'agarrar',
            'levantar', 'transportar', 'bajar', 'soltar', 'retroceder'
        ]
        
        # =====================================================================
        # DEFINICIÓN DE LA SECUENCIA COMPLETA
        # =====================================================================
        # Cada paso de la secuencia es un diccionario con:
        # - 'name': Nombre descriptivo del paso (para logging)
        # - 'type': Tipo de acción ('pose', 'walk', 'turn', 'wait')
        # - 'pose': Pose de brazos a mantener durante este paso
        # - 'duration': Duración en segundos
        # - 'locomotion': Comandos de movimiento {vx, vy, yaw_rate}
        #   * vx: velocidad hacia adelante (m/s)
        #   * vy: velocidad lateral (m/s)
        #   * yaw_rate: velocidad angular (rad/s)
        self.sequence = [
            # =================================================================
            # FASE 1: APROXIMACIÓN AL OBJETO
            # =================================================================
            {
                'name': '1. Posición inicial',
                'type': 'pose',
                'pose': 'relajado',
                'duration': 1.0,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '2. Caminar hacia objeto',
                'type': 'walk',
                'pose': 'relajado',
                'duration': 5.0,  # 5 segundos caminando
                'locomotion': {'vx': 0.6, 'vy': 0, 'yaw_rate': 0}  # 0.6 m/s adelante
            },
            {
                'name': '3. Detenerse frente al objeto',
                'type': 'wait',
                'pose': 'relajado',
                'duration': 0.5,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            
            # =================================================================
            # FASE 2: AGARRE DEL OBJETO
            # =================================================================
            {
                'name': '4. Preparar brazos para agarre',
                'type': 'pose',
                'pose': 'preparar_agarre',
                'duration': 1.5,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '5. Alcanzar objeto',
                'type': 'pose',
                'pose': 'alcanzar',
                'duration': 1.5,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '6. Cerrar brazos (agarrar)',
                'type': 'pose',
                'pose': 'agarrar',
                'duration': 1.5,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '7. Mantener agarre',
                'type': 'wait',
                'pose': 'agarrar',
                'duration': 1.0,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '8. Levantar objeto',
                'type': 'pose',
                'pose': 'levantar',
                'duration': 2.0,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '9. Preparar para transporte',
                'type': 'pose',
                'pose': 'transportar',
                'duration': 1.0,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            
            # =================================================================
            # FASE 3: GIRAR 180° Y TRANSPORTAR AL DESTINO
            # El giro se divide en 4 partes para mejor control
            # =================================================================
            {
                'name': '10. Girar 180° (parte 1/4)',
                'type': 'turn',
                'pose': 'transportar',
                'duration': 1.5,
                # vx pequeño activa el patrón de caminata, yaw_rate gira
                'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.4}
            },
            {
                'name': '11. Girar 180° (parte 2/4)',
                'type': 'turn',
                'pose': 'transportar',
                'duration': 1.5,
                'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.4}
            },
            {
                'name': '12. Girar 180° (parte 3/4)',
                'type': 'turn',
                'pose': 'transportar',
                'duration': 1.5,
                'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.6}
            },
            {
                'name': '13. Girar 180° (parte 4/4)',
                'type': 'turn',
                'pose': 'transportar',
                'duration': 1.5,
                'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.7}
            },
            {
                'name': '14. Estabilizar después de giro',
                'type': 'wait',
                'pose': 'transportar',
                'duration': 1.0,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '15. Caminar al destino',
                'type': 'walk',
                'pose': 'transportar',
                'duration': 7.0,  # 7 segundos caminando con el objeto
                'locomotion': {'vx': 0.6, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '16. Detenerse en destino',
                'type': 'wait',
                'pose': 'transportar',
                'duration': 0.5,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            
            # =================================================================
            # FASE 4: ENTREGA DEL OBJETO
            # =================================================================
            {
                'name': '17. Bajar objeto',
                'type': 'pose',
                'pose': 'bajar',
                'duration': 2.0,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '18. Soltar objeto',
                'type': 'pose',
                'pose': 'soltar',
                'duration': 1.5,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '19. Confirmar soltar',
                'type': 'wait',
                'pose': 'soltar',
                'duration': 0.5,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '20. Retroceder brazos',
                'type': 'pose',
                'pose': 'retroceder',
                'duration': 1.0,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            
            # =================================================================
            # FASE 5: RETORNO A POSICIÓN INICIAL
            # =================================================================
            {
                'name': '21. Brazos relajados',
                'type': 'pose',
                'pose': 'relajado',
                'duration': 1.0,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            # Giro de regreso (180° dividido en 4 partes)
            {
                'name': '22. Girar hacia origen (parte 1/4)',
                'type': 'turn',
                'pose': 'relajado',
                'duration': 1.5,
                'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.4}
            },
            {
                'name': '23. Girar hacia origen (parte 2/4)',
                'type': 'turn',
                'pose': 'relajado',
                'duration': 1.5,
                'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.4}
            },
            {
                'name': '24. Girar hacia origen (parte 3/4)',
                'type': 'turn',
                'pose': 'relajado',
                'duration': 1.5,
                'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.6}
            },
            {
                'name': '25. Girar hacia origen (parte 4/4)',
                'type': 'turn',
                'pose': 'relajado',
                'duration': 1.5,
                'locomotion': {'vx': 0.02, 'vy': 0, 'yaw_rate': 0.7}
            },
            {
                'name': '26. Estabilizar',
                'type': 'wait',
                'pose': 'relajado',
                'duration': 1.0,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
            {
                'name': '27. Caminar de regreso',
                'type': 'walk',
                'pose': 'relajado',
                'duration': 3.0,
                'locomotion': {'vx': 0.3, 'vy': 0, 'yaw_rate': 0}  # Más lento al regresar
            },
            {
                'name': '28. Posición final',
                'type': 'pose',
                'pose': 'relajado',
                'duration': 1.0,
                'locomotion': {'vx': 0, 'vy': 0, 'yaw_rate': 0}
            },
        ]
        
        # =====================================================================
        # VARIABLES DE ESTADO DEL CONTROLADOR
        # =====================================================================
        
        # Indica si la secuencia está activa
        self.active = False
        
        # Índice del paso actual en la secuencia
        self.current_step = 0
        
        # Tiempo transcurrido en el paso actual
        self.step_time = 0.0
        
        # Variables para interpolación de trayectorias
        self.trajectory_start_pos = np.zeros(num_dofs)   # Posición inicial
        self.trajectory_target_pos = np.zeros(num_dofs)  # Posición objetivo
        self.trajectory_T = 1.0  # Duración total de la transición
        self.trajectory_t = 0.0  # Tiempo transcurrido en la transición
        
        # Indica si estamos ejecutando una pose individual (no secuencia)
        self.single_pose_active = False
        
        # Comandos de locomoción actuales
        self.current_locomotion = {'vx': 0, 'vy': 0, 'yaw_rate': 0}
        
        # Acumulador de yaw para mantener la orientación durante giros
        # Los giros se realizan integrando yaw_rate, no como ángulo absoluto
        self.accumulated_yaw = 0.0
        
    def _name_to_index(self, joint_name):
        """
        Convierte el nombre de una articulación a su índice numérico.
        
        Parámetros:
        -----------
        joint_name : str
            Nombre de la articulación (ej: 'left_shoulder_pitch')
        
        Retorna:
        --------
        int : Índice de la articulación en el array de posiciones
        """
        return self.joint_map[joint_name]
    
    def _get_pose_target(self, pose_name):
        """
        Genera un array completo de posiciones objetivo para una pose dada.
        
        Combina las posiciones por defecto de las piernas con las posiciones
        específicas de brazos y torso definidas en la pose.
        
        Parámetros:
        -----------
        pose_name : str
            Nombre de la pose (ej: 'agarrar', 'transportar')
        
        Retorna:
        --------
        target : array
            Array con las 23 posiciones objetivo de todas las articulaciones
        """
        # Empezar con las posiciones por defecto (importante para piernas)
        target = self.default_dof_pos.copy()
        
        # Sobrescribir solo las articulaciones definidas en la pose
        pose = self.arm_poses[pose_name]
        for name, value in pose.items():
            idx = self._name_to_index(name)
            target[idx] = value
            
        return target
    
    def interpolate_cosine(self, q_init, q_target, t, T):
        """
        Interpola suavemente entre dos valores usando una función coseno.
        
        La interpolación con coseno produce un movimiento más natural que
        la interpolación lineal, con aceleración/desaceleración suaves
        al inicio y final del movimiento.
        
        Fórmula: ratio = (1 - cos(π * t/T)) / 2
        - En t=0: ratio=0, retorna q_init
        - En t=T: ratio=1, retorna q_target
        - Derivada es 0 en ambos extremos (suave)
        
        Parámetros:
        -----------
        q_init : float
            Valor inicial
        q_target : float
            Valor objetivo
        t : float
            Tiempo actual
        T : float
            Tiempo total de la transición
        
        Retorna:
        --------
        float : Valor interpolado
        """
        if t >= T:
            return q_target
        # Función coseno normalizada para interpolación suave
        ratio = (1 - math.cos(math.pi * (t / T))) / 2
        return q_init + (q_target - q_init) * ratio
    
    def start_sequence(self, current_pos):
        """
        Inicia la secuencia completa de pick & place.
        
        Parámetros:
        -----------
        current_pos : array
            Posiciones actuales de todas las articulaciones
        """
        print("\n" + "=" * 60)
        print(" INICIANDO SECUENCIA COMPLETA DE PICK & PLACE")
        print("=" * 60)
        
        # Activar el modo de secuencia completa
        self.active = True
        self.single_pose_active = False
        self.current_step = 0
        self.step_time = 0.0
        self.accumulated_yaw = 0.0  # Resetear orientación acumulada
        
        # Configurar el primer paso de la secuencia
        self._start_step(current_pos)
    
    def start_single_pose(self, current_pos, pose_index, duration=2.0):
        """
        Mueve el robot a una pose individual (fuera de la secuencia).
        
        Útil para probar poses específicas con las teclas 1-8.
        
        Parámetros:
        -----------
        current_pos : array
            Posiciones actuales de las articulaciones
        pose_index : int
            Índice de la pose (0-8)
        duration : float
            Duración de la transición en segundos
        """
        if pose_index < 0 or pose_index >= len(self.pose_names):
            print(f"Pose {pose_index} no válida")
            return
        
        pose_name = self.pose_names[pose_index]
        print(f"\n>> Moviendo a pose: {pose_name}")
        
        # Activar modo pose individual
        self.single_pose_active = True
        self.active = False  # Desactivar secuencia si estaba activa
        
        # Configurar la trayectoria
        self.trajectory_start_pos = current_pos.copy()
        self.trajectory_target_pos = self._get_pose_target(pose_name)
        self.trajectory_T = duration
        self.trajectory_t = 0.0
        
        # No hay movimiento de locomoción durante poses individuales
        self.current_locomotion = {'vx': 0, 'vy': 0, 'yaw_rate': 0}
    
    def _start_step(self, current_pos):
        """
        Configura el inicio de un nuevo paso en la secuencia.
        
        Parámetros:
        -----------
        current_pos : array
            Posiciones actuales de las articulaciones
        """
        if self.current_step >= len(self.sequence):
            return
        
        step = self.sequence[self.current_step]
        
        # Mostrar en consola el paso actual
        print(f"\n>> {step['name']}")
        
        # Configurar la trayectoria de interpolación
        self.trajectory_start_pos = current_pos.copy()
        self.trajectory_target_pos = self._get_pose_target(step['pose'])
        self.trajectory_T = step['duration']
        self.trajectory_t = 0.0
        self.step_time = 0.0
        
        # Copiar los comandos de locomoción para este paso
        self.current_locomotion = step['locomotion'].copy()
    
    def reset_to_default(self, current_pos):
        """
        Resetea el robot a la pose relajada (por defecto).
        
        Parámetros:
        -----------
        current_pos : array
            Posiciones actuales de las articulaciones
        """
        print("\n>> Reseteando a pose relajada")
        self.accumulated_yaw = 0.0  # Resetear orientación
        # Usar pose índice 0 (relajado)
        self.start_single_pose(current_pos, 0, duration=1.5)
    
    def stop_sequence(self):
        """
        Detiene la secuencia actual inmediatamente.
        """
        self.active = False
        self.single_pose_active = False
        self.current_locomotion = {'vx': 0, 'vy': 0, 'yaw_rate': 0}
        print("\n  Secuencia detenida")
    
    def update(self, current_pos, dt):
        """
        Actualiza el estado del controlador y calcula las posiciones objetivo.
        
        Este método debe llamarse en cada ciclo de control. Calcula las
        posiciones interpoladas de los brazos y los comandos de locomoción
        actuales.
        
        Parámetros:
        -----------
        current_pos : array
            Posiciones actuales de las articulaciones
        dt : float
            Tiempo transcurrido desde la última actualización (en segundos)
        
        Retorna:
        --------
        tuple : (arm_targets, locomotion_commands)
            - arm_targets: Array con posiciones objetivo, o None si terminó
            - locomotion_commands: Dict con vx, vy, yaw, o None si terminó
        """
        # =================================================================
        # MANEJO DE POSE INDIVIDUAL
        # =================================================================
        if self.single_pose_active:
            self.trajectory_t += dt
            
            # Interpolar cada articulación individualmente
            interpolated = np.zeros(self.num_dofs)
            for i in range(self.num_dofs):
                interpolated[i] = self.interpolate_cosine(
                    self.trajectory_start_pos[i],
                    self.trajectory_target_pos[i],
                    self.trajectory_t,
                    self.trajectory_T
                )
            
            # Verificar si se completó la transición
            if self.trajectory_t >= self.trajectory_T:
                self.single_pose_active = False
                print("   Pose alcanzada")
                return None, None
            
            # Preparar salida de locomoción
            loco_out = {
                'vx': self.current_locomotion['vx'],
                'vy': self.current_locomotion['vy'],
                'yaw': self.accumulated_yaw
            }
            return interpolated, loco_out
        
        # =================================================================
        # MANEJO DE SECUENCIA COMPLETA
        # =================================================================
        if not self.active:
            return None, None
        
        # Avanzar los contadores de tiempo
        self.trajectory_t += dt
        self.step_time += dt
        
        step = self.sequence[self.current_step]
        
        # Acumular yaw cuando hay velocidad angular (yaw_rate)
        # Esto convierte velocidad angular a ángulo absoluto
        yaw_rate = self.current_locomotion.get('yaw_rate', 0)
        self.accumulated_yaw += yaw_rate * dt
        
        # Calcular posiciones interpoladas para todas las articulaciones
        interpolated = np.zeros(self.num_dofs)
        for i in range(self.num_dofs):
            interpolated[i] = self.interpolate_cosine(
                self.trajectory_start_pos[i],
                self.trajectory_target_pos[i],
                self.trajectory_t,
                self.trajectory_T
            )
        
        # =================================================================
        # VERIFICAR FIN DEL PASO ACTUAL
        # =================================================================
        if self.step_time >= step['duration']:
            self.current_step += 1
            
            # ¿Se completó toda la secuencia?
            if self.current_step >= len(self.sequence):
                self.active = False
                self.current_locomotion = {'vx': 0, 'vy': 0, 'yaw_rate': 0}
                print("\n" + "=" * 60)
                print(" SECUENCIA COMPLETADA EXITOSAMENTE")
                print("=" * 60 + "\n")
                return None, None
            
            # Iniciar el siguiente paso
            self._start_step(self.trajectory_target_pos.copy())
        
        # Preparar salida de locomoción con yaw acumulado
        loco_out = {
            'vx': self.current_locomotion['vx'],
            'vy': self.current_locomotion['vy'],
            'yaw': self.accumulated_yaw  # Ángulo acumulado, no velocidad
        }
        
        return interpolated, loco_out
    
    def is_active(self):
        """
        Verifica si hay alguna secuencia o pose activa.
        
        Retorna:
        --------
        bool : True si el controlador está ejecutando algo
        """
        return self.active or self.single_pose_active
    
    def get_locomotion_commands(self):
        """
        Obtiene los comandos de locomoción actuales.
        
        Retorna:
        --------
        dict : Diccionario con vx, vy, yaw
        """
        return {
            'vx': self.current_locomotion['vx'],
            'vy': self.current_locomotion['vy'],
            'yaw': self.accumulated_yaw
        }


# =============================================================================
# ENTORNO DE SIMULACIÓN PRINCIPAL
# =============================================================================

class HumanoidEnv:
    """
    Entorno de simulación del robot humanoide G1 con capacidad de pick & place.
    
    Esta clase integra:
    - El motor de física MuJoCo para simular el robot
    - Una política de red neuronal para controlar las piernas (locomoción)
    - El controlador de pick & place para los brazos
    - El visualizador 3D interactivo
    
    El robot G1 de Unitree es un humanoide con:
    - 2 piernas de 6 DOF cada una (12 DOF total)
    - 1 torso de 3 DOF
    - 2 brazos de 4 DOF cada uno (8 DOF total)
    - Total: 23 DOF
    
    La política neuronal controla las 15 primeras articulaciones
    (piernas + torso), mientras que el controlador de pick & place
    gestiona los brazos (8 articulaciones).
    
    Atributos principales:
    ----------------------
    model : MjModel
        Modelo de MuJoCo cargado del XML
    data : MjData
        Estado actual de la simulación
    viewer : MujocoViewer
        Visualizador 3D
    policy_jit : torch.jit.ScriptModule
        Política de control de locomoción
    pick_place_controller : PickAndPlaceController
        Controlador de brazos
    """
    
    def __init__(self, policy_jit, robot_type="g1", device="cuda"):
        """
        Inicializa el entorno de simulación.
        
        Parámetros:
        -----------
        policy_jit : torch.jit.ScriptModule
            Política de control preentrenada (red neuronal)
        robot_type : str
            Tipo de robot ('g1' por defecto)
        device : str
            Dispositivo para PyTorch ('cuda' o 'cpu')
        """
        self.robot_type = robot_type
        self.device = device

        # =====================================================================
        # CONFIGURACIÓN ESPECÍFICA DEL ROBOT G1
        # =====================================================================
        if robot_type == "g1":
            # Ruta al archivo XML del modelo (debe ajustarse según instalación)
            model_path = "/home/iudc/unitree_robotic/unitree_mujoco/unitree_robots/g1/interaccion.xml"
            
            # -----------------------------------------------------------------
            # GANANCIAS PD PARA CADA ARTICULACIÓN
            # -----------------------------------------------------------------
            # stiffness (Kp): Qué tan fuerte el motor intenta alcanzar la posición objetivo
            # Valores más altos = movimientos más rígidos pero más precisos
            # Valores más bajos = movimientos más suaves pero menos precisos
            self.stiffness = np.array([
                # Pierna izquierda (cadera tiene más rigidez que tobillo)
                150, 150, 150, 300, 80, 20,
                # Pierna derecha
                150, 150, 150, 300, 80, 20,
                # Cintura (alta rigidez para estabilidad)
                400, 400, 400,
                # Brazos (menor rigidez para movimientos más suaves)
                80, 80, 40, 60,
                80, 80, 40, 60,
            ])
            
            # damping (Kd): Amortiguamiento, reduce oscilaciones
            # Valores más altos = menos vibración pero movimientos más lentos
            self.damping = np.array([
                2, 2, 2, 4, 2, 1,
                2, 2, 2, 4, 2, 1,
                15, 15, 15,
                2, 2, 1, 1,
                2, 2, 1, 1,
            ])
            
            # Número de acciones que genera la política neuronal
            # (15 primeras articulaciones: piernas + torso)
            self.num_actions = 15
            
            # Número total de grados de libertad del robot
            self.num_dofs = 23
            
            # -----------------------------------------------------------------
            # POSICIONES POR DEFECTO DE LAS ARTICULACIONES
            # -----------------------------------------------------------------
            # Estas son las posiciones "naturales" del robot de pie
            self.default_dof_pos = np.array([
                # Pierna izquierda: ligeramente flexionada
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                # Pierna derecha: simétrica
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                # Cintura: recta
                0.0, 0.0, 0.0,
                # Brazos: relajados a los lados
                0.5, 0.0, 0.2, 0.3,   # Izquierdo
                0.5, 0.0, -0.2, 0.3,  # Derecho
            ])
            
            # -----------------------------------------------------------------
            # LÍMITES DE TORQUE
            # -----------------------------------------------------------------
            # Torque máximo que puede ejercer cada motor (en Nm)
            # Evita que el robot aplique fuerzas irreales
            self.torque_limits = np.array([
                # Piernas: motores más potentes en cadera y rodilla
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                # Cintura
                88, 50, 50,
                # Brazos: motores más pequeños
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
        else:
            raise ValueError(f"Robot '{robot_type}' no soportado")

        # =====================================================================
        # PARÁMETROS DE SIMULACIÓN
        # =====================================================================
        
        # Duración total de la simulación (segundos)
        self.sim_duration = 100 * 20.0  # 2000 segundos
        
        # Paso de tiempo de la simulación física (más pequeño = más preciso)
        self.sim_dt = 0.002  # 2 ms = 500 Hz
        
        # Decimación: cada cuántos pasos de física se ejecuta el control
        self.sim_decimation = 10
        
        # Período de control (20 ms = 50 Hz)
        self.control_dt = self.sim_dt * self.sim_decimation

        # =====================================================================
        # INICIALIZACIÓN DE MUJOCO
        # =====================================================================
        
        # Cargar el modelo del robot desde el archivo XML
        self.model = mujoco.MjModel.from_xml_path(model_path)
        
        # Configurar el paso de tiempo en el modelo
        self.model.opt.timestep = self.sim_dt
        
        # Crear el estado de la simulación
        self.data = mujoco.MjData(self.model)
        
        # Resetear a la pose inicial definida en el XML (keyframe 0)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Ejecutar un paso inicial para estabilizar
        mujoco.mj_step(self.model, self.data)
        
        # =====================================================================
        # CONFIGURACIÓN DEL VISUALIZADOR
        # =====================================================================
        
        # Crear ventana de visualización 3D
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        # Array de comandos del usuario [vx, yaw, vy, altura, torso_yaw, torso_roll, torso_pitch, ...]
        self.viewer.commands = np.zeros(8, dtype=np.float32)
        
        # Configurar cámara inicial
        self.viewer.cam.distance = 2.5   # Distancia al robot
        self.viewer.cam.elevation = 0.0  # Ángulo de elevación
        
        # Flags para controlar secuencias desde el teclado
        self.viewer.start_grab_sequence = False
        self.viewer.reset_to_default_pose = False
        self.viewer.go_to_pose = -1  # -1 = ninguna pose solicitada
        
        # Inyectar el callback de teclado personalizado
        self.viewer._key_callback = types.MethodType(_key_callback, self.viewer)
        glfw.set_key_callback(self.viewer.window, self.viewer._key_callback)

        # =====================================================================
        # PARÁMETROS DE LA POLÍTICA
        # =====================================================================
        
        # Factor de escala para las acciones de la red neuronal
        self.action_scale = 0.25
        
        # Escalas para normalizar observaciones
        self.scales_ang_vel = 0.25   # Velocidad angular
        self.scales_dof_vel = 0.05   # Velocidad de articulaciones

        # =====================================================================
        # DIMENSIONES DE OBSERVACIÓN
        # =====================================================================
        
        self.nj = 23  # Número de articulaciones
        self.n_priv = 3  # Información privilegiada (solo entrenamiento)
        
        # Dimensión de observaciones propioceptivas:
        # 3 (velocidad angular) + 2 (roll, pitch) + 2 (sin/cos yaw) +
        # 23*3 (pos, vel, last_action) + 2 (gait) + 15 (adapter output)
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        
        self.history_len = 10       # Longitud del historial de observaciones
        self.extra_history_len = 25  # Historial extra para la política
        self._n_demo_dof = 8        # DOFs de demostración (brazos)

        # =====================================================================
        # BUFFERS DE ESTADO
        # =====================================================================
        
        # Estado actual del robot
        self.dof_pos = np.zeros(self.nj, dtype=np.float32)  # Posiciones
        self.dof_vel = np.zeros(self.nj, dtype=np.float32)  # Velocidades
        self.quat = np.zeros(4, dtype=np.float32)           # Orientación
        self.ang_vel = np.zeros(3, dtype=np.float32)        # Velocidad angular
        self.last_action = np.zeros(self.nj)                # Última acción

        # Template para observaciones de demostración
        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3,))
        self.demo_obs_template[:self._n_demo_dof] = self.default_dof_pos[15:]
        self.demo_obs_template[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75

        # Variables de control de orientación
        self.target_yaw = 0.0
        self._in_place_stand_flag = True  # True = robot quieto
        
        # Parámetros del ciclo de marcha (para coordinar pasos)
        self.gait_cycle = np.array([0.25, 0.25])  # Fase de cada pierna
        self.gait_freq = 1.3  # Frecuencia de paso (Hz)

        # =====================================================================
        # HISTORIAL DE OBSERVACIONES
        # =====================================================================
        # La política usa observaciones pasadas para estimar velocidad, etc.
        
        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        
        # Inicializar con ceros
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for _ in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))

        # =====================================================================
        # REDES NEURONALES
        # =====================================================================
        
        # Política principal (control de locomoción)
        self.policy_jit = policy_jit

        # Adaptador: traduce comandos de alto nivel a objetivos articulares
        self.adapter = torch.jit.load(
            "/home/iudc/unitree_robotic/politicas/adapter.pt",
            map_location=self.device
        )
        self.adapter.eval()  # Modo evaluación (no entrenamiento)
        for param in self.adapter.parameters():
            param.requires_grad = False  # Congelar pesos

        # Estadísticas de normalización para el adaptador
        norm_stats = torch.load(
            "/home/iudc/unitree_robotic/politicas/adapter_norm_stats.pt",
            weights_only=False
        )
        self.input_mean = torch.tensor(norm_stats['input_mean'], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=self.device, dtype=torch.float32)

        # Buffers para entrada/salida del adaptador
        self.adapter_input = torch.zeros((1, 12), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

        # =====================================================================
        # CONTROLADOR DE PICK & PLACE
        # =====================================================================
        
        self.pick_place_controller = PickAndPlaceController(
            self.num_dofs, 
            self.default_dof_pos
        )
        self.use_pick_place = False  # Se activa con tecla P
        
        # Mostrar instrucciones al usuario
        self._print_instructions()
    
    def _print_instructions(self):
        """
        Imprime las instrucciones de control en la consola.
        """
        print("\n" + "=" * 60)
        print("🤖 ROBOT G1 - SIMULADOR CON PICK & PLACE COMPLETO")
        print("=" * 60)
        
        print("\n--- LOCOMOCIÓN ---")
        print("W/S     : Adelante / Atrás")
        print("A/D     : Girar izquierda / derecha")
        print("Q/E     : Movimiento lateral")
        print("Z/X     : Subir / Bajar altura")
        
        print("\n--- SECUENCIA DE PICK & PLACE ---")
        print("P       : Iniciar secuencia completa")
        print("R       : Resetear a pose relajada")
        print("1-8     : Poses individuales")
        
        print("\n--- SISTEMA ---")
        print("ESC     : Salir")
        print("=" * 60 + "\n")

    def extract_data(self):
        """
        Extrae el estado actual del robot desde MuJoCo.
        
        Lee las posiciones, velocidades y sensores del modelo
        y los almacena en los buffers de la clase.
        """
        # Extraer posiciones de articulaciones (últimos num_dofs valores de qpos)
        # qpos incluye la pose base (7 valores: pos xyz + quat) + articulaciones
        self.dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        
        # Extraer velocidades de articulaciones
        self.dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        
        # Leer sensores
        # 'orientation': IMU que mide la orientación del torso (cuaternión)
        self.quat = self.data.sensor('orientation').data.astype(np.float32)
        
        # 'angular-velocity': Giroscopio que mide velocidad angular
        self.ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)

    def get_observation(self):
        """
        Construye el vector de observación para la política neuronal.
        
        La observación incluye:
        - Estado propioceptivo: orientación, velocidades, posiciones
        - Comandos del usuario: velocidad deseada, altura, etc.
        - Historial de observaciones pasadas
        
        Retorna:
        --------
        obs : array
            Vector de observación completo
        """
        # Convertir cuaternión a ángulos de Euler
        rpy = quatToEuler(self.quat)

        # Calcular error de orientación (diferencia entre actual y deseado)
        self.target_yaw = self.viewer.commands[1]
        dyaw = rpy[2] - self.target_yaw
        
        # Normalizar el error al rango [-π, π]
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        
        # Si el robot está quieto y no hay comando de giro, ignorar error de yaw
        # Esto evita correcciones innecesarias cuando está parado
        if self._in_place_stand_flag and abs(self.viewer.commands[1]) < 0.01:
            dyaw = 0.0

        # Observación del ciclo de marcha (para sincronizar pasos)
        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        # -----------------------------------------------------------------
        # PROCESAMIENTO CON EL ADAPTADOR
        # -----------------------------------------------------------------
        # El adaptador traduce comandos de alto nivel (altura, inclinación)
        # a objetivos para las articulaciones
        
        self.adapter_input = np.concatenate([np.zeros(4), self.dof_pos[15:]])
        self.adapter_input[0] = 0.75 + self.viewer.commands[3]  # Altura
        self.adapter_input[1] = self.viewer.commands[4]         # Torso yaw
        self.adapter_input[2] = self.viewer.commands[5]         # Torso roll
        self.adapter_input[3] = self.viewer.commands[6]         # Torso pitch

        # Convertir a tensor y normalizar
        self.adapter_input = torch.tensor(self.adapter_input).to(
            self.device, dtype=torch.float32
        ).unsqueeze(0)
        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        
        # Ejecutar el adaptador
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        
        # Desnormalizar la salida
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        # -----------------------------------------------------------------
        # CONSTRUIR OBSERVACIÓN PROPIOCEPTIVA
        # -----------------------------------------------------------------
        obs_prop = np.concatenate([
            # Velocidad angular escalada
            self.ang_vel * self.scales_ang_vel,
            # Orientación (roll, pitch) - yaw se maneja aparte
            rpy[:2],
            # Error de yaw como sin/cos (evita discontinuidad en ±π)
            (np.sin(dyaw), np.cos(dyaw)),
            # Error de posición respecto a pose por defecto
            (self.dof_pos - self.default_dof_pos),
            # Velocidades de articulaciones escaladas
            self.dof_vel * self.scales_dof_vel,
            # Última acción (para suavidad temporal)
            self.last_action,
            # Fase del ciclo de marcha
            gait_obs,
            # Salida del adaptador
            self.adapter_output.cpu().numpy().squeeze(),
        ])

        # Información privilegiada (no disponible en robot real)
        obs_priv = np.zeros((self.n_priv,))
        
        # Historial aplanado
        obs_hist = np.array(self.proprio_history_buf).flatten()

        # Observación de demostración
        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = self.dof_pos[15:]       # Posición brazos
        obs_demo[self._n_demo_dof] = self.viewer.commands[0]  # vx comando
        obs_demo[self._n_demo_dof + 1] = self.viewer.commands[2]  # vy comando
        
        # Determinar si el robot debe estar quieto
        # (sin comandos de velocidad lineal ni angular)
        self._in_place_stand_flag = (np.abs(self.viewer.commands[0]) < 0.1 and 
                                      np.abs(self.viewer.commands[1]) < 0.1)
        
        obs_demo[self._n_demo_dof + 3] = self.viewer.commands[4]  # Torso
        obs_demo[self._n_demo_dof + 4] = self.viewer.commands[5]
        obs_demo[self._n_demo_dof + 5] = self.viewer.commands[6]
        obs_demo[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75 + self.viewer.commands[3]

        # Actualizar historiales
        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)

        # Concatenar todo en un solo vector
        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))

    def _handle_commands(self):
        """
        Procesa los comandos de teclado pendientes.
        
        Lee los flags establecidos por el callback de teclado
        y ejecuta las acciones correspondientes.
        """
        # Verificar si se solicitó iniciar la secuencia de pick & place
        if self.viewer.start_grab_sequence:
            self.viewer.start_grab_sequence = False
            self.use_pick_place = True
            self.pick_place_controller.start_sequence(self.dof_pos)
        
        # Verificar si se solicitó resetear a pose por defecto
        if self.viewer.reset_to_default_pose:
            self.viewer.reset_to_default_pose = False
            self.use_pick_place = True
            self.pick_place_controller.reset_to_default(self.dof_pos)
        
        # Verificar si se solicitó una pose individual
        if self.viewer.go_to_pose >= 0:
            pose_idx = self.viewer.go_to_pose
            self.viewer.go_to_pose = -1  # Resetear flag
            self.use_pick_place = True
            self.pick_place_controller.start_single_pose(self.dof_pos, pose_idx, duration=2.0)

    def run(self):
        """
        Bucle principal de simulación.
        
        Este método ejecuta la simulación hasta que el usuario
        cierre la ventana o se alcance el tiempo máximo.
        
        El bucle hace:
        1. Extraer estado del robot
        2. Cada N pasos (decimación): ejecutar política y control
        3. Calcular torques con control PD
        4. Ejecutar paso de física
        5. Renderizar visualización
        """
        # Posición objetivo inicial (pose por defecto)
        pd_target = self.default_dof_pos.copy()
        
        # Calcular número total de pasos de simulación
        total_steps = int(self.sim_duration / self.sim_dt)
        
        for i in range(total_steps):
            # =================================================================
            # PASO 1: EXTRAER ESTADO ACTUAL DEL ROBOT
            # =================================================================
            self.extract_data()

            # =================================================================
            # PASO 2: CONTROL (cada sim_decimation pasos)
            # =================================================================
            # El control se ejecuta a 50 Hz, la física a 500 Hz
            if i % self.sim_decimation == 0:
                
                # Procesar comandos de teclado pendientes
                self._handle_commands()
                
                # Obtener observación para la política
                obs = self.get_observation()
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

                # ---------------------------------------------------------
                # EJECUTAR LA POLÍTICA NEURONAL
                # ---------------------------------------------------------
                with torch.no_grad():  # Sin cálculo de gradientes
                    # Preparar historial extra
                    extra_hist = torch.tensor(
                        np.array(self.extra_history_buf).flatten().copy(),
                        dtype=torch.float
                    ).view(1, -1).to(self.device)
                    
                    # Obtener acción de la política
                    raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()

                # Clipear acciones para evitar valores extremos
                raw_action = np.clip(raw_action, -40., 40.)
                
                # Guardar acción completa (incluyendo brazos)
                self.last_action = np.concatenate([
                    raw_action.copy(),
                    (self.dof_pos - self.default_dof_pos)[15:] / self.action_scale
                ])
                
                # Escalar acciones
                scaled_actions = raw_action * self.action_scale

                # Calcular posiciones objetivo (base = acciones + pose por defecto)
                # La política solo genera 15 acciones (piernas + torso)
                # Los brazos (8 articulaciones) se añaden como ceros
                pd_target = np.concatenate([scaled_actions, np.zeros(8)]) + self.default_dof_pos
                
                # =============================================================
                # INTEGRACIÓN DEL CONTROLADOR DE PICK & PLACE
                # =============================================================
                if self.use_pick_place:
                    # Actualizar el controlador de brazos
                    arm_targets, loco_cmds = self.pick_place_controller.update(
                        self.dof_pos, 
                        self.control_dt
                    )
                    
                    if arm_targets is not None:
                        # Sobrescribir objetivos de brazos y torso (índices 12+)
                        pd_target[12:] = arm_targets[12:]
                        
                        # Actualizar comandos de locomoción
                        if loco_cmds is not None:
                            self.viewer.commands[0] = loco_cmds['vx']    # Adelante/atrás
                            self.viewer.commands[2] = loco_cmds['vy']    # Lateral
                            self.viewer.commands[1] = loco_cmds['yaw']   # Orientación
                    else:
                        # El controlador terminó, desactivar pick & place
                        if not self.pick_place_controller.is_active():
                            self.use_pick_place = False
                            # Detener el robot
                            self.viewer.commands[0] = 0
                            self.viewer.commands[1] = 0
                            self.viewer.commands[2] = 0

                # ---------------------------------------------------------
                # ACTUALIZAR CICLO DE MARCHA
                # ---------------------------------------------------------
                # El ciclo de marcha coordina el movimiento de las piernas
                # Cada pierna tiene una fase (0-1) que determina su posición en el paso
                
                self.gait_cycle = np.remainder(
                    self.gait_cycle + self.control_dt * self.gait_freq, 1.0
                )
                
                # Lógica de sincronización de pasos
                # Cuando el robot está quieto, ambas piernas en fase 0.25 (parado)
                # Cuando camina, las piernas están desfasadas 0.5 (alternando)
                if self._in_place_stand_flag and (
                    (np.abs(self.gait_cycle[0] - 0.25) < 0.05) or
                    (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
                ):
                    self.gait_cycle = np.array([0.25, 0.25])  # Sincronizar parado
                    
                if (not self._in_place_stand_flag) and (
                    (np.abs(self.gait_cycle[0] - 0.25) < 0.05) and
                    (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
                ):
                    self.gait_cycle = np.array([0.25, 0.75])  # Desfasar para caminar

                # ---------------------------------------------------------
                # ACTUALIZAR CÁMARA Y RENDERIZAR
                # ---------------------------------------------------------
                # La cámara sigue al robot
                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                self.viewer.render()

            # =================================================================
            # PASO 3: CALCULAR TORQUES (Control PD)
            # =================================================================
            # Ley de control PD: τ = Kp * (q_target - q) - Kd * q_dot
            # 
            # - Término proporcional (Kp): fuerza proporcional al error de posición
            # - Término derivativo (Kd): amortiguamiento basado en velocidad
            
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            
            # Aplicar límites de torque para realismo
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            
            # Enviar torques a MuJoCo
            self.data.ctrl = torque

            # =================================================================
            # PASO 4: EJECUTAR PASO DE FÍSICA
            # =================================================================
            mujoco.mj_step(self.model, self.data)

        # Cerrar el visualizador al terminar
        self.viewer.close()


# =============================================================================
# PUNTO DE ENTRADA DEL PROGRAMA
# =============================================================================

if __name__ == "__main__":
    """
    Función principal que inicializa y ejecuta la simulación.
    
    1. Detecta si hay GPU disponible
    2. Carga la política de control preentrenada
    3. Crea el entorno de simulación
    4. Ejecuta el bucle principal
    """
    
    # Tipo de robot a simular
    robot = "g1"
    
    # Seleccionar dispositivo (GPU si está disponible)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # Ruta a la política preentrenada
    # Esta es una red neuronal que controla la locomoción del robot
    policy_pth = '/home/iudc/unitree_robotic/politicas/test.pt'
    
    # Cargar la política (modelo JIT de PyTorch)
    print("Cargando política...")
    policy_jit = torch.jit.load(policy_pth, map_location=device)

    # Crear el entorno de simulación
    print("Inicializando entorno...")
    env = HumanoidEnv(policy_jit=policy_jit, robot_type=robot, device=device)

    # Ejecutar la simulación
    print("Iniciando simulación...")
    env.run()