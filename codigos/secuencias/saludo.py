"""
=============================================================================
SIMULADOR DE ROBOT HUMANOIDE G1 CON SECUENCIA DE SALUDO
=============================================================================

Descripción General:
--------------------
Este programa simula un robot humanoide Unitree G1 en el entorno de física
MuJoCo. Permite controlar la locomoción del robot mediante teclado y ejecutar
una secuencia animada de saludo con el brazo derecho.

El sistema utiliza:
- MuJoCo: Motor de física para simulación realista
- PyTorch: Para ejecutar la política de control (red neuronal)
- GLFW: Para capturar entrada del teclado

Arquitectura del Sistema:
-------------------------
┌─────────────────────────────────────────────────────────────────────────┐
│                          BUCLE PRINCIPAL                                │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐      │
│  │  Extraer  │───▶│  Obtener  │───▶│  Ejecutar │───▶│  Calcular │      │
│  │  Estado   │    │Observación│    │  Política │    │  Torques  │      │
│  └───────────┘    └───────────┘    └───────────┘    └───────────┘      │
│        │                                                    │           │
│        │              ┌───────────────┐                    │           │
│        │              │  Controlador  │                    │           │
│        └─────────────▶│   de Saludo   │◀───────────────────┘           │
│                       └───────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────┘

Controles de Teclado:
---------------------
- W/S: Velocidad adelante/atrás
- A/D: Girar izquierda/derecha  
- Q/E: Movimiento lateral
- Z/X: Ajustar altura
- G: Iniciar secuencia de saludo
- R: Resetear pose de brazos
- ESC: Salir

Requisitos:
-----------
- Python 3.8+
- MuJoCo 2.3+
- PyTorch 1.12+
- NumPy
- GLFW

Autor: [Julian Rivera]
Versión: 1.0
Fecha: [18-Dic-2025]
=============================================================================
"""

# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================

import types          # Para asignar métodos dinámicamente a objetos
import numpy as np    # Operaciones matemáticas y arrays
import mujoco         # Motor de física
import mujoco_viewer  # Visualizador 3D para MuJoCo
import glfw           # Manejo de ventanas y entrada de teclado
from collections import deque  # Cola de tamaño fijo para historiales
import torch          # Framework de aprendizaje profundo
import math           # Funciones matemáticas (cos, sin, pi)


# =============================================================================
# CONSTANTES GLOBALES
# =============================================================================

# Estas constantes definen valores importantes usados en todo el programa.
# Centralizarlas aquí facilita su modificación y mantenimiento.

# Rutas a archivos del modelo y políticas
# (Modificar según la ubicación en tu sistema)
MODEL_PATH = "" # Ruta de modelo del robot en xml
POLICY_PATH = "" # Ruta de la politica principal
ADAPTER_PATH = "" # Ruta de la politica de adapatación
NORM_STATS_PATH = "" # Ruta de la politica de la normalizacion de estados


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def quatToEuler(quat):
    """
    Convierte una orientación de Cuaternión a Ángulos de Euler.
    
    ¿Por qué necesitamos esta conversión?
    -------------------------------------
    Los cuaterniones son la representación preferida en robótica y gráficos 3D
    porque:
    1. No sufren de "gimbal lock" (bloqueo de cardán)
    2. Son más eficientes para componer rotaciones
    3. Interpolan suavemente entre orientaciones
    
    Sin embargo, los ángulos de Euler son más intuitivos para humanos:
    - Roll: ¿Cuánto está inclinado hacia los lados? (como un barco)
    - Pitch: ¿Cuánto está inclinado hacia adelante/atrás? (como un avión)
    - Yaw: ¿Hacia dónde está mirando? (como una brújula)
    
    Explicación Matemática Simplificada:
    ------------------------------------
    Un cuaternión q = [w, x, y, z] representa una rotación donde:
    - w = cos(θ/2), siendo θ el ángulo de rotación
    - [x, y, z] = sin(θ/2) * [eje de rotación]
    
    Las fórmulas usadas aquí derivan de la matriz de rotación equivalente
    y extraen los ángulos de Euler en orden ZYX (yaw-pitch-roll).
    
    Parámetros:
    -----------
    quat : numpy.ndarray
        Array de 4 elementos [w, x, y, z] representando el cuaternión.
        - w: Componente escalar (parte real)
        - x, y, z: Componentes vectoriales (parte imaginaria)
        - El cuaternión debe estar normalizado (|q| = 1)
    
    Retorna:
    --------
    eulerVec : numpy.ndarray
        Array de 3 elementos [roll, pitch, yaw] en radianes.
        - roll:  Rotación alrededor del eje X, rango [-π, π]
        - pitch: Rotación alrededor del eje Y, rango [-π/2, π/2]
        - yaw:   Rotación alrededor del eje Z, rango [-π, π]
    
    Ejemplo:
    --------
    >>> # Cuaternión identidad (sin rotación)
    >>> q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    >>> euler = quatToEuler(q_identity)
    >>> print(euler)  # [0.0, 0.0, 0.0]
    
    >>> # Rotación de 90° alrededor del eje Z (yaw)
    >>> q_yaw90 = np.array([0.707, 0.0, 0.0, 0.707])
    >>> euler = quatToEuler(q_yaw90)
    >>> print(np.degrees(euler))  # [0.0, 0.0, 90.0]
    
    Notas:
    ------
    - El orden de extracción es ZYX (primero yaw, luego pitch, luego roll)
    - Existe una singularidad cuando pitch = ±90° (gimbal lock)
    - Esta función maneja el gimbal lock limitando pitch a ±90°
    
    Referencias:
    ------------
    - https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    - "Quaternions and Rotation Sequences" by Jack B. Kuipers
    """
    # Crear array de salida inicializado en ceros
    # Índice 0: Roll, Índice 1: Pitch, Índice 2: Yaw
    eulerVec = np.zeros(3)
    
    # Extraer componentes del cuaternión para mayor claridad
    # Notación estándar: w es la parte escalar, (x,y,z) es la parte vectorial
    qw = quat[0]  # Componente escalar (coseno del medio ángulo)
    qx = quat[1]  # Componente X del vector
    qy = quat[2]  # Componente Y del vector
    qz = quat[3]  # Componente Z del vector

    # =========================================================================
    # CÁLCULO DEL ROLL (Rotación alrededor del eje X)
    # =========================================================================
    # 
    # El roll representa la inclinación lateral, como un barco meciéndose
    # de babor a estribor, o un avión inclinando sus alas.
    #
    # Fórmula derivada de la matriz de rotación:
    # roll = atan2(2(qw*qx + qy*qz), 1 - 2(qx² + qy²))
    #
    # Usamos atan2 en lugar de atan porque:
    # 1. atan2(y, x) maneja correctamente todos los cuadrantes
    # 2. Retorna valores en el rango [-π, π]
    # 3. Maneja el caso cuando x = 0 (donde atan fallaría)
    
    sinr_cosp = 2.0 * (qw * qx + qy * qz)  # Numerador: sin(roll) * cos(pitch)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)  # Denominador: cos(roll) * cos(pitch)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # =========================================================================
    # CÁLCULO DEL PITCH (Rotación alrededor del eje Y)
    # =========================================================================
    #
    # El pitch representa la inclinación hacia adelante/atrás, como un avión
    # apuntando su nariz hacia arriba o hacia abajo.
    #
    # Fórmula: pitch = asin(2(qw*qy - qz*qx))
    #
    # IMPORTANTE: Manejo del Gimbal Lock
    # ----------------------------------
    # Cuando pitch = ±90°, el seno vale ±1 y perdemos un grado de libertad
    # (roll y yaw se vuelven indistinguibles). Esto se llama "gimbal lock".
    #
    # Además, debido a errores de punto flotante, el argumento de asin
    # podría exceder ligeramente el rango [-1, 1], causando un error.
    # Por eso verificamos y limitamos el valor.
    
    sinp = 2.0 * (qw * qy - qz * qx)  # sin(pitch)
    
    if np.abs(sinp) >= 1.0:
        # Estamos en o cerca del gimbal lock
        # Limitar a exactamente ±90° (±π/2 radianes)
        # np.copysign copia el signo de sinp al valor π/2
        eulerVec[1] = np.copysign(np.pi / 2.0, sinp)
    else:
        # Caso normal: calcular arcoseno
        eulerVec[1] = np.arcsin(sinp)

    # =========================================================================
    # CÁLCULO DEL YAW (Rotación alrededor del eje Z)
    # =========================================================================
    #
    # El yaw representa la dirección hacia donde apunta, como una brújula
    # o un coche girando en una intersección.
    #
    # Fórmula: yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy² + qz²))
    
    siny_cosp = 2.0 * (qw * qz + qx * qy)  # Numerador: sin(yaw) * cos(pitch)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)  # Denominador: cos(yaw) * cos(pitch)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)

    return eulerVec


def _key_callback(self, window, key, scancode, action, mods):
    """
    Función callback que maneja todos los eventos de teclado.
    
    ¿Qué es un Callback?
    --------------------
    Un callback es una función que se "registra" para ser llamada automáticamente
    cuando ocurre un evento específico. En este caso, GLFW llama a esta función
    cada vez que el usuario presiona o suelta una tecla.
    
    ¿Por qué usar 'self' como primer parámetro?
    -------------------------------------------
    Esta función se asigna dinámicamente como método del objeto viewer usando
    types.MethodType(). Esto permite que la función acceda a los atributos
    del viewer (como self.commands) como si fuera un método de clase.
    
    Parámetros:
    -----------
    self : MujocoViewer
        Referencia al objeto viewer (asignada dinámicamente).
        Contiene el array 'commands' que almacena los comandos del usuario.
    
    window : GLFWwindow
        Handle de la ventana GLFW donde ocurrió el evento.
        No lo usamos directamente, pero GLFW lo requiere.
    
    key : int
        Código de la tecla presionada (constantes GLFW como glfw.KEY_W).
        Ver: https://www.glfw.org/docs/latest/group__keys.html
    
    scancode : int
        Código de escaneo específico del sistema operativo.
        No lo usamos, pero GLFW lo proporciona.
    
    action : int
        Tipo de evento:
        - glfw.PRESS: La tecla fue presionada
        - glfw.RELEASE: La tecla fue soltada
        - glfw.REPEAT: La tecla está siendo mantenida (repetición automática)
    
    mods : int
        Modificadores activos (Shift, Ctrl, Alt, etc.).
        Es una máscara de bits que podemos verificar con operadores AND.
    
    Estructura del Array commands[]:
    --------------------------------
    El array self.commands tiene 8 elementos que controlan diferentes aspectos:
    
    Índice | Nombre       | Descripción                    | Teclas  | Unidad
    -------|--------------|--------------------------------|---------|--------
    [0]    | vx           | Velocidad hacia adelante/atrás | W/S     | m/s
    [1]    | yaw_rate     | Velocidad de giro              | A/D     | rad/s
    [2]    | vy           | Velocidad lateral              | Q/E     | m/s
    [3]    | height_adj   | Ajuste de altura del cuerpo    | Z/X     | m
    [4]    | torso_yaw    | Rotación del torso             | U/J     | rad
    [5]    | torso_pitch  | Inclinación del torso          | I/K     | rad
    [6]    | torso_roll   | Balanceo del torso             | O/L     | rad
    [7]    | (reservado)  | Para uso futuro                | -       | -
    
    Ejemplo de Uso:
    ---------------
    El usuario presiona 'W' tres veces:
    - Primera vez: commands[0] = 0.0 + 0.05 = 0.05 m/s
    - Segunda vez: commands[0] = 0.05 + 0.05 = 0.10 m/s
    - Tercera vez: commands[0] = 0.10 + 0.05 = 0.15 m/s
    
    El robot ahora camina hacia adelante a 0.15 m/s.
    """
    
    # =========================================================================
    # FILTRAR EVENTOS
    # =========================================================================
    # Solo procesamos cuando la tecla es PRESIONADA (no cuando se suelta
    # o se repite). Esto evita que los comandos se acumulen demasiado rápido.
    
    if action != glfw.PRESS:
        return  # Ignorar eventos de soltar o repetir
    
    # =========================================================================
    # CONTROLES DE LOCOMOCIÓN (Movimiento del robot)
    # =========================================================================
    # Estos controles afectan cómo se mueve el robot en el espacio.
    
    # --- Velocidad Adelante/Atrás (Eje X del robot) ---
    # W: Acelerar hacia adelante (dirección positiva del eje X)
    # S: Acelerar hacia atrás (dirección negativa del eje X)
    if key == glfw.KEY_W:
        self.commands[0] += 0.05  # Incrementar velocidad frontal
    elif key == glfw.KEY_S:
        self.commands[0] -= 0.05  # Decrementar velocidad frontal (ir hacia atrás)
    
    # --- Velocidad de Giro (Rotación alrededor del eje Z) ---
    # A: Girar a la izquierda (sentido antihorario visto desde arriba)
    # D: Girar a la derecha (sentido horario visto desde arriba)
    elif key == glfw.KEY_A:
        self.commands[1] += 0.1  # Incrementar velocidad de giro (izquierda)
    elif key == glfw.KEY_D:
        self.commands[1] -= 0.1  # Decrementar velocidad de giro (derecha)
    
    # --- Velocidad Lateral (Eje Y del robot) ---
    # Q: Moverse hacia la izquierda (dirección positiva del eje Y)
    # E: Moverse hacia la derecha (dirección negativa del eje Y)
    elif key == glfw.KEY_Q:
        self.commands[2] += 0.05  # Incrementar velocidad lateral izquierda
    elif key == glfw.KEY_E:
        self.commands[2] -= 0.05  # Incrementar velocidad lateral derecha
    
    # --- Altura del Cuerpo ---
    # Z: Subir el centro de masa del robot
    # X: Bajar el centro de masa del robot
    # Nota: La altura base es 0.75m, este valor se suma a esa base
    elif key == glfw.KEY_Z:
        self.commands[3] += 0.05  # Subir
    elif key == glfw.KEY_X:
        self.commands[3] -= 0.05  # Bajar
    
    # =========================================================================
    # CONTROLES DEL TORSO (Orientación del tronco superior)
    # =========================================================================
    # Estos controles permiten inclinar el torso del robot independientemente
    # de la dirección de movimiento. Útil para mirar alrededor o mantener
    # equilibrio en terreno inclinado.
    
    # --- Rotación del Torso (Yaw) ---
    # J: Rotar torso hacia la izquierda
    # U: Rotar torso hacia la derecha
    elif key == glfw.KEY_J:
        self.commands[4] += 0.1
    elif key == glfw.KEY_U:
        self.commands[4] -= 0.1
    
    # --- Inclinación del Torso (Pitch) ---
    # K: Inclinar torso hacia adelante
    # I: Inclinar torso hacia atrás
    elif key == glfw.KEY_K:
        self.commands[5] += 0.05
    elif key == glfw.KEY_I:
        self.commands[5] -= 0.05
    
    # --- Balanceo del Torso (Roll) ---
    # L: Inclinar torso hacia la derecha
    # O: Inclinar torso hacia la izquierda
    elif key == glfw.KEY_L:
        self.commands[6] += 0.05
    elif key == glfw.KEY_O:
        self.commands[6] -= 0.1
    
    # =========================================================================
    # CONTROLES DE SECUENCIA DE SALUDO
    # =========================================================================
    # Estas teclas activan animaciones pre-programadas de los brazos.
    
    # --- Iniciar Secuencia de Saludo ---
    # G: El robot ejecuta una secuencia completa de saludo con el brazo derecho
    elif key == glfw.KEY_G:
        self.start_wave_sequence = True  # Flag que será procesado en el bucle principal
        print("\n>>> INICIANDO SECUENCIA DE SALUDO <<<")
    
    # --- Resetear Pose de Brazos ---
    # R: Los brazos vuelven a su posición neutral/relajada
    elif key == glfw.KEY_R:
        self.reset_to_default_pose = True
        print("\n>>> RESETEANDO A POSE POR DEFECTO <<<")
    
    # =========================================================================
    # CONTROL DEL SISTEMA
    # =========================================================================
    
    # --- Salir del Programa ---
    # ESC: Cierra la simulación de forma limpia
    elif key == glfw.KEY_ESCAPE:
        print("Presionado ESC")
        print("Cerrando simulación...")
        glfw.set_window_should_close(self.window, True)
        return  # Salir inmediatamente sin imprimir estado
    
    # =========================================================================
    # MOSTRAR ESTADO ACTUAL
    # =========================================================================
    # Después de cada comando, mostramos el estado actual para feedback visual.
    # El formato usa f-strings con alineación para una salida ordenada.
    
    print(
        f"vx: {self.commands[0]:<8.2f}"      # Velocidad frontal, 8 chars, 2 decimales
        f"vy: {self.commands[2]:<8.2f}"      # Velocidad lateral
        f"yaw: {self.commands[1]:<8.2f}"     # Velocidad de giro
        f"altura: {(0.75 + self.commands[3]):<8.2f}"  # Altura total (base + ajuste)
        f"torso_yaw: {self.commands[4]:<8.2f}"
        f"torso_pitch: {self.commands[5]:<8.2f}"
        f"torso_roll: {self.commands[6]:<8.2f}"
    )


# =============================================================================
# CLASE: CONTROLADOR DE SECUENCIA DE SALUDO
# =============================================================================

class WaveController:
    """
    Controlador que gestiona la secuencia de saludo del robot G1.
    
    Descripción:
    ------------
    Este controlador permite que el robot ejecute un saludo natural y fluido
    levantando el brazo derecho y moviéndolo de lado a lado. El movimiento
    está interpolado suavemente para verse natural y no dañar los motores.
    
    Secuencia de Saludo:
    --------------------
    La secuencia completa consiste en los siguientes pasos:
    
    1. POSICIÓN INICIAL (0.5s)
       └── Brazos relajados a los lados del cuerpo
       
    2. LEVANTAR BRAZO (1.0s)
       └── El brazo derecho se eleva y extiende hacia el lado
       
    3. SALUDAR x3 (1.0s + 1.0s cada ciclo)
       ├── Flexionar codo (mano hacia izquierda)
       └── Extender codo (mano hacia derecha)
       
    4. BAJAR BRAZO (1.0s)
       └── Volver a la posición inicial
    
    Tiempo total: ~8.5 segundos
    
    Arquitectura del Controlador:
    -----------------------------
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                      WaveController                              │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
    │  │    Poses    │    │ Secuencia   │    │Interpolador │         │
    │  │  (5 poses)  │───▶│  de Pasos   │───▶│   Suave     │──▶ pos  │
    │  └─────────────┘    └─────────────┘    └─────────────┘         │
    │         ▲                  │                   ▲                │
    │         │                  ▼                   │                │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
    │  │  Joint Map  │    │   Timers    │    │   Estado    │         │
    │  │(nombre→idx) │    │ (duración)  │    │  (t, activo)│         │
    │  └─────────────┘    └─────────────┘    └─────────────┘         │
    └─────────────────────────────────────────────────────────────────┘
    
    Interpolación Suave:
    --------------------
    Para que los movimientos se vean naturales, usamos interpolación
    con función coseno en lugar de lineal:
    
    Lineal:     posición = inicio + t * (fin - inicio)
    Coseno:     posición = inicio + (1-cos(πt))/2 * (fin - inicio)
    
    La interpolación coseno produce:
    - Inicio suave (aceleración gradual)
    - Velocidad máxima en el medio
    - Final suave (desaceleración gradual)
    
    Esto es similar a cómo los humanos mueven naturalmente sus extremidades.
    
    Atributos:
    ----------
    joint_map : dict
        Diccionario que mapea nombres legibles de articulaciones a sus
        índices numéricos en el array de posiciones del robot.
        Ejemplo: {'right_shoulder_pitch': 19, 'right_elbow': 22, ...}
    
    poses : list[dict]
        Lista de poses predefinidas. Cada pose es un diccionario con:
        - 'name': Nombre descriptivo para debugging
        - 'joints': Diccionario {nombre_articulación: ángulo_radianes}
    
    sequence_steps : list[int]
        Orden de ejecución de las poses (índices en la lista 'poses').
        Ejemplo: [0, 1, 2, 3, 2, 3, 2, 3, 4] para el saludo completo.
    
    sequence_durations : list[float]
        Duración en segundos de cada transición.
        Debe tener la misma longitud que sequence_steps.
    
    trajectory_active : bool
        True si hay una transición de pose en progreso.
    
    sequence_active : bool
        True si la secuencia completa está en ejecución.
    
    Ejemplo de Uso:
    ---------------
    >>> controller = WaveController(num_dofs=23, default_dof_pos=default_pos)
    >>> 
    >>> # Iniciar la secuencia de saludo
    >>> controller.start_sequence(current_joint_positions)
    >>> 
    >>> # En el bucle de control (cada 20ms):
    >>> while True:
    ...     target_positions = controller.update(current_positions, dt=0.02)
    ...     if target_positions is not None:
    ...         apply_to_robot(target_positions)
    ...     else:
    ...         break  # Secuencia terminada
    """
    
    def __init__(self, num_dofs, default_dof_pos):
        """
        Inicializa el controlador de saludo.
        
        Parámetros:
        -----------
        num_dofs : int
            Número total de grados de libertad del robot.
            Para el G1: 23 DOF (12 piernas + 3 cintura + 8 brazos)
        
        default_dof_pos : numpy.ndarray
            Posiciones angulares por defecto de todas las articulaciones.
            Shape: (num_dofs,)
            Unidades: radianes
        
        Proceso de Inicialización:
        --------------------------
        1. Guardar configuración básica
        2. Crear mapeo de nombres de articulaciones
        3. Definir todas las poses de la secuencia
        4. Inicializar variables de estado
        5. Configurar la secuencia de pasos y duraciones
        """
        # Guardar parámetros básicos
        self.num_dofs = num_dofs
        self.default_dof_pos = default_dof_pos.copy()  # Copia para evitar modificaciones
        
        # =====================================================================
        # MAPEO DE NOMBRES DE ARTICULACIONES A ÍNDICES
        # =====================================================================
        # 
        # Este diccionario es fundamental para la legibilidad del código.
        # En lugar de recordar que el índice 19 es el hombro derecho,
        # podemos escribir 'right_shoulder_pitch'.
        #
        # Estructura Física del Robot G1:
        # --------------------------------
        #
        #                    [CABEZA]
        #                       │
        #            ┌──────[TORSO]──────┐
        #            │    (cintura)      │
        #     [BRAZO IZQ]           [BRAZO DER]
        #     (4 joints)             (4 joints)
        #            │                   │
        #            └────────┬──────────┘
        #                     │
        #              ┌──────┴──────┐
        #        [PIERNA IZQ]  [PIERNA DER]
        #         (6 joints)   (6 joints)
        #
        # Total: 6 + 6 + 3 + 4 + 4 = 23 grados de libertad
        #
        # =====================================================================
        
        self.joint_map = {
            # -----------------------------------------------------------------
            # PIERNA IZQUIERDA (Índices 0-5)
            # -----------------------------------------------------------------
            # Orden: cadera → rodilla → tobillo
            # Cada articulación tiene movimientos específicos:
            
            'left_hip_pitch': 0,    # Flexión de cadera: levantar pierna adelante/atrás
            'left_hip_roll': 1,     # Abducción de cadera: separar pierna del cuerpo
            'left_hip_yaw': 2,      # Rotación de cadera: girar pierna sobre su eje
            'left_knee': 3,         # Flexión de rodilla: doblar la pierna
            'left_ankle_pitch': 4,  # Flexión de tobillo: apuntar pie arriba/abajo
            'left_ankle_roll': 5,   # Inversión de tobillo: inclinar pie lateral
            
            # -----------------------------------------------------------------
            # PIERNA DERECHA (Índices 6-11)
            # -----------------------------------------------------------------
            # Estructura idéntica a la pierna izquierda
            
            'right_hip_pitch': 6,
            'right_hip_roll': 7,
            'right_hip_yaw': 8,
            'right_knee': 9,
            'right_ankle_pitch': 10,
            'right_ankle_roll': 11,
            
            # -----------------------------------------------------------------
            # CINTURA (Índices 12-14)
            # -----------------------------------------------------------------
            # Conecta el torso superior con las piernas
            # Permite que el torso se mueva independientemente
            
            'waist_yaw': 12,    # Rotación: girar torso izquierda/derecha
            'waist_roll': 13,   # Balanceo: inclinar torso lateralmente
            'waist_pitch': 14,  # Inclinación: inclinar torso adelante/atrás
            
            # -----------------------------------------------------------------
            # BRAZO IZQUIERDO (Índices 15-18)
            # -----------------------------------------------------------------
            # Orden: hombro (3 DOF) → codo (1 DOF)
            # Nota: Este modelo no incluye muñeca
            
            'left_shoulder_pitch': 15,  # Flexión: levantar brazo adelante/atrás
            'left_shoulder_roll': 16,   # Abducción: separar brazo del cuerpo
            'left_shoulder_yaw': 17,    # Rotación: girar brazo sobre su eje
            'left_elbow': 18,           # Flexión: doblar el codo
            
            # -----------------------------------------------------------------
            # BRAZO DERECHO (Índices 19-22)
            # -----------------------------------------------------------------
            # Estructura idéntica al brazo izquierdo
            # Este es el brazo que usamos para saludar
            
            'right_shoulder_pitch': 19,
            'right_shoulder_roll': 20,
            'right_shoulder_yaw': 21,
            'right_elbow': 22,
        }
        
        # =====================================================================
        # DEFINICIÓN DE POSES PARA LA SECUENCIA DE SALUDO
        # =====================================================================
        #
        # Cada pose define las posiciones objetivo de las articulaciones
        # relevantes (brazos y cintura). Las piernas no se modifican.
        #
        # Convención de Signos (específica del robot G1):
        # -----------------------------------------------
        # - shoulder_pitch: Negativo = hacia adelante, Positivo = hacia atrás
        # - shoulder_roll:  Negativo = separar del cuerpo, Positivo = hacia cuerpo
        # - shoulder_yaw:   Rotación interna/externa
        # - elbow:          Negativo = extender, Positivo = flexionar
        #
        # Nota: Los valores fueron ajustados experimentalmente para verse
        # naturales en la simulación. Pueden requerir ajustes para el robot real.
        #
        # =====================================================================
        
        self.poses = [
            # =================================================================
            # POSE 0: POSICIÓN INICIAL
            # =================================================================
            # Los brazos están relajados a los lados del cuerpo.
            # Esta es la pose "home" o de descanso.
            {
                'name': "Posición inicial",
                'joints': {
                    # --- Brazo Izquierdo ---
                    # Se mantiene en posición neutral durante todo el saludo
                    'left_shoulder_pitch': 0.5,   # Ligeramente hacia atrás
                    'left_shoulder_roll': 0.0,    # Pegado al cuerpo
                    'left_shoulder_yaw': 0.2,     # Ligera rotación externa
                    'left_elbow': 0.3,            # Ligeramente flexionado
                    
                    # --- Brazo Derecho ---
                    # Posición simétrica al izquierdo
                    'right_shoulder_pitch': 0.5,
                    'right_shoulder_roll': 0.0,
                    'right_shoulder_yaw': -0.2,   # Rotación opuesta (simetría)
                    'right_elbow': 0.3,
                    
                    # --- Cintura ---
                    # Sin rotación
                    'waist_yaw': 0.0,
                    'waist_roll': 0.0,
                    'waist_pitch': 0.0,
                }
            },
            
            # =================================================================
            # POSE 1: LEVANTAR BRAZO
            # =================================================================
            # El brazo derecho se eleva hacia el lado, preparándose para saludar.
            # El codo está extendido.
            {
                'name': "Levantar brazo",
                'joints': {
                    # --- Brazo Izquierdo (sin cambios) ---
                    'left_shoulder_pitch': 0.5,
                    'left_shoulder_roll': 0.0,
                    'left_shoulder_yaw': 0.2,
                    'left_elbow': 0.3,
                    
                    # --- Brazo Derecho (levantado) ---
                    'right_shoulder_pitch': 0.0,    # Horizontal
                    'right_shoulder_roll': -1.25,   # Muy separado del cuerpo
                    'right_shoulder_yaw': -1.78,    # Rotado hacia afuera
                    'right_elbow': 0.0,             # Codo completamente extendido
                    
                    # --- Cintura ---
                    'waist_yaw': 0.0,
                    'waist_roll': 0.0,
                    'waist_pitch': 0.0,
                }
            },
            
            # =================================================================
            # POSE 2: SALUDO - MANO HACIA IZQUIERDA
            # =================================================================
            # El codo se flexiona, moviendo la mano hacia la izquierda.
            # Esta es la primera posición del movimiento de "adiós".
            {
                'name': "Saludo izquierda",
                'joints': {
                    # --- Brazo Izquierdo (sin cambios) ---
                    'left_shoulder_pitch': 0.5,
                    'left_shoulder_roll': 0.0,
                    'left_shoulder_yaw': 0.2,
                    'left_elbow': 0.3,
                    
                    # --- Brazo Derecho (codo flexionado) ---
                    'right_shoulder_pitch': 0.0,
                    'right_shoulder_roll': -1.25,
                    'right_shoulder_yaw': -1.78,
                    'right_elbow': -0.827,  # Codo flexionado (mano hacia izq)
                    
                    # --- Cintura ---
                    'waist_yaw': 0.0,
                    'waist_roll': 0.0,
                    'waist_pitch': 0.0,
                }
            },
            
            # =================================================================
            # POSE 3: SALUDO - MANO HACIA DERECHA
            # =================================================================
            # El codo se extiende parcialmente, moviendo la mano hacia la derecha.
            # Alternando entre poses 2 y 3 se crea el movimiento de "adiós".
            {
                'name': "Saludo derecha",
                'joints': {
                    # --- Brazo Izquierdo (sin cambios) ---
                    'left_shoulder_pitch': 0.5,
                    'left_shoulder_roll': 0.0,
                    'left_shoulder_yaw': 0.2,
                    'left_elbow': 0.3,
                    
                    # --- Brazo Derecho (codo más extendido) ---
                    'right_shoulder_pitch': 0.0,
                    'right_shoulder_roll': -1.25,
                    'right_shoulder_yaw': -1.78,
                    'right_elbow': 0.3,  # Codo extendido (mano hacia der)
                    
                    # --- Cintura ---
                    'waist_yaw': 0.0,
                    'waist_roll': 0.0,
                    'waist_pitch': 0.0,
                }
            },
            
            # =================================================================
            # POSE 4: VOLVER A POSICIÓN INICIAL
            # =================================================================
            # Idéntica a la Pose 0. El brazo vuelve a descansar al costado.
            {
                'name': "Volver a inicial",
                'joints': {
                    # --- Brazo Izquierdo ---
                    'left_shoulder_pitch': 0.5,
                    'left_shoulder_roll': 0.0,
                    'left_shoulder_yaw': 0.2,
                    'left_elbow': 0.3,
                    
                    # --- Brazo Derecho (vuelve a neutral) ---
                    'right_shoulder_pitch': 0.5,
                    'right_shoulder_roll': 0.0,
                    'right_shoulder_yaw': -0.2,
                    'right_elbow': 0.3,
                    
                    # --- Cintura ---
                    'waist_yaw': 0.0,
                    'waist_roll': 0.0,
                    'waist_pitch': 0.0,
                }
            },
        ]
        
        # =====================================================================
        # VARIABLES DE ESTADO DE LA TRAYECTORIA
        # =====================================================================
        # Estas variables rastrean el progreso de la transición actual
        # entre dos poses.
        
        self.trajectory_active = False  # ¿Hay una transición en progreso?
        self.trajectory_t = 0.0         # Tiempo transcurrido en esta transición
        self.trajectory_T = 1.0         # Duración total de la transición
        
        # Posiciones de inicio y fin de la transición actual
        self.trajectory_start_pos = np.zeros(num_dofs)
        self.trajectory_target_pos = np.zeros(num_dofs)
        
        # =====================================================================
        # VARIABLES DE ESTADO DE LA SECUENCIA
        # =====================================================================
        # Estas variables rastrean el progreso de la secuencia completa.
        
        self.sequence_active = False  # ¿La secuencia está en ejecución?
        self.current_step = 0         # Índice del paso actual en sequence_steps
        
        # =====================================================================
        # DEFINICIÓN DE LA SECUENCIA DE PASOS
        # =====================================================================
        # Esta lista define el ORDEN en que se ejecutan las poses.
        # Los números son índices en la lista self.poses.
        #
        # Secuencia: inicial → levantar → (izq→der)×3 → bajar
        
        self.sequence_steps = [
            0,      # Paso 0: Ir a posición inicial
            1,      # Paso 1: Levantar brazo
            2, 3,   # Pasos 2-3: Primer saludo (izquierda → derecha)
            2, 3,   # Pasos 4-5: Segundo saludo
            2, 3,   # Pasos 6-7: Tercer saludo
            4,      # Paso 8: Bajar brazo (volver a inicial)
        ]
        
        # =====================================================================
        # DURACIONES DE CADA TRANSICIÓN
        # =====================================================================
        # Cuántos segundos toma cada transición entre poses.
        # Debe tener la misma longitud que sequence_steps.
        
        self.sequence_durations = [
            0.5,        # Inicial → Inicial (rápido, casi instantáneo)
            1.0,        # Inicial → Levantar (movimiento grande, más lento)
            1.0, 1.0,   # Saludo 1 (izq → der)
            1.0, 1.0,   # Saludo 2
            1.0, 1.0,   # Saludo 3
            1.0,        # Levantar → Inicial (bajar el brazo)
        ]
    
    def _name_to_index(self, joint_name):
        """
        Convierte el nombre de una articulación a su índice numérico.
        
        Esta función simplifica el código al permitir usar nombres
        descriptivos en lugar de recordar índices numéricos.
        
        Parámetros:
        -----------
        joint_name : str
            Nombre de la articulación.
            Debe existir en self.joint_map.
            Ejemplos: 'right_shoulder_pitch', 'left_elbow', 'waist_yaw'
        
        Retorna:
        --------
        int
            Índice numérico de la articulación (0-22 para el G1).
        
        Raises:
        -------
        KeyError
            Si joint_name no existe en joint_map.
        
        Ejemplo:
        --------
        >>> idx = self._name_to_index('right_elbow')
        >>> print(idx)  # 22
        """
        return self.joint_map[joint_name]
    
    def _convert_pose_to_indices(self, pose_joints):
        """
        Convierte un diccionario de nombres de articulaciones a índices.
        
        Las poses se definen usando nombres legibles, pero internamente
        necesitamos índices numéricos para acceder a los arrays de posición.
        
        Parámetros:
        -----------
        pose_joints : dict
            Diccionario con pares {nombre_articulación: ángulo}.
            Ejemplo: {'right_elbow': 0.3, 'right_shoulder_pitch': 0.0}
        
        Retorna:
        --------
        dict
            Diccionario con pares {índice: ángulo}.
            Ejemplo: {22: 0.3, 19: 0.0}
        
        Ejemplo:
        --------
        >>> pose = {'right_elbow': 0.5, 'left_elbow': 0.3}
        >>> indexed = self._convert_pose_to_indices(pose)
        >>> print(indexed)  # {22: 0.5, 18: 0.3}
        """
        indexed_joints = {}
        for joint_name, value in pose_joints.items():
            idx = self._name_to_index(joint_name)
            indexed_joints[idx] = value
        return indexed_joints
    
    def interpolate_position(self, q_init, q_target):
        """
        Calcula una posición interpolada suavemente entre inicio y objetivo.
        
        Descripción:
        ------------
        Esta función implementa interpolación con función coseno, que produce
        movimientos más naturales que la interpolación lineal.
        
        Comparación de métodos de interpolación:
        -----------------------------------------
        
        LINEAL:
        - Fórmula: q(t) = q_init + t * (q_target - q_init)
        - Velocidad constante durante todo el movimiento
        - Cambio brusco al inicio y al final
        - Se ve robótico y puede causar vibraciones
        
        COSENO (usado aquí):
        - Fórmula: q(t) = q_init + (1-cos(πt))/2 * (q_target - q_init)
        - Comienza lento, acelera, luego desacelera
        - Transiciones suaves al inicio y al final
        - Se ve natural, similar al movimiento humano
        
        Visualización del perfil de velocidad:
        
        Lineal:     ████████████████████████████
                    ↑ inicio            final ↑
        
        Coseno:           ▄▄████████████▄▄
                    ↑ inicio            final ↑
        
        Parámetros:
        -----------
        q_init : float
            Posición inicial de la articulación (radianes).
        
        q_target : float
            Posición objetivo de la articulación (radianes).
        
        Retorna:
        --------
        float
            Posición interpolada para el tiempo actual.
        
        Notas:
        ------
        - Usa self.trajectory_t (tiempo actual) y self.trajectory_T (duración)
        - Si t >= T, retorna exactamente q_target (sin overshoot)
        """
        # Si ya terminamos, retornar el objetivo exacto
        if self.trajectory_t >= self.trajectory_T:
            return q_target
        
        # Calcular el ratio de progreso [0, 1]
        # La fórmula (1 - cos(π*t/T)) / 2 produce:
        # - t=0: ratio=0 (inicio)
        # - t=T/2: ratio=0.5 (mitad)
        # - t=T: ratio=1 (final)
        # Pero con aceleración/desaceleración suave
        ratio = (1.0 - math.cos(math.pi * (self.trajectory_t / self.trajectory_T))) / 2.0
        
        # Interpolar entre inicio y objetivo
        return q_init + (q_target - q_init) * ratio
    
    def start_trajectory(self, current_pos, pose_index, duration=1.0):
        """
        Inicia una nueva transición hacia una pose específica.
        
        Esta función configura todos los parámetros necesarios para
        comenzar a interpolar desde la posición actual hacia la pose objetivo.
        
        Parámetros:
        -----------
        current_pos : numpy.ndarray
            Posiciones actuales de todas las articulaciones.
            Shape: (num_dofs,), típicamente (23,) para el G1.
        
        pose_index : int
            Índice de la pose objetivo en self.poses.
            Rango válido: 0 a len(self.poses)-1
        
        duration : float, opcional
            Duración de la transición en segundos.
            Default: 1.0
        
        Proceso:
        --------
        1. Guardar posición actual como punto de inicio
        2. Copiar posición actual como base para el objetivo
        3. Sobrescribir solo las articulaciones definidas en la pose
        4. Configurar temporizadores
        5. Activar el flag de trayectoria
        
        Nota:
        -----
        Las articulaciones NO definidas en la pose mantienen su valor actual.
        Esto permite definir poses parciales (ej: solo brazos) sin afectar
        las piernas.
        """
        # Guardar la posición actual como punto de inicio
        self.trajectory_start_pos = current_pos.copy()
        
        # Iniciar el objetivo como copia de la posición actual
        # Las articulaciones no definidas en la pose mantendrán estos valores
        self.trajectory_target_pos = current_pos.copy()
        
        # Obtener la pose objetivo y convertir nombres a índices
        pose = self.poses[pose_index]['joints']
        indexed_pose = self._convert_pose_to_indices(pose)
        
        # Sobrescribir solo las articulaciones definidas en la pose
        for joint_idx, target_angle in indexed_pose.items():
            self.trajectory_target_pos[joint_idx] = target_angle
        
        # Configurar temporizadores
        self.trajectory_T = duration    # Duración total
        self.trajectory_t = 0.0         # Tiempo transcurrido (empezando)
        
        # Activar la trayectoria
        self.trajectory_active = True
        
        # Mensaje de debug
        print(f"   -> {self.poses[pose_index]['name']}")
    
    def start_sequence(self, current_pos):
        """
        Inicia la secuencia completa de saludo.
        
        Parámetros:
        -----------
        current_pos : numpy.ndarray
            Posiciones actuales de todas las articulaciones.
        
        Proceso:
        --------
        1. Imprimir mensaje de inicio
        2. Activar el flag de secuencia
        3. Resetear el contador de pasos
        4. Iniciar la primera transición
        """
        print("\n¡Hola! Iniciando saludo...")
        
        # Activar la secuencia
        self.sequence_active = True
        self.current_step = 0
        
        # Obtener el primer paso y su duración
        pose_idx = self.sequence_steps[0]
        duration = self.sequence_durations[0]
        
        # Iniciar la primera transición
        self.start_trajectory(current_pos, pose_idx, duration)
    
    def reset_to_default(self, current_pos):
        """
        Resetea los brazos a la posición por defecto.
        
        Útil para interrumpir una secuencia en progreso o para
        volver a la pose neutral en cualquier momento.
        
        Parámetros:
        -----------
        current_pos : numpy.ndarray
            Posiciones actuales de todas las articulaciones.
        """
        # Detener cualquier secuencia en curso
        self.sequence_active = False
        
        # Ir a la pose inicial (índice 0)
        self.start_trajectory(current_pos, 0, duration=1.0)
    
    def update(self, current_pos, dt):
        """
        Actualiza el estado del controlador y calcula posiciones objetivo.
        
        Esta es la función principal que debe llamarse en cada ciclo
        de control. Avanza el tiempo, calcula las posiciones interpoladas,
        y gestiona las transiciones entre pasos de la secuencia.
        
        Parámetros:
        -----------
        current_pos : numpy.ndarray
            Posiciones actuales de todas las articulaciones.
            (No se usa directamente, pero puede ser útil para extensiones)
        
        dt : float
            Tiempo transcurrido desde la última llamada, en segundos.
            Típicamente 0.02 (20ms) para control a 50Hz.
        
        Retorna:
        --------
        numpy.ndarray o None
            - Si hay una trayectoria activa: array de posiciones objetivo
            - Si no hay trayectoria activa: None
        
        Flujo de Ejecución:
        -------------------
        
        ┌─────────────────────────────────────────────────────────┐
        │                     update(dt)                          │
        │  ┌─────────────┐                                        │
        │  │ ¿Trayectoria│──No──▶ return None                    │
        │  │   activa?   │                                        │
        │  └──────┬──────┘                                        │
        │         │ Sí                                            │
        │         ▼                                               │
        │  ┌─────────────┐                                        │
        │  │ Avanzar     │                                        │
        │  │ tiempo (dt) │                                        │
        │  └──────┬──────┘                                        │
        │         ▼                                               │
        │  ┌─────────────┐                                        │
        │  │ Interpolar  │                                        │
        │  │ posiciones  │                                        │
        │  └──────┬──────┘                                        │
        │         ▼                                               │
        │  ┌─────────────┐                                        │
        │  │ ¿Transición │──No──▶ return posiciones              │
        │  │ terminada?  │                                        │
        │  └──────┬──────┘                                        │
        │         │ Sí                                            │
        │         ▼                                               │
        │  ┌─────────────┐                                        │
        │  │ ¿Secuencia  │──No──▶ return posiciones              │
        │  │   activa?   │                                        │
        │  └──────┬──────┘                                        │
        │         │ Sí                                            │
        │         ▼                                               │
        │  ┌─────────────┐                                        │
        │  │ ¿Más pasos? │──Sí──▶ Iniciar siguiente transición   │
        │  └──────┬──────┘                                        │
        │         │ No                                            │
        │         ▼                                               │
        │  ┌─────────────┐                                        │
        │  │ Secuencia   │                                        │
        │  │ completada! │                                        │
        │  └─────────────┘                                        │
        └─────────────────────────────────────────────────────────┘
        """
        # Si no hay trayectoria activa, no hay nada que hacer
        if not self.trajectory_active:
            return None
        
        # Avanzar el tiempo de la trayectoria
        self.trajectory_t += dt
        
        # Calcular posiciones interpoladas para TODAS las articulaciones
        interpolated_pos = np.zeros(self.num_dofs)
        for i in range(self.num_dofs):
            interpolated_pos[i] = self.interpolate_position(
                self.trajectory_start_pos[i],
                self.trajectory_target_pos[i]
            )
        
        # Verificar si la transición actual ha terminado
        if self.trajectory_t >= self.trajectory_T:
            # Desactivar la trayectoria actual
            self.trajectory_active = False
            
            # Si hay una secuencia activa, intentar pasar al siguiente paso
            if self.sequence_active:
                self.current_step += 1
                
                # Verificar si quedan más pasos
                if self.current_step < len(self.sequence_steps):
                    # Obtener el siguiente paso y su duración
                    pose_idx = self.sequence_steps[self.current_step]
                    duration = self.sequence_durations[self.current_step]
                    
                    # Iniciar la siguiente transición
                    # Usamos trajectory_target_pos como nueva posición inicial
                    self.start_trajectory(
                        self.trajectory_target_pos.copy(),
                        pose_idx,
                        duration
                    )
                else:
                    # No hay más pasos - secuencia completada
                    self.sequence_active = False
                    print("\n>>> ¡SALUDO COMPLETADO! <<<\n")
        
        return interpolated_pos


# =============================================================================
# CLASE: ENTORNO DE SIMULACIÓN DEL ROBOT HUMANOIDE
# =============================================================================

class HumanoidEnv:
    """
    Entorno de simulación para el robot humanoide Unitree G1.
    
    Descripción General:
    --------------------
    Esta clase encapsula toda la lógica necesaria para simular el robot G1
    en MuJoCo, incluyendo:
    
    1. FÍSICA: Carga del modelo, configuración de parámetros físicos
    2. CONTROL: Política de red neuronal + controlador PD
    3. INTERFAZ: Entrada de teclado y visualización 3D
    4. ANIMACIONES: Secuencia de saludo con los brazos
    
    Arquitectura del Sistema de Control:
    ------------------------------------
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        SISTEMA DE CONTROL                           │
    │                                                                     │
    │   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐  │
    │   │ Sensores │────▶│Observador│────▶│ Política │────▶│ Acciones │  │
    │   │  (IMU,   │     │  (obs)   │     │   (NN)   │     │(15 joints)│  │
    │   │ encoders)│     │          │     │          │     │          │  │
    │   └──────────┘     └──────────┘     └──────────┘     └─────┬────┘  │
    │                                                            │       │
    │                                                            ▼       │
    │   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐  │
    │   │ Torques  │◀────│Control PD│◀────│ Posición │◀────│  Escalar │  │
    │   │ (23 DOF) │     │          │     │ Objetivo │     │ + Default│  │
    │   └────┬─────┘     └──────────┘     └──────────┘     └──────────┘  │
    │        │                                                           │
    │        ▼                                                           │
    │   ┌──────────┐                      ┌──────────┐                   │
    │   │  MuJoCo  │─────────────────────▶│  Estado  │────▶ (loop)      │
    │   │  Física  │                      │  Robot   │                   │
    │   └──────────┘                      └──────────┘                   │
    └─────────────────────────────────────────────────────────────────────┘
    
    Control PD (Proporcional-Derivativo):
    -------------------------------------
    El control PD es un algoritmo clásico que calcula el torque necesario
    para mover una articulación a una posición objetivo:
    
    torque = Kp * (objetivo - actual) - Kd * velocidad
    
    - Kp (Proporcional/Stiffness): Qué tan fuerte "empuja" hacia el objetivo
      - Valor alto: Respuesta rápida pero puede oscilar
      - Valor bajo: Respuesta lenta pero más suave
    
    - Kd (Derivativo/Damping): Qué tan fuerte frena el movimiento
      - Valor alto: Movimiento muy amortiguado, puede ser lento
      - Valor bajo: Puede oscilar o vibrar
    
    Parámetros del Robot G1:
    ------------------------
    - Altura: ~1.27m (parado)
    - Peso: ~35kg
    - DOFs: 23 (12 piernas + 3 cintura + 8 brazos)
    - Frecuencia de control: 50Hz (cada 20ms)
    - Frecuencia de física: 500Hz (cada 2ms)
    
    Atributos Principales:
    ----------------------
    model : mujoco.MjModel
        Modelo físico del robot cargado desde el archivo XML.
        Contiene la geometría, masas, límites de articulaciones, etc.
    
    data : mujoco.MjData
        Estado actual de la simulación.
        Contiene posiciones, velocidades, fuerzas, etc.
    
    viewer : MujocoViewer
        Visualizador 3D de la simulación.
        Permite ver el robot y controlar la cámara.
    
    policy_jit : torch.jit.ScriptModule
        Red neuronal pre-entrenada que genera comandos de control.
        Entrada: Observaciones del robot
        Salida: Acciones para 15 articulaciones (piernas + cintura)
    
    wave_controller : WaveController
        Controlador para la secuencia de saludo.
        Gestiona las animaciones de los brazos.
    
    Ejemplo de Uso:
    ---------------
    >>> # Cargar política pre-entrenada
    >>> policy = torch.jit.load('policy.pt')
    >>> 
    >>> # Crear entorno
    >>> env = HumanoidEnv(policy_jit=policy, robot_type="g1", device="cuda")
    >>> 
    >>> # Ejecutar simulación
    >>> env.run()  # Bucle principal hasta ESC o timeout
    """
    
    def __init__(self, policy_jit, robot_type="g1", device="cuda"):
        """
        Inicializa el entorno de simulación.
        
        Parámetros:
        -----------
        policy_jit : torch.jit.ScriptModule
            Modelo de política pre-entrenado (red neuronal compilada con TorchScript).
            Se espera que tenga el método forward(obs, extra_hist) -> actions.
        
        robot_type : str, opcional
            Tipo de robot a simular. Actualmente solo se soporta "g1".
            Default: "g1"
        
        device : str, opcional
            Dispositivo para ejecutar la política.
            Opciones: "cuda" (GPU) o "cpu"
            Default: "cuda"
        
        Raises:
        -------
        ValueError
            Si robot_type no es soportado.
        
        Proceso de Inicialización:
        --------------------------
        1. Configurar parámetros del robot (ganancias PD, límites, etc.)
        2. Cargar modelo de MuJoCo
        3. Inicializar visualizador y controles
        4. Configurar buffers de observación
        5. Cargar modelos de ML (política y adaptador)
        6. Inicializar controlador de saludo
        """
        # Guardar parámetros básicos
        self.robot_type = robot_type
        self.device = device

        # =====================================================================
        # CONFIGURACIÓN ESPECÍFICA DEL ROBOT G1
        # =====================================================================
        
        if robot_type == "g1":
            # Ruta al archivo XML del modelo MuJoCo
            model_path = MODEL_PATH
            
            # -----------------------------------------------------------------
            # GANANCIAS DEL CONTROLADOR PD
            # -----------------------------------------------------------------
            # Estos valores fueron ajustados experimentalmente para el G1.
            # 
            # Principios de ajuste:
            # - Piernas: Alta stiffness para soportar peso y mantener balance
            # - Cintura: Muy alta stiffness para estabilidad del torso
            # - Brazos: Baja stiffness para movimientos suaves y seguros
            
            self.stiffness = np.array([
                # Pierna izquierda (Kp)
                # hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                150, 150, 150, 300, 80, 20,
                
                # Pierna derecha (Kp)
                150, 150, 150, 300, 80, 20,
                
                # Cintura (Kp) - Alta para mantener torso estable
                400, 400, 400,
                
                # Brazo izquierdo (Kp) - Baja para movimientos suaves
                80, 80, 40, 60,
                
                # Brazo derecho (Kp)
                80, 80, 40, 60,
            ])
            
            self.damping = np.array([
                # Pierna izquierda (Kd)
                2, 2, 2, 4, 2, 1,
                
                # Pierna derecha (Kd)
                2, 2, 2, 4, 2, 1,
                
                # Cintura (Kd) - Alto para evitar oscilaciones
                15, 15, 15,
                
                # Brazo izquierdo (Kd)
                2, 2, 1, 1,
                
                # Brazo derecho (Kd)
                2, 2, 1, 1,
            ])
            
            # -----------------------------------------------------------------
            # DIMENSIONES Y CONFIGURACIÓN
            # -----------------------------------------------------------------
            
            self.num_actions = 15    # Salidas de la política (piernas + cintura)
            self.num_dofs = 23       # Total de grados de libertad
            
            # Posiciones angulares por defecto (pose "home")
            # Estas posiciones definen cómo se ve el robot cuando está parado
            self.default_dof_pos = np.array([
                # Pierna izquierda
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # Ligeramente flexionada
                
                # Pierna derecha
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                
                # Cintura
                0.0, 0.0, 0.0,  # Sin rotación
                
                # Brazo izquierdo
                0.5, 0.0, 0.2, 0.3,  # Relajado al costado
                
                # Brazo derecho
                0.5, 0.0, -0.2, 0.3,
            ])
            
            # Límites de torque (para proteger los motores)
            # Estos valores representan el torque máximo que cada motor puede aplicar
            self.torque_limits = np.array([
                # Pierna izquierda (Nm)
                88, 139, 88, 139, 50, 50,
                
                # Pierna derecha (Nm)
                88, 139, 88, 139, 50, 50,
                
                # Cintura (Nm)
                88, 50, 50,
                
                # Brazo izquierdo (Nm) - Más bajos por seguridad
                25, 25, 25, 25,
                
                # Brazo derecho (Nm)
                25, 25, 25, 25,
            ])
            
            # Nombres de las articulaciones (para referencia y debugging)
            self.dof_names = [
                "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
                "left_knee", "left_ankle_pitch", "left_ankle_roll",
                "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
                "right_knee", "right_ankle_pitch", "right_ankle_roll",
                "waist_yaw", "waist_roll", "waist_pitch",
                "left_shoulder_pitch", "left_shoulder_roll",
                "left_shoulder_yaw", "left_elbow",
                "right_shoulder_pitch", "right_shoulder_roll",
                "right_shoulder_yaw", "right_elbow"
            ]
        else:
            raise ValueError(f"Tipo de robot '{robot_type}' no soportado. Use 'g1'.")

        # =====================================================================
        # CONFIGURACIÓN DE LA SIMULACIÓN
        # =====================================================================
        
        self.sim_duration = 100 * 20.0  # Duración total: 2000 segundos
        self.sim_dt = 0.002              # Paso de física: 2ms (500Hz)
        self.sim_decimation = 10         # Ejecutar política cada 10 pasos
        self.control_dt = self.sim_dt * self.sim_decimation  # 20ms (50Hz)

        # =====================================================================
        # INICIALIZACIÓN DE MUJOCO
        # =====================================================================
        
        # Cargar el modelo desde el archivo XML
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt  # Establecer paso de tiempo
        
        # Crear estructura de datos para el estado de la simulación
        self.data = mujoco.MjData(self.model)
        
        # Resetear a la pose inicial definida en el XML (keyframe 0)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Hacer un paso de simulación para estabilizar
        mujoco.mj_step(self.model, self.data)
        
        # Crear el visualizador 3D
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        # =====================================================================
        # CONFIGURACIÓN DEL VISUALIZADOR Y CONTROLES
        # =====================================================================
        
        # Array de comandos del usuario (8 elementos)
        # Ver documentación de _key_callback para detalles
        self.viewer.commands = np.zeros(8, dtype=np.float32)
        
        # Configuración de la cámara
        self.viewer.cam.distance = 2.5   # Distancia al robot (metros)
        self.viewer.cam.elevation = 0.0  # Ángulo vertical (grados)
        
        # Flags para controlar secuencias de brazos
        self.viewer.start_wave_sequence = False   # Activar saludo
        self.viewer.reset_to_default_pose = False # Resetear pose
        
        # Asignar la función callback al visualizador
        # types.MethodType convierte una función en un método de instancia
        self.viewer._key_callback = types.MethodType(_key_callback, self.viewer)
        
        # Registrar el callback con GLFW
        glfw.set_key_callback(self.viewer.window, self.viewer._key_callback)

        # =====================================================================
        # VARIABLES DE ESTADO DEL CONTROL
        # =====================================================================
        
        self.action_scale = 0.25     # Factor de escala para las acciones
        self.scales_ang_vel = 0.25   # Escala para velocidades angulares
        self.scales_dof_vel = 0.05   # Escala para velocidades de joints

        # Dimensiones para la construcción de observaciones
        self.nj = 23                 # Número de joints
        self.n_priv = 3              # Dimensión de información privilegiada
        self.n_proprio = 3 + 2 + 2 + 23*3 + 2 + 15  # Dim. propiocepción
        self.history_len = 10        # Longitud del historial de observaciones
        self.extra_history_len = 25  # Historial extra para la política
        self._n_demo_dof = 8         # DOFs de demostración (brazos)

        # Buffers de estado actual
        self.dof_pos = np.zeros(self.nj, dtype=np.float32)  # Posiciones
        self.dof_vel = np.zeros(self.nj, dtype=np.float32)  # Velocidades
        self.quat = np.zeros(4, dtype=np.float32)           # Orientación
        self.ang_vel = np.zeros(3, dtype=np.float32)        # Vel. angular
        self.last_action = np.zeros(self.nj)                # Última acción

        # Template para observación de demostración
        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3,))
        self.demo_obs_template[:self._n_demo_dof] = self.default_dof_pos[15:]
        self.demo_obs_template[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75

        # Control de orientación
        self.target_yaw = 0.0            # Yaw objetivo
        self._in_place_stand_flag = True # ¿Está parado en el lugar?

        # Control del ciclo de marcha
        # Estos valores sincronizan el movimiento de los pies
        self.gait_cycle = np.array([0.25, 0.25])  # Fase de cada pie
        self.gait_freq = 1.3                       # Frecuencia de paso

        # =====================================================================
        # BUFFERS DE HISTORIAL
        # =====================================================================
        # La política usa historiales de observaciones pasadas para
        # entender el contexto temporal (velocidades, aceleraciones, etc.)
        
        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        
        # Inicializar con ceros
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for _ in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))

        # =====================================================================
        # CARGAR MODELOS DE APRENDIZAJE AUTOMÁTICO
        # =====================================================================
        
        # Política principal (genera acciones de locomoción)
        self.policy_jit = policy_jit

        # Adaptador (procesa información de postura del torso)
        self.adapter = torch.jit.load(ADAPTER_PATH, map_location=self.device)
        self.adapter.eval()  # Modo evaluación (desactiva dropout, etc.)
        for param in self.adapter.parameters():
            param.requires_grad = False  # No necesitamos gradientes

        # Cargar estadísticas de normalización para el adaptador
        norm_stats = torch.load(NORM_STATS_PATH, weights_only=False)
        
        # Convertir a tensores en el dispositivo correcto
        self.input_mean = torch.tensor(
            norm_stats['input_mean'], device=self.device, dtype=torch.float32
        )
        self.input_std = torch.tensor(
            norm_stats['input_std'], device=self.device, dtype=torch.float32
        )
        self.output_mean = torch.tensor(
            norm_stats['output_mean'], device=self.device, dtype=torch.float32
        )
        self.output_std = torch.tensor(
            norm_stats['output_std'], device=self.device, dtype=torch.float32
        )

        # Buffers pre-asignados para entrada/salida del adaptador
        self.adapter_input = torch
                # Buffers pre-asignados para entrada/salida del adaptador
        # Pre-asignar memoria mejora el rendimiento al evitar allocaciones repetidas
        self.adapter_input = torch.zeros(
            (1, 8 + 4),  # 8 DOFs de brazos + 4 parámetros de postura
            device=self.device,
            dtype=torch.float32
        )
        self.adapter_output = torch.zeros(
            (1, 15),  # 15 valores de salida
            device=self.device,
            dtype=torch.float32
        )

        # =====================================================================
        # INICIALIZAR CONTROLADOR DE SALUDO
        # =====================================================================
        
        self.wave_controller = WaveController(self.num_dofs, self.default_dof_pos)
        self.use_wave_control = False  # Flag para activar control de saludo
        
        # Mostrar instrucciones al usuario
        self._print_instructions()
    
    def _print_instructions(self):
        """
        Imprime las instrucciones de control en la consola.
        
        Esta función se llama una vez al inicio del programa para
        informar al usuario sobre los controles disponibles.
        
        El formato usa caracteres ASCII para crear una presentación
        clara y organizada de la información.
        """
        print("\n" + "=" * 60)
        print("CONTROLES DEL ROBOT HUMANOIDE G1")
        print("=" * 60)
        
        print("\n--- LOCOMOCIÓN ---")
        print("W / S     : Velocidad adelante / atrás")
        print("A / D     : Girar izquierda / derecha")
        print("Q / E     : Movimiento lateral izquierda / derecha")
        print("Z / X     : Subir / bajar altura del cuerpo")
        
        print("\n--- CONTROL DEL TORSO ---")
        print("U / J     : Rotación del torso (yaw)")
        print("I / K     : Inclinación del torso (pitch)")
        print("O / L     : Balanceo del torso (roll)")
        
        print("\n--- SECUENCIA DE SALUDO ---")
        print("G         : Iniciar secuencia de saludo")
        print("R         : Resetear brazos a posición inicial")
        
        print("\n--- SISTEMA ---")
        print("ESC       : Salir de la simulación")
        
        print("=" * 60 + "\n")

    def extract_data(self):
        """
        Extrae los datos del estado actual del robot desde MuJoCo.
        
        Esta función lee los sensores virtuales de la simulación y
        almacena los valores en los atributos de la clase para su
        uso posterior en la construcción de observaciones.
        
        Datos Extraídos:
        ----------------
        1. dof_pos : Posiciones angulares de todas las articulaciones
           - Fuente: data.qpos (vector de posiciones generalizadas)
           - Unidades: radianes
           - Shape: (23,)
        
        2. dof_vel : Velocidades angulares de todas las articulaciones
           - Fuente: data.qvel (vector de velocidades generalizadas)
           - Unidades: radianes/segundo
           - Shape: (23,)
        
        3. quat : Orientación del cuerpo principal en cuaternión
           - Fuente: Sensor IMU 'orientation' definido en el XML
           - Formato: [w, x, y, z]
           - Normalizado: |q| = 1
           - Shape: (4,)
        
        4. ang_vel : Velocidad angular del cuerpo principal
           - Fuente: Sensor IMU 'angular-velocity' definido en el XML
           - Unidades: radianes/segundo
           - Frame: Cuerpo local
           - Shape: (3,) = [wx, wy, wz]
        
        Nota sobre Índices:
        -------------------
        Usamos [-self.num_dofs:] porque data.qpos incluye:
        - Primeros 7 elementos: Posición y orientación del cuerpo base
          [x, y, z, qw, qx, qy, qz]
        - Siguientes 23 elementos: Posiciones de las articulaciones
        
        Al tomar los últimos 23 elementos, obtenemos solo las articulaciones.
        """
        # Extraer posiciones de articulaciones (últimos 23 elementos de qpos)
        self.dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        
        # Extraer velocidades de articulaciones (últimos 23 elementos de qvel)
        self.dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        
        # Leer orientación del sensor IMU
        # El sensor 'orientation' está definido en el archivo XML del robot
        self.quat = self.data.sensor('orientation').data.astype(np.float32)
        
        # Leer velocidad angular del sensor IMU
        self.ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)

    def get_observation(self):
        """
        Construye el vector de observación completo para la política.
        
        La observación es el "input" que recibe la red neuronal para
        decidir qué acciones tomar. Contiene toda la información
        relevante sobre el estado actual del robot y los comandos del usuario.
        
        Estructura de la Observación:
        -----------------------------
        La observación final es la concatenación de varios componentes:
        
        ┌─────────────────────────────────────────────────────────────────┐
        │                    VECTOR DE OBSERVACIÓN                        │
        ├─────────────────────────────────────────────────────────────────┤
        │ obs_prop (propiocepción):                                       │
        │   ├── ang_vel (3)        : Velocidad angular del cuerpo         │
        │   ├── rpy[:2] (2)        : Roll y Pitch del cuerpo              │
        │   ├── sin/cos(dyaw) (2)  : Error de orientación (yaw)           │
        │   ├── dof_pos_rel (23)   : Posiciones relativas a default       │
        │   ├── dof_vel (23)       : Velocidades de articulaciones        │
        │   ├── last_action (23)   : Acciones del paso anterior           │
        │   ├── gait_obs (2)       : Fase del ciclo de marcha             │
        │   └── adapter_out (15)   : Salida del adaptador                 │
        │                                                                 │
        │ obs_demo (demostración):                                        │
        │   ├── arm_pos (8)        : Posiciones de brazos                 │
        │   ├── vel_cmd (2)        : Comandos de velocidad                │
        │   ├── torso_cmd (3)      : Comandos del torso                   │
        │   └── height (3)         : Altura objetivo                      │
        │                                                                 │
        │ obs_priv (privilegiada):                                        │
        │   └── zeros (3)          : Info privilegiada (no usada aquí)    │
        │                                                                 │
        │ obs_hist (historial):                                           │
        │   └── history (n_proprio * history_len)                         │
        └─────────────────────────────────────────────────────────────────┘
        
        ¿Por qué esta estructura?
        -------------------------
        - Propiocepción: Lo que el robot "siente" de su propio cuerpo
        - Demostración: Lo que el usuario quiere que haga
        - Privilegiada: Información solo disponible en simulación
        - Historial: Contexto temporal para entender el movimiento
        
        Retorna:
        --------
        numpy.ndarray
            Vector de observación concatenado.
            Shape: (n_proprio + n_demo + n_priv + n_proprio * history_len,)
        
        Notas:
        ------
        - Todos los valores se escalan para estar en rangos similares
        - El historial permite a la política "recordar" estados pasados
        - El adaptador procesa información del torso para mejor control
        """
        # -----------------------------------------------------------------
        # PASO 1: Convertir orientación a ángulos de Euler
        # -----------------------------------------------------------------
        rpy = quatToEuler(self.quat)  # [roll, pitch, yaw]

        # -----------------------------------------------------------------
        # PASO 2: Calcular error de orientación (yaw)
        # -----------------------------------------------------------------
        # El target_yaw viene de los comandos del usuario (teclas A/D)
        self.target_yaw = self.viewer.commands[1]
        
        # Calcular diferencia angular
        dyaw = rpy[2] - self.target_yaw
        
        # Normalizar a rango [-π, π]
        # Esto maneja correctamente el "wrap-around" (ej: -179° a 179° = 2°)
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        
        # Si está parado en el lugar, ignorar error de yaw
        # Esto evita que el robot intente girar cuando no está caminando
        if self._in_place_stand_flag:
            dyaw = 0.0

        # -----------------------------------------------------------------
        # PASO 3: Calcular fase del ciclo de marcha
        # -----------------------------------------------------------------
        # gait_cycle contiene la fase [0, 1] de cada pierna
        # Convertimos a seno para tener valores suaves en [-1, 1]
        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        # -----------------------------------------------------------------
        # PASO 4: Preparar y ejecutar el adaptador
        # -----------------------------------------------------------------
        # El adaptador procesa información del torso y brazos
        
        # Construir entrada: [altura, torso_yaw, torso_pitch, torso_roll, arm_positions]
        self.adapter_input = np.concatenate([np.zeros(4), self.dof_pos[15:]])
        
        # Llenar los primeros 4 valores con comandos de postura
        self.adapter_input[0] = 0.75 + self.viewer.commands[3]  # Altura
        self.adapter_input[1] = self.viewer.commands[4]          # Torso yaw
        self.adapter_input[2] = self.viewer.commands[5]          # Torso pitch
        self.adapter_input[3] = self.viewer.commands[6]          # Torso roll

        # Convertir a tensor de PyTorch
        self.adapter_input = torch.tensor(self.adapter_input).to(
            self.device, dtype=torch.float32
        ).unsqueeze(0)  # Añadir dimensión de batch

        # Normalizar entrada usando estadísticas pre-calculadas
        # z = (x - mean) / std
        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        
        # Ejecutar el adaptador
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        
        # Desnormalizar salida
        # x = z * std + mean
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        # -----------------------------------------------------------------
        # PASO 5: Construir observación propioceptiva
        # -----------------------------------------------------------------
        obs_prop = np.concatenate([
            # Velocidad angular del cuerpo (escalada)
            self.ang_vel * self.scales_ang_vel,
            
            # Roll y Pitch del cuerpo
            rpy[:2],
            
            # Error de yaw codificado como sin/cos
            # Esto evita discontinuidades en ±180°
            (np.sin(dyaw), np.cos(dyaw)),
            
            # Posiciones de articulaciones relativas a la pose default
            (self.dof_pos - self.default_dof_pos),
            
            # Velocidades de articulaciones (escaladas)
            self.dof_vel * self.scales_dof_vel,
            
            # Acciones del paso anterior
            self.last_action,
            
            # Fase del ciclo de marcha
            gait_obs,
            
            # Salida del adaptador
            self.adapter_output.cpu().numpy().squeeze(),
        ])

        # -----------------------------------------------------------------
        # PASO 6: Información privilegiada (no disponible en robot real)
        # -----------------------------------------------------------------
        # En simulación podríamos incluir información como fricción del suelo,
        # pero aquí usamos ceros
        obs_priv = np.zeros((self.n_priv,))
        
        # -----------------------------------------------------------------
        # PASO 7: Historial de observaciones
        # -----------------------------------------------------------------
        obs_hist = np.array(self.proprio_history_buf).flatten()

        # -----------------------------------------------------------------
        # PASO 8: Observación de demostración (comandos del usuario)
        # -----------------------------------------------------------------
        obs_demo = self.demo_obs_template.copy()
        
        # Posiciones actuales de los brazos
        obs_demo[:self._n_demo_dof] = self.dof_pos[15:]
        
        # Comandos de velocidad
        obs_demo[self._n_demo_dof] = self.viewer.commands[0]      # vx
        obs_demo[self._n_demo_dof + 1] = self.viewer.commands[2]  # vy
        
        # Detectar si está parado en el lugar (velocidad muy baja)
        self._in_place_stand_flag = np.abs(self.viewer.commands[0]) < 0.1
        
        # Comandos del torso
        obs_demo[self._n_demo_dof + 3] = self.viewer.commands[4]  # yaw
        obs_demo[self._n_demo_dof + 4] = self.viewer.commands[5]  # pitch
        obs_demo[self._n_demo_dof + 5] = self.viewer.commands[6]  # roll
        
        # Altura objetivo
        obs_demo[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75 + self.viewer.commands[3]

        # -----------------------------------------------------------------
        # PASO 9: Actualizar historiales
        # -----------------------------------------------------------------
        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)

        # -----------------------------------------------------------------
        # PASO 10: Concatenar y retornar
        # -----------------------------------------------------------------
        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))

    def _handle_wave_commands(self):
        """
        Procesa los comandos de saludo recibidos del teclado.
        
        Esta función verifica los flags establecidos por el callback
        de teclado y activa las acciones correspondientes en el
        controlador de saludo.
        
        Flags procesados:
        -----------------
        start_wave_sequence : bool
            Si True, inicia la secuencia completa de saludo.
            Se activa con la tecla 'G'.
        
        reset_to_default_pose : bool
            Si True, resetea los brazos a la posición neutral.
            Se activa con la tecla 'R'.
        
        Nota:
        -----
        Los flags se resetean a False después de procesarlos para
        evitar activaciones repetidas.
        """
        # Verificar si se solicitó iniciar la secuencia de saludo
        if self.viewer.start_wave_sequence:
            # Resetear el flag para evitar re-activación
            self.viewer.start_wave_sequence = False
            
            # Activar el control de saludo
            self.use_wave_control = True
            
            # Iniciar la secuencia con las posiciones actuales
            self.wave_controller.start_sequence(self.dof_pos)
        
        # Verificar si se solicitó resetear la pose
        if self.viewer.reset_to_default_pose:
            # Resetear el flag
            self.viewer.reset_to_default_pose = False
            
            # Activar el control de saludo (para la transición)
            self.use_wave_control = True
            
            # Ir a la pose por defecto
            self.wave_controller.reset_to_default(self.dof_pos)

    def run(self):
        """
        Bucle principal de la simulación.
        
        Este método ejecuta el ciclo completo de simulación:
        
        ┌─────────────────────────────────────────────────────────────────┐
        │                      BUCLE PRINCIPAL                            │
        │                                                                 │
        │  Para cada paso de simulación (i = 0, 1, 2, ...):              │
        │                                                                 │
        │  1. EXTRAER ESTADO                                              │
        │     └── Leer sensores de MuJoCo                                │
        │                                                                 │
        │  2. CADA 10 PASOS (50 Hz):                                     │
        │     ├── Procesar comandos de teclado                           │
        │     ├── Construir observación                                  │
        │     ├── Ejecutar política (red neuronal)                       │
        │     ├── Calcular posiciones objetivo                           │
        │     ├── Actualizar ciclo de marcha                             │
        │     └── Renderizar visualización                               │
        │                                                                 │
        │  3. CADA PASO (500 Hz):                                        │
        │     ├── Calcular torques con control PD                        │
        │     ├── Aplicar torques al robot                               │
        │     └── Avanzar física de MuJoCo                               │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘
        
        Frecuencias:
        ------------
        - Física: 500 Hz (cada 2ms)
          - Control PD necesita alta frecuencia para estabilidad
          - MuJoCo simula con precisión a esta velocidad
        
        - Política: 50 Hz (cada 20ms)
          - Suficiente para planificación de movimientos
          - Reduce carga computacional
          - Coincide con frecuencia típica de robots reales
        
        - Renderizado: 50 Hz (cada 20ms)
          - Suficiente para visualización fluida
          - Sincronizado con la política
        
        Control PD:
        -----------
        El control PD se ejecuta a 500 Hz para mantener estabilidad.
        La fórmula es:
        
        torque = Kp * (objetivo - actual) - Kd * velocidad
        
        Donde:
        - Kp = stiffness (rigidez)
        - Kd = damping (amortiguamiento)
        - objetivo = posición deseada
        - actual = posición medida
        - velocidad = velocidad angular actual
        
        Terminación:
        ------------
        El bucle termina cuando:
        - Se alcanza sim_duration (timeout)
        - El usuario presiona ESC (cierra ventana GLFW)
        """
        # Calcular número total de pasos de simulación
        total_steps = int(self.sim_duration / self.sim_dt)
        
        # Variable para almacenar el objetivo de posición
        # Se declara aquí para que persista entre iteraciones
        pd_target = self.default_dof_pos.copy()
        
        # =====================================================================
        # BUCLE PRINCIPAL DE SIMULACIÓN
        # =====================================================================
        
        for i in range(total_steps):
            
            # =================================================================
            # PASO 1: Extraer estado actual del robot
            # =================================================================
            # Leer posiciones, velocidades y orientación desde MuJoCo
            self.extract_data()

            # =================================================================
            # PASO 2: Ejecutar control de alto nivel (cada 10 pasos = 50 Hz)
            # =================================================================
            
            if i % self.sim_decimation == 0:
                
                # -------------------------------------------------------------
                # 2.1: Procesar comandos de saludo del teclado
                # -------------------------------------------------------------
                self._handle_wave_commands()
                
                # -------------------------------------------------------------
                # 2.2: Construir vector de observación
                # -------------------------------------------------------------
                obs = self.get_observation()
                
                # Convertir a tensor de PyTorch para la política
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

                # -------------------------------------------------------------
                # 2.3: Ejecutar la política (red neuronal)
                # -------------------------------------------------------------
                # Usamos torch.no_grad() porque no necesitamos gradientes
                # (no estamos entrenando, solo evaluando)
                
                with torch.no_grad():
                    # Preparar historial extra como tensor
                    extra_hist = torch.tensor(
                        np.array(self.extra_history_buf).flatten().copy(),
                        dtype=torch.float
                    ).view(1, -1).to(self.device)
                    
                    # Ejecutar la política
                    # Entrada: observación + historial
                    # Salida: acciones para 15 articulaciones
                    raw_action = self.policy_jit(
                        obs_tensor, extra_hist
                    ).cpu().numpy().squeeze()

                # -------------------------------------------------------------
                # 2.4: Post-procesar acciones
                # -------------------------------------------------------------
                
                # Limitar acciones a un rango razonable
                # Esto evita comandos extremos que podrían dañar el robot
                raw_action = np.clip(raw_action, -40.0, 40.0)
                
                # Guardar las acciones para incluirlas en la próxima observación
                # La política necesita saber qué comandó en el paso anterior
                self.last_action = np.concatenate([
                    raw_action.copy(),
                    # Para los brazos, usamos la posición actual como "acción"
                    (self.dof_pos - self.default_dof_pos)[15:] / self.action_scale
                ])
                
                # Escalar acciones
                # La política produce valores en un rango normalizado
                # Multiplicamos por action_scale para obtener radianes
                scaled_actions = raw_action * self.action_scale

                # -------------------------------------------------------------
                # 2.5: Calcular posiciones objetivo
                # -------------------------------------------------------------
                
                # Las acciones de la política son INCREMENTOS sobre la pose default
                # pd_target = default + acción escalada
                
                # Para las piernas y cintura (índices 0-14):
                # Usamos las acciones de la política
                
                # Para los brazos (índices 15-22):
                # Usamos ceros (se sobrescribirán si hay saludo activo)
                
                pd_target = np.concatenate([
                    scaled_actions,     # 15 acciones para piernas + cintura
                    np.zeros(8)         # 8 valores para brazos (placeholder)
                ]) + self.default_dof_pos
                
                # -------------------------------------------------------------
                # 2.6: Aplicar control de saludo si está activo
                # -------------------------------------------------------------
                
                if self.use_wave_control:
                    # Actualizar el controlador de saludo
                    wave_target = self.wave_controller.update(
                        self.dof_pos,      # Posición actual
                        self.control_dt    # Tiempo transcurrido
                    )
                    
                    if wave_target is not None:
                        # Sobrescribir posiciones de cintura y brazos
                        # Índices 12-22 = cintura (3) + brazos (8)
                        pd_target[12:] = wave_target[12:]
                    
                    # Verificar si el saludo terminó
                    if not self.wave_controller.trajectory_active and \
                       not self.wave_controller.sequence_active:
                        # Desactivar el control de saludo
                        self.use_wave_control = False

                # -------------------------------------------------------------
                # 2.7: Actualizar ciclo de marcha
                # -------------------------------------------------------------
                # El ciclo de marcha sincroniza el movimiento de los pies
                # para producir un caminar natural
                
                # Avanzar la fase del ciclo
                self.gait_cycle = np.remainder(
                    self.gait_cycle + self.control_dt * self.gait_freq,
                    1.0  # Mantener en rango [0, 1]
                )
                
                # Sincronización cuando está parado
                # Ambos pies deben estar en la misma fase (0.25)
                if self._in_place_stand_flag and (
                    (np.abs(self.gait_cycle[0] - 0.25) < 0.05) or
                    (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
                ):
                    self.gait_cycle = np.array([0.25, 0.25])
                
                # Desfase cuando camina
                # Los pies deben estar en fases opuestas (0.25 y 0.75)
                if (not self._in_place_stand_flag) and (
                    (np.abs(self.gait_cycle[0] - 0.25) < 0.05) and
                    (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
                ):
                    self.gait_cycle = np.array([0.25, 0.75])

                # -------------------------------------------------------------
                # 2.8: Actualizar cámara y renderizar
                # -------------------------------------------------------------
                
                # La cámara sigue al robot
                # data.qpos[:3] contiene la posición [x, y, z] del cuerpo base
                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                
                # Renderizar el frame
                self.viewer.render()

            # =================================================================
            # PASO 3: Aplicar control PD (cada paso = 500 Hz)
            # =================================================================
            # Este control de bajo nivel se ejecuta a alta frecuencia
            # para mantener el robot estable
            
            # Calcular error de posición
            position_error = pd_target - self.dof_pos
            
            # Calcular torque usando control PD
            # torque = Kp * error_posición - Kd * velocidad
            torque = position_error * self.stiffness - self.dof_vel * self.damping
            
            # Limitar torques a los límites físicos de los motores
            # Esto previene comandos que dañarían el hardware real
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            
            # Aplicar torques al robot en MuJoCo
            self.data.ctrl = torque

            # =================================================================
            # PASO 4: Avanzar la simulación física
            # =================================================================
            # MuJoCo calcula el nuevo estado del robot basándose en:
            # - Estado actual (posiciones, velocidades)
            # - Torques aplicados
            # - Fuerzas externas (gravedad, contactos)
            
            mujoco.mj_step(self.model, self.data)

        # =====================================================================
        # LIMPIEZA AL TERMINAR
        # =====================================================================
        
        # Cerrar el visualizador y liberar recursos
        self.viewer.close()
        print("\nSimulación terminada.")


# =============================================================================
# PUNTO DE ENTRADA DEL PROGRAMA
# =============================================================================

if __name__ == "__main__":
    """
    Punto de entrada principal del programa.
    
    Este bloque solo se ejecuta cuando el archivo se corre directamente
    (no cuando se importa como módulo).
    
    Proceso:
    --------
    1. Detectar dispositivo disponible (GPU o CPU)
    2. Cargar la política pre-entrenada
    3. Crear el entorno de simulación
    4. Ejecutar el bucle principal
    
    Requisitos:
    -----------
    - Archivo de política en POLICY_PATH
    - Archivo de modelo en MODEL_PATH
    - Archivos del adaptador en ADAPTER_PATH y NORM_STATS_PATH
    
    Uso:
    ----
    $ python nombre_del_archivo.py
    
    El programa mostrará las instrucciones de control y abrirá
    una ventana de visualización 3D del robot.
    """
    
    # =========================================================================
    # CONFIGURACIÓN
    # =========================================================================
    
    robot = "g1"  # Tipo de robot (actualmente solo G1 soportado)
    
    # Detectar dispositivo: usar GPU si está disponible, sino CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo seleccionado: {device}")
    
    if device == "cuda":
        # Mostrar información de la GPU
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # =========================================================================
    # CARGAR POLÍTICA PRE-ENTRENADA
    # =========================================================================
    
    print(f"\nCargando política desde: {POLICY_PATH}")
    
    try:
        # torch.jit.load carga un modelo compilado con TorchScript
        # map_location asegura que se cargue en el dispositivo correcto
        policy_jit = torch.jit.load(POLICY_PATH, map_location=device)
        print("Política cargada exitosamente.")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de política: {POLICY_PATH}")
        print("Asegúrate de que la ruta sea correcta.")
        exit(1)
    except Exception as e:
        print(f"ERROR al cargar la política: {e}")
        exit(1)
    
    # =========================================================================
    # CREAR ENTORNO DE SIMULACIÓN
    # =========================================================================
    
    print("\nInicializando entorno de simulación...")
    
    try:
        env = HumanoidEnv(
            policy_jit=policy_jit,
            robot_type=robot,
            device=device
        )
        print("Entorno inicializado exitosamente.")
    except FileNotFoundError as e:
        print(f"ERROR: No se encontró un archivo requerido: {e}")
        exit(1)
    except Exception as e:
        print(f"ERROR al inicializar el entorno: {e}")
        exit(1)
    
    # =========================================================================
    # EJECUTAR SIMULACIÓN
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("INICIANDO SIMULACIÓN")
    print("=" * 60)
    print("\nUsa las teclas mostradas arriba para controlar el robot.")
    print("Presiona ESC para salir.\n")
    
    try:
        # Ejecutar el bucle principal
        env.run()
    except KeyboardInterrupt:
        print("\n\nSimulación interrumpida por el usuario (Ctrl+C)")
    except Exception as e:
        print(f"\nERROR durante la simulación: {e}")
        raise  # Re-lanzar para ver el traceback completo
    
    print("\n¡Hasta luego!")