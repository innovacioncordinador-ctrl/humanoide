
import time
import math
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import sys



class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    kNotUsedJoint = 29


# ===========================================================
#        EJECUTOR DE TRAYECTORIAS PARA EL G1
# ===========================================================
class G1TrajectoryExecutor:
    def __init__(self, interface, id):

        ChannelFactoryInitialize(id, interface)
        self.control_dt = 0.02       # 50 Hz
        self.kp = 60.0               # Ganancia P genérica
        self.kd = 1.5                # Ganancia D genérica
        self.crc = CRC()

        # Variables de interpolación
        self.t = 0.0
        self.T = 3.0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update = False

        # Diccionario donde guardamos la postura objetivo
        self.target_pos = {}

        # -------------------------------
        # Articulaciones que SÍ controlamos
        # -------------------------------
        self.arm_joints = [
            G1JointIndex.LeftShoulderPitch, G1JointIndex.LeftShoulderRoll,
            G1JointIndex.LeftShoulderYaw,   G1JointIndex.LeftElbow,
            G1JointIndex.LeftWristRoll,     G1JointIndex.LeftWristPitch,
            G1JointIndex.LeftWristYaw,
            G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
            G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
            G1JointIndex.RightWristYaw,
            G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch
        ]

        # Articulaciones que solo mantenemos firmes
        self.rest_joints = [
            j for j in range(G1JointIndex.kNotUsedJoint)
            if j not in self.arm_joints
        ]

        # LISTA DE POSTURAS PREDEFINIDAS
        # (nombre, diccionario)
        self.pasos = [
            ("Paso 1 (inicial)", {
                G1JointIndex.LeftShoulderPitch: 0.28122639656066895,
                G1JointIndex.LeftShoulderRoll: 0.12417888641357422,
                G1JointIndex.LeftShoulderYaw: -0.001921,
                G1JointIndex.LeftElbow: 0.976243,
                G1JointIndex.LeftWristRoll: 0.167062,
                G1JointIndex.LeftWristPitch: -0.042408,
                G1JointIndex.LeftWristYaw: 0.037438,
                G1JointIndex.RightShoulderPitch: 0.280304,
                G1JointIndex.RightShoulderRoll: -0.125569,
                G1JointIndex.RightShoulderYaw: 0.038258,
                G1JointIndex.RightElbow: 0.980461,
                G1JointIndex.RightWristRoll: -0.129420,
                G1JointIndex.RightWristPitch: -0.006482,
                G1JointIndex.RightWristYaw: -0.017703,
                G1JointIndex.WaistYaw: 0.0,
                G1JointIndex.WaistRoll: 0.0,
                G1JointIndex.WaistPitch: 0.0
            }),

            ("Paso 2 (abrir para agarrar)", {
                G1JointIndex.LeftShoulderPitch: 0.0,
                G1JointIndex.LeftShoulderRoll: 0.35,
                G1JointIndex.LeftShoulderYaw: 0.4,
                G1JointIndex.LeftElbow: -0.1,
                G1JointIndex.LeftWristRoll: 0.0,
                G1JointIndex.LeftWristPitch: 0.0,
                G1JointIndex.LeftWristYaw: 0.0,
                G1JointIndex.RightShoulderPitch: 0.0,
                G1JointIndex.RightShoulderRoll: -0.35,
                G1JointIndex.RightShoulderYaw: -0.4,
                G1JointIndex.RightElbow: -0.1,
                G1JointIndex.RightWristRoll: 0.0,
                G1JointIndex.RightWristPitch: 0.0,
                G1JointIndex.RightWristYaw: 0.0,
                G1JointIndex.WaistYaw: 0.0,
                G1JointIndex.WaistRoll: 0.0,
                G1JointIndex.WaistPitch: 0.0
            }),

            ("Paso 3 (agarre)", {
                G1JointIndex.LeftShoulderPitch: 0.0021,
                G1JointIndex.LeftShoulderRoll: 0.1068,
                G1JointIndex.LeftShoulderYaw: -0.4033,
                G1JointIndex.LeftElbow: -0.1815,
                G1JointIndex.LeftWristRoll: -0.1584,
                G1JointIndex.LeftWristPitch: 0.0,
                G1JointIndex.LeftWristYaw: 0.4522,
                G1JointIndex.RightShoulderPitch: 0.0021,
                G1JointIndex.RightShoulderRoll: -0.1068,
                G1JointIndex.RightShoulderYaw: 0.4033,
                G1JointIndex.RightElbow: -0.1815,
                G1JointIndex.RightWristRoll: 0.1584,
                G1JointIndex.RightWristPitch: 0.0,
                G1JointIndex.RightWristYaw: -0.4522,
                G1JointIndex.WaistYaw: 0.0,
                G1JointIndex.WaistRoll: 0.0,
                G1JointIndex.WaistPitch: 0.0
            }),

            ("Paso 4 (dejar)", {
                G1JointIndex.LeftShoulderPitch: -0.584,
                G1JointIndex.LeftShoulderRoll: 0.1068,
                G1JointIndex.LeftShoulderYaw: 0.0262,
                G1JointIndex.LeftElbow: 0.429,
                G1JointIndex.LeftWristRoll: -0.1584,
                G1JointIndex.LeftWristPitch: 0.0,
                G1JointIndex.LeftWristYaw: 0.4522,
                G1JointIndex.RightShoulderPitch: -0.555,
                G1JointIndex.RightShoulderRoll: -0.1068,
                G1JointIndex.RightShoulderYaw: 0.183,
                G1JointIndex.RightElbow: 0.414,
                G1JointIndex.RightWristRoll: 0.1584,
                G1JointIndex.RightWristPitch: 0.0,
                G1JointIndex.RightWristYaw: -0.4522,
                G1JointIndex.WaistYaw: 0.0,
                G1JointIndex.WaistRoll: 0.0,
                G1JointIndex.WaistPitch: 0.0
            }),

            

        ]

    # ===========================================================
    #      INICIALIZACIÓN DE SUSCRIPCIÓN Y PUBLICACIÓN
    # ===========================================================
    def Init(self):
     
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()

        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self.LowStateHandler, 10)

    # ===========================================================
    #      INICIO DEL LOOP DE CONTROL (50 Hz)
    # ===========================================================
    def Start(self):
        
        self.thread = RecurrentThread(
            interval=self.control_dt,
            target=self.LowCmdWrite,
            name="arm_control"
        )

        while not self.first_update:
            time.sleep(0.1)

        self.thread.Start()

    # ===========================================================
    #      MANEJADOR DE ESTADO
    # ===========================================================
    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if not self.first_update:
            self.first_update = True

    # ===========================================================
    #      INTERPOLACIÓN SUAVE (coseno)
    # ===========================================================
    def interpolate_position(self, q_init, q_target):
        """
        Esta función suaviza el movimiento.
        No saltamos directo al ángulo final.
        """
        ratio = (1 - math.cos(math.pi * (self.t / self.T))) / 2 \
             if self.t < self.T else 1.0
        return q_init + (q_target - q_init) * ratio

    # ===========================================================
    #      LOOP PRINCIPAL DE CONTROL (ENVÍO DE COMANDOS)
    # ===========================================================
    def LowCmdWrite(self):
        if self.low_state is None:
            return

        # Liberador de seguridad (siempre así en SDK2)
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1

        # -------------------------------------------------------
        # CONTROLAMOS LOS BRAZOS
        # -------------------------------------------------------
        for joint in self.arm_joints:
            q_init = self.low_state.motor_state[joint].q
            q_target = self.target_pos.get(joint, q_init)
            pos = self.interpolate_position(q_init, q_target)

            cmd = self.low_cmd.motor_cmd[joint]
            cmd.q = pos
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = self.kp
            cmd.kd = self.kd

        # -------------------------------------------------------
        # RESTO DEL CUERPO → mantener estable
        # -------------------------------------------------------
        for joint in self.rest_joints:
            current_q = self.low_state.motor_state[joint].q
            cmd = self.low_cmd.motor_cmd[joint]
            cmd.q = current_q
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = 60.0
            cmd.kd = 1.5

        # # Enviar mensaje
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
        self.t += self.control_dt

    # ===========================================================
    #      FUNCIÓN CLAVE: move_to()
    # ===========================================================
    def move_to(self, updates: dict, duration=1.0):
        """
        move_to() hace tres cosas:
        1. Actualiza la postura objetivo.
        2. Define cuánto tiempo debe durar el movimiento.
        3. Espera hasta que la interpolación termine.

        No mueve nada aquí dentro.
        El movimiento real ocurre en LowCmdWrite().
        """
        self.target_pos.update(updates)
        self.T = duration
        self.t = 0.0
        while self.t < self.T:
            time.sleep(self.control_dt)

    # ===========================================================
    #      FUNCIÓN CLAVE: freeze_and_release()
    # ===========================================================
    def freeze_and_release(self):
    
        for joint in self.arm_joints:
            cmd = self.low_cmd.motor_cmd[joint]
            cmd.q = self.low_state.motor_state[joint].q
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = 0.0
            cmd.kd = 0.0

        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)


# ===========================================================
#                    MAIN PROGRAM
# ===========================================================
def main():

    # 1. Modo real vs simulado
    if len(sys.argv) > 1:
        interface = sys.argv[1]
        id = 0
        print("Modo robot real")
    else:
        interface = "lo"
        id = 1
        print("Modo robot simulado")

    input("\nAsegúrate de que no haya obstáculos.\nEnter para iniciar...")

    try:
        # 2. Crear ejecutor
        robot = G1TrajectoryExecutor(interface, id)
        robot.Init()
        robot.Start()

        # 3. Preguntar repeticiones
        n = int(input("¿Cuántas cajas quieres repetir? "))

        # 4. Repetición de la secuencia
        for i in range(n):
            print(f"\n======== CAJA {i+1} ========")

            print(">> Paso 1")
            robot.move_to(robot.pasos[0][1], duration=2.0)

            print(">> Paso 2")
            robot.move_to(robot.pasos[1][1], duration=2.0)

            print(">> Paso 3")
            robot.move_to(robot.pasos[2][1], duration=2.0)

            print(">> Paso 4")
            robot.move_to(robot.pasos[3][1], duration=2.0)

            print(">> Volviendo a postura neutra")
            robot.move_to(robot.pasos[0][1], duration=1.5)

        print("\n>> Secuencia completa.")
        robot.freeze_and_release()

    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario.")

    finally:
        try:
            robot.move_to(robot.pasos[0][1], duration=1.5)
            time.sleep(0.5)
            robot.freeze_and_release()
        except:
            pass


if __name__ == "__main__":
    main()