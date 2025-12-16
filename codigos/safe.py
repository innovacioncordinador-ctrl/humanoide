import time
import math
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
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


G1_NUM_MOTOR = 29


class G1BalanceSuave:
    
    def __init__(self, interface, id):
        ChannelFactoryInitialize(id, interface)
        
        self.control_dt = 0.002 
        self.kp = 600.0    
        self.kd = 4.0      
        
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.ready = False
        
        # ===== IMU =====
        self.imu_pitch = 0.0
        self.imu_roll = 0.0
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        
        # ===== IMU FILTRADO (suavizado) =====
        self.pitch_filtered = 0.0
        self.roll_filtered = 0.0
        self.filter_alpha = 0.05  # Filtro muy suave (0.05 = lento, 1.0 = sin filtro)
        
        # ===== BALANCE SUAVE =====
        self.Kp_pitch = 0.0    # Antes: 3.0 - Mucho más suave
        self.Kp_roll = 0.0     # Antes: 3.0
        self.Kd_pitch = 0.0    # Antes: 0.5
        self.Kd_roll = 0.0
        
        # ===== CORRECCIONES SUAVIZADAS =====
        self.current_corrections = {}  # Correcciones actuales
        self.smooth_factor = 0.02      # Qué tan rápido cambian (0.02 = muy suave)
        
       # ===== POSTURA PERFECTAMENTE CENTRADA =====
        self.base_pose = {
            # Pierna izquierda - CENTRADA
            G1JointIndex.LeftHipPitch: -0.35,     
            G1JointIndex.LeftHipRoll: 0.12,       
            G1JointIndex.LeftHipYaw: 0.0,
            G1JointIndex.LeftKnee: 0.5,           
            G1JointIndex.LeftAnklePitch: -0.25,   
            G1JointIndex.LeftAnkleRoll: -0.06,    
            
            # Pierna derecha - SIMÉTRICA
            G1JointIndex.RightHipPitch: -0.35,
            G1JointIndex.RightHipRoll: -0.12,
            G1JointIndex.RightHipYaw: 0.0,
            G1JointIndex.RightKnee: 0.5,
            G1JointIndex.RightAnklePitch: -0.25,
            G1JointIndex.RightAnkleRoll: 0.06,
            
            # Torso RECTO (sin inclinación)
            G1JointIndex.WaistYaw: 0.0,
            G1JointIndex.WaistRoll: 0.0,
            G1JointIndex.WaistPitch: 0.0,
            
            # Brazos simétricos y relajados
            G1JointIndex.LeftShoulderPitch: 0.3,
            G1JointIndex.LeftShoulderRoll: 0.3,
            G1JointIndex.LeftShoulderYaw: 0.0,
            G1JointIndex.LeftElbow: 0.5,
            G1JointIndex.LeftWristRoll: 0.0,
            G1JointIndex.LeftWristPitch: 0.0,
            G1JointIndex.LeftWristYaw: 0.0,
            
            G1JointIndex.RightShoulderPitch: 0.3,
            G1JointIndex.RightShoulderRoll: -0.3,
            G1JointIndex.RightShoulderYaw: 0.0,
            G1JointIndex.RightElbow: 0.5,
            G1JointIndex.RightWristRoll: 0.0,
            G1JointIndex.RightWristPitch: 0.0,
            G1JointIndex.RightWristYaw: 0.0,
        }
        
        # Inicializar correcciones en 0
        for joint in range(G1_NUM_MOTOR):
            self.current_corrections[joint] = 0.0
        
        self.print_counter = 0

    def Init(self):
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.thread = RecurrentThread(
            interval=self.control_dt,
            target=self.Loop,
            name="balance"
        )
        while not self.ready:
            time.sleep(0.01)
        print("[OK] Balance suave activo")
        self.thread.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        
        if hasattr(msg, 'imu_state'):
            imu = msg.imu_state
            if hasattr(imu, 'rpy') and len(imu.rpy) >= 2:
                self.imu_roll = imu.rpy[0]
                self.imu_pitch = imu.rpy[1]
            if hasattr(imu, 'gyroscope') and len(imu.gyroscope) >= 2:
                self.gyro_x = imu.gyroscope[0]
                self.gyro_y = imu.gyroscope[1]
        
        self.ready = True

    def compute_balance(self):
        """
        Calcula correcciones SUAVES para mantener equilibrio.
        """
        # ===== FILTRAR IMU (suavizar lecturas) =====
        self.pitch_filtered += self.filter_alpha * (self.imu_pitch - self.pitch_filtered)
        self.roll_filtered += self.filter_alpha * (self.imu_roll - self.roll_filtered)
        
        pitch = self.pitch_filtered
        roll = self.roll_filtered
        
        # ===== CALCULAR CORRECCIÓN DESEADA =====
        pitch_corr = self.Kp_pitch * pitch + self.Kd_pitch * self.gyro_y
        roll_corr = self.Kp_roll * roll + self.Kd_roll * self.gyro_x
        
        # ===== LIMITAR CORRECCIONES (muy pequeñas) =====
        pitch_corr = max(-0.3, min(0.3, pitch_corr))  # Antes: -0.8 a 0.8
        roll_corr = max(-0.2, min(0.2, roll_corr))    # Antes: -0.6 a 0.6
        
        # ===== CORRECCIONES OBJETIVO =====
        target_corrections = {
            # Tobillos - respuesta principal (reducida)
            G1JointIndex.LeftAnklePitch: -pitch_corr * 0.8,
            G1JointIndex.RightAnklePitch: -pitch_corr * 0.8,
            G1JointIndex.LeftAnkleRoll: -roll_corr * 0.6,
            G1JointIndex.RightAnkleRoll: -roll_corr * 0.6,
            
            # Caderas - respuesta secundaria (reducida)
            G1JointIndex.LeftHipPitch: pitch_corr * 0.4,
            G1JointIndex.RightHipPitch: pitch_corr * 0.4,
            G1JointIndex.LeftHipRoll: roll_corr * 0.3,
            G1JointIndex.RightHipRoll: roll_corr * 0.3,
            
            # Rodillas - mínimo
            G1JointIndex.LeftKnee: -pitch_corr * 0.1,
            G1JointIndex.RightKnee: -pitch_corr * 0.1,
            
            # Cintura - mínimo
            G1JointIndex.WaistPitch: -pitch_corr * 0.1,
            G1JointIndex.WaistRoll: -roll_corr * 0.2,
        }
        
        # ===== SUAVIZAR CORRECCIONES (transición lenta) =====
        for joint, target in target_corrections.items():
            current = self.current_corrections.get(joint, 0.0)
            # Interpolar suavemente hacia el objetivo
            self.current_corrections[joint] = current + self.smooth_factor * (target - current)

    def Loop(self):
        if self.low_state is None:
            return
        
        # Calcular balance
        self.compute_balance()
        
        # Enviar comandos
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = 0
        
        for joint in range(G1_NUM_MOTOR):
            # Posición = base + corrección suavizada
            q_target = self.base_pose.get(joint, 0.0) + self.current_corrections.get(joint, 0.0)
            
            cmd = self.low_cmd.motor_cmd[joint]
            cmd.mode = 0x01
            cmd.q = q_target
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = self.kp
            cmd.kd = self.kd
        
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
        
        # Mostrar estado
        self.print_counter += 1
        if self.print_counter >= 250:
            self.print_counter = 0
            self.print_status()

    def print_status(self):
        p = math.degrees(self.imu_pitch)
        r = math.degrees(self.imu_roll)
        pf = math.degrees(self.pitch_filtered)
        rf = math.degrees(self.roll_filtered)
        ok = abs(self.imu_pitch) < 0.1 and abs(self.imu_roll) < 0.1
        status = "✓ ESTABLE" if ok else "~ balance"
        print(f"\r[{status}] Real: P:{p:+5.1f}° R:{r:+5.1f}° | Filtrado: P:{pf:+5.1f}° R:{rf:+5.1f}°   ", end="", flush=True)

    def freeze(self):
        for joint in range(G1_NUM_MOTOR):
            cmd = self.low_cmd.motor_cmd[joint]
            cmd.mode = 0x00
            cmd.kp = 0.0
            cmd.kd = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)


def main():
    if len(sys.argv) > 1:
        interface = sys.argv[1]
        id = 0
        print("[MODO] Robot real")
    else:
        interface = "lo"
        id = 1
        print("[MODO] Simulación")
    
    robot = G1BalanceSuave(interface, id)
    robot.Init()
    
    print("\n" + "="*50)
    print("  G1 - EQUILIBRIO SUAVE")
    print("="*50)
    print("  Movimientos lentos y controlados")
    print("="*50 + "\n")
    
    robot.Start()
    
    print("[INFO] Ctrl+C para salir\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        robot.freeze()
        print("\n\n[OK] Robot liberado")


if __name__ == "__main__":
    main()