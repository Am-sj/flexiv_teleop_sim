"""Two USB UArm leaders controlling a dual Flexiv simulation.

Run: python dual_teleop_sim.py --left-port /dev/ttyUSB0 --right-port /dev/ttyUSB1
"""

import argparse
from pathlib import Path
from queue import Empty, Queue
import re
import select
import sys
from threading import Event, Thread
import time

import gymnasium as gym
import mani_skill.envs  # Register scenes and robots.
import numpy as np
import serial
import sapien

from mani_skill.agents.robots.flexiv_rizon4.flexiv_rizon4 import FlexivRizon4


TRIGGER_PRESS_ANGLE = np.deg2rad(-40.)
TRIGGER_RELEASE_ANGLE = np.deg2rad(-30.)

SIDES = ("left", "right")
AGENT_IDS = {"left": "flexiv_rizon4-0", "right": "flexiv_rizon4-1"}
# List indices 0..6 correspond to URDF joint1..joint7. Adjust after hardware checks.
ARM_MAPPING = {
    "left": dict(
        servo_indices=[0, 1, 2, 3, 4, 5, 6],
        joint_signs=[1, -1, 1, 1, 1, 1, 1],
        joint_scales=[1.] * 7,
    ),
    "right": dict(
        servo_indices=[0, 1, 2, 3, 4, 5, 6],
        joint_signs=[1, -1, 1, 1, 1, 1, 1],
        joint_scales=[1.] * 7,
    ),
}


class ArmMapping:
    """Independent reference, joint limits, and trigger state for one follower."""

    def __init__(self, agent, mapping):
        arm = agent.controller.controllers["arm"]
        self.initial = arm.qpos[0].detach().cpu().numpy().copy()
        self.low = arm.single_action_space.low.copy()
        self.high = arm.single_action_space.high.copy()
        if self.initial.shape != (7,):
            raise ValueError("Each follower must have seven arm joints")
        self.indices = np.asarray(mapping["servo_indices"], dtype=int)
        self.signs = np.asarray(mapping["joint_signs"], dtype=np.float32)
        self.scales = np.asarray(mapping["joint_scales"], dtype=np.float32)
        if (sorted(self.indices.tolist()) != list(range(7))
                or self.signs.shape != (7,) or not np.isin(self.signs, [-1, 1]).all()
                or self.scales.shape != (7,) or not np.isfinite(self.scales).all()):
            raise ValueError("Invalid arm mapping: expected seven channels, signs and finite scales")
        self.closed = False
        self.pressed = False
        self.action = np.r_[self.initial, 1.].astype(np.float32)

    def convert(self, pose):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (8,) or not np.isfinite(pose).all():
            raise ValueError("Expected eight finite servo offsets in radians")
        offsets = pose[:7][self.indices] * self.signs * self.scales
        self.action[:7] = np.clip(self.initial + offsets, self.low, self.high)
        if not self.pressed and pose[7] <= TRIGGER_PRESS_ANGLE:
            self.closed = not self.closed
            self.pressed = True
        elif self.pressed and pose[7] >= TRIGGER_RELEASE_ANGLE:
            self.pressed = False
        self.action[7] = -1. if self.closed else 1.
        return self.action.copy()


class SingleArmWorldMapping(ArmMapping):
    """Replay the working upright single-arm TCP motion at a mounted arm's origin.

    The leader still uses the single-arm channel order/signs. FK produces its
    virtual single-arm TCP displacement in world axes; IK realizes that motion
    at the mounted follower while preserving its initial TCP orientation offset.
    """

    def __init__(self, agent, mapping):
        super().__init__(agent, mapping)
        self.label = f"{agent.uid}-{agent._agent_idx}"
        self.model = agent.robot.create_pinocchio_model()
        self.link_index = agent.tcp.index
        self.arm_indices = np.array([
            agent.robot.active_joints_map[name].active_index[0].item()
            for name in agent.arm_joint_names
        ], dtype=int)
        self.seed = agent.robot.qpos[0].detach().cpu().numpy().astype(np.float64).copy()
        self.mask = np.zeros(len(self.seed), dtype=np.int32)
        self.mask[self.arm_indices] = 1
        self.reference = self.seed.copy()
        self.reference[self.arm_indices] = FlexivRizon4.DEFAULT_ARM_QPOS
        self.reference_initial = self.fk(self.reference)
        self.root = agent.robot.pose.sp
        self.world_initial = self.root * self.fk(self.seed)
        self.last_target_world = self.world_initial
        self.last_ik_success = True
        self.last_warning_time = -float("inf")

    def fk(self, qpos):
        self.model.compute_forward_kinematics(qpos)
        return self.model.get_link_pose(self.link_index)

    def target_world(self, pose):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (8,) or not np.isfinite(pose).all():
            raise ValueError("Expected eight finite servo offsets in radians")
        reference = self.reference.copy()
        offsets = pose[:7][self.indices] * self.signs * self.scales
        reference[self.arm_indices] = np.clip(
            self.reference[self.arm_indices] + offsets, self.low, self.high)
        reference_pose = self.fk(reference)
        delta_rotation = (sapien.Pose(q=reference_pose.q)
                          * sapien.Pose(q=self.reference_initial.q).inv())
        orientation = delta_rotation * sapien.Pose(q=self.world_initial.q)
        return sapien.Pose(
            p=self.world_initial.p + reference_pose.p - self.reference_initial.p,
            q=orientation.q,
        )

    def convert(self, pose):
        # Keep the existing independent trigger state machine and validation.
        previous = self.action[:7].copy()
        super().convert(pose)
        self.last_target_world = self.target_world(pose)
        result, success, _ = self.model.compute_inverse_kinematics(
            self.link_index, self.root.inv() * self.last_target_world,
            initial_qpos=self.seed, active_qmask=self.mask,
            eps=1e-5, max_iterations=150, dt=0.5, damp=1e-5,
        )
        arm_result = result[self.arm_indices]
        self.last_ik_success = bool(
            success and np.isfinite(arm_result).all()
            and np.all(arm_result >= self.low - 1e-6)
            and np.all(arm_result <= self.high + 1e-6)
        )
        if self.last_ik_success:
            self.action[:7] = np.clip(arm_result, self.low, self.high)
            self.seed[self.arm_indices] = self.action[:7]
        else:
            # Never clip an invalid IK solution into an unrelated TCP movement.
            self.action[:7] = previous
            if time.monotonic() - self.last_warning_time >= 2.:
                print(f"[WARN] {self.label} target is unreachable or outside joint limits; "
                      "holding its last arm target. Move leader toward its reference pose.")
                self.last_warning_time = time.monotonic()
        return self.action.copy()


class SerialLeader:
    """Single-arm serial acquisition strategy, with independent state per side."""

    def __init__(self, side, port, rate, stop_event, errors, debug_angles=True):
        self.side = side
        self.SERIAL_PORT = port
        self.BAUDRATE = 115200
        self.num_servos = 8
        self.rate = rate
        self.stop_event = stop_event
        self.errors = errors
        self.debug_angles = debug_angles
        self.calibration_requested = Event()
        self.calibrated = Event()
        self.arm_pos_queue = Queue(maxsize=1)
        self.zero_angles = [0.0] * self.num_servos
        self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE,
                                 timeout=0.01, write_timeout=0.5)
        self.thread = Thread(target=self.angle_stream_loop, args=(self.default_sender,),
                             name=f"{side}-serial", daemon=True)
        try:
            self._init_servos()
        except Exception:
            self.ser.close()
            raise

    def _init_servos(self):
        self.send_command("#000PVER!")
        for i in range(self.num_servos):
            self.send_command("#000PCSK!")
            self.send_command(f"#{i:03d}PULK!")
        print(f"[INFO] {self.side}: Servos unlocked; waiting for manual calibration")

    def send_command(self, command):
        self.ser.write(command.encode("ascii"))
        time.sleep(0.008)
        return self.ser.read_all().decode("ascii", errors="ignore")

    def pwm_to_angle(self, response_str, servo_num, pwm_min=500, pwm_max=2500,
                     angle_range=270):
        match = re.search(rf"#{servo_num:03d}P(\d{{4}})", response_str)
        if match is None:
            return None
        return (int(match.group(1)) - pwm_min) / (pwm_max - pwm_min) * angle_range

    def read_angles(self):
        angles = []
        for i in range(self.num_servos):
            response = self.send_command(f"#{i:03d}PRAD!")
            angle = self.pwm_to_angle(response.strip(), i)
            if angle is None or not np.isfinite(angle):
                raise ValueError(f"{self.side} servo {i:03d}: invalid response {response!r}")
            angles.append(angle)
        return np.asarray(angles)

    def request_calibration(self):
        if self.calibrated.is_set():
            print(f"[INFO] {self.side}: Already calibrated; teleoperation active")
            return
        self.calibration_requested.set()

    def _calibrate_servos(self):
        # Commit only a complete reference; a failed channel leaves it unchanged.
        reference = self.read_angles()
        self.zero_angles = reference.tolist()
        self.calibrated.set()
        print(f"[INFO] {self.side}: Calibration complete; waiting for both leaders before following")

    def publish_arm_pos(self, arm_pos):
        while True:
            try:
                self.arm_pos_queue.get_nowait()
            except Empty:
                break
        self.arm_pos_queue.put_nowait(list(arm_pos))

    def get_latest_arm_pos(self, timeout=0.0):
        try:
            return self.arm_pos_queue.get(timeout=timeout)
        except Empty:
            return None

    def default_sender(self, arm_pos):
        if self.debug_angles:
            print(f"[{self.side}] Servo angles (degrees): {np.degrees(arm_pos)}")

    def angle_stream_loop(self, on_send):
        period = max(1.0 / self.rate, 1e-6)
        next_time = time.monotonic()
        try:
            while not self.stop_event.is_set():
                if not self.calibrated.is_set():
                    if not self.calibration_requested.wait(timeout=0.1):
                        continue
                    if self.stop_event.is_set():
                        break
                    self.calibration_requested.clear()
                    try:
                        self._calibrate_servos()
                    except ValueError as exc:
                        print(f"[ERROR] {exc}. Hold UArm still and press Enter to retry.")
                        continue
                    next_time = time.monotonic()
                arm_pos = np.radians(self.read_angles() - self.zero_angles)
                self.publish_arm_pos(arm_pos)
                on_send(list(arm_pos))
                next_time += period
                sleep_dt = next_time - time.monotonic()
                if sleep_dt > 0:
                    self.stop_event.wait(sleep_dt)
                else:
                    next_time = time.monotonic()
        except Exception as exc:
            self.errors.put((self.side, exc))
            self.stop_event.set()

    def close(self):
        if self.thread.ident is not None:
            self.thread.join(timeout=3.)
            if self.thread.is_alive():
                self.ser.cancel_read()
                self.ser.cancel_write()
                self.thread.join(timeout=1.)
        self.ser.close()


class DualServoTeleoperatorSim:
    """Two serial producers and one simulation consumer, using single-arm defaults.

    Rendering and terminal polling stay on the main thread. Both leaders must
    finish calibration; each follower retains its own reference and trigger.
    """

    def __init__(self, scene="FlexivDualArm-v1", left_port="/dev/ttyUSB0",
                 right_port="/dev/ttyUSB1", rate=50., headless=False,
                 mapping_mode="joint", debug_angles=True):
        if mapping_mode not in ("single-world", "joint"):
            raise ValueError("Unknown mapping mode")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("--rate must be finite and positive")
        if Path(left_port).resolve() == Path(right_port).resolve():
            raise ValueError("Left and right leaders require different serial ports")
        self.scene = scene
        self.robot_uids = ("flexiv_rizon4", "flexiv_rizon4")
        self.control_mode = "pd_joint_pos"
        self.rate = rate
        self.stop_event = Event()
        self.errors = Queue()
        self.leaders = {}
        self.env = None
        self.headless = headless
        try:
            self.env = gym.make(
                self.scene, robot_uids=self.robot_uids,
                obs_mode="state", control_mode=self.control_mode, sim_backend="cpu",
                render_backend="none" if headless else "sapien_cuda",
                render_mode=None if headless else "human",
                sim_config=dict(default_materials_config=dict(
                    static_friction=10., dynamic_friction=10., restitution=0.)),
                # Keep scene camera/shader defaults for dual-arm viewer picking.
            )
            self.env.reset(seed=0)
            print("Action space:", self.env.action_space)
            agents = getattr(self.env.unwrapped.agent, "agents_dict", {})
            if set(agents) != set(AGENT_IDS.values()):
                raise ValueError(f"Scene {scene!r} must support two Flexiv Rizon4 agents")
            mapper_type = SingleArmWorldMapping if mapping_mode == "single-world" else ArmMapping
            self.mappings = {
                side: mapper_type(agents[AGENT_IDS[side]], ARM_MAPPING[side]) for side in SIDES
            }
            print(f"[INFO] Arm mapping mode: {mapping_mode}")
            for side, port in zip(SIDES, (left_port, right_port)):
                self.leaders[side] = SerialLeader(
                    side, port, self.rate, self.stop_event, self.errors,
                    debug_angles=debug_angles,
                )
        except Exception:
            self.close()
            raise

    def request_calibration(self):
        if all(leader.calibrated.is_set() for leader in self.leaders.values()):
            print("[INFO] Already calibrated; teleoperation active")
            return
        for leader in self.leaders.values():
            if not leader.calibrated.is_set():
                leader.request_calibration()

    def convert_pose_to_action(self, poses):
        for side, pose in poses.items():
            self.mappings[side].convert(pose)
        return {AGENT_IDS[side]: self.mappings[side].action.copy() for side in SIDES}

    def teleop_sim_handler(self, action, dwell=0.01):
        if self.env is None or action is None:
            return
        self.env.step(action)
        if not self.headless:
            self.env.render()
        self.stop_event.wait(dwell)

    def step_if_ready(self, on_pose=None):
        # Match single-arm: no step before calibration or without a fresh sample.
        if not all(leader.calibrated.is_set() for leader in self.leaders.values()):
            return False
        poses = {}
        for side, leader in self.leaders.items():
            pose = leader.get_latest_arm_pos(timeout=0.0)
            if pose is not None:
                poses[side] = pose
        if not poses:
            return False
        action = self.convert_pose_to_action(poses)
        handler = self.teleop_sim_handler if on_pose is None else on_pose
        handler(action)
        return True

    def pose_consumer_loop(self, on_pose):
        period = max(1.0 / self.rate, 1e-6)
        next_time = time.monotonic()
        print("Action Space Shape:", {
            uid: space.shape[0] for uid, space in self.env.action_space.spaces.items()
        })
        while not self.stop_event.is_set():
            # Poll input here so serial failures can stop the simulation promptly.
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:
                    print("Terminal input closed; stopping without automatic calibration")
                    break
                command = line.strip().lower()
                if command == "q":
                    break
                if command == "":
                    self.request_calibration()
            if self.stop_event.is_set():
                break
            if not self.step_if_ready(on_pose) and not self.headless:
                self.env.render()
            next_time += period
            sleep_dt = next_time - time.monotonic()
            if sleep_dt > 0:
                self.stop_event.wait(sleep_dt)
            else:
                next_time = time.monotonic()

    def run(self):
        try:
            for side, leader in self.leaders.items():
                print(f"[{side}] Starting angle reading thread...")
                leader.thread.start()
            print("Starting simulation control loop...")
            print("Simulation ready. Position BOTH UArms and release BOTH gripper triggers.")
            print("Keep BOTH UArms still, then press Enter IN THIS TERMINAL to calibrate and start following.")
            print("Type q + Enter or press Ctrl+C to stop.")
            print("Left → flexiv_rizon4-0 (+Y); right → flexiv_rizon4-1 (-Y).")
            self.pose_consumer_loop(self.teleop_sim_handler)
        except KeyboardInterrupt:
            print("Received interrupt signal, preparing to stop...")
        finally:
            self.close()
            print("All threads stopped")
            print("Resource cleanup completed")
        if not self.errors.empty():
            side, exc = self.errors.get_nowait()
            raise RuntimeError(f"{side} serial acquisition stopped: {exc}") from exc

    def close(self):
        self.stop_event.set()
        for leader in self.leaders.values():
            leader.close()
        self.leaders.clear()
        if self.env is not None:
            self.env.close()
            self.env = None


def main():
    parser = argparse.ArgumentParser(
        description="Dual robot arm teleoperation simulation program",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene", "-s", default="FlexivDualArm-v1", help="Scene supporting two Rizon4 agents")
    parser.add_argument("--left-port", default="/dev/ttyUSB0", help="Left UArm serial port")
    parser.add_argument("--right-port", default="/dev/ttyUSB1", help="Right UArm serial port")
    parser.add_argument("--rate", type=float, default=50.,
                        help="Target acquisition and control loop rate (Hz); limited by serial and rendering latency")
    parser.add_argument("--mapping-mode", choices=["joint", "single-world"], default="joint",
                        help="joint: same joint-offset strategy as single-arm teleop; single-world: optional TCP retargeting")
    parser.add_argument("--debug-angles", action=argparse.BooleanOptionalAction, default=True,
                        help="Print each sample's calibrated servo offsets in degrees, like single-arm teleop")
    parser.add_argument("--headless", action="store_true", help="Run CPU physics without a window (still requires terminal and leaders)")
    args = parser.parse_args()
    print("=" * 60)
    print("Dual robot arm teleoperation simulation program")
    print("Robot: flexiv_rizon4 + flexiv_rizon4")
    print(f"Scene: {args.scene}")
    print(f"Rate: {args.rate} Hz")
    print(f"Serial port [left]: {args.left_port}")
    print(f"Serial port [right]: {args.right_port}")
    print("=" * 60)
    try:
        DualServoTeleoperatorSim(**vars(args)).run()
    except Exception as exc:
        import traceback
        print(f"Program runtime error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
