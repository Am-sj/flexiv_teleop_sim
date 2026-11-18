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


SIDES = ("left", "right")
AGENT_IDS = {"left": "flexiv_rizon4-0", "right": "flexiv_rizon4-1"}
# Radians, joint1..joint7; defaults are the collision-free FlexivDualArm-v1 mount poses.
INIT_ARM_QPOS = {
    "left": [1.57, -0.6, 0.0, 1.8, 0.0, 0.8, 0.0],
    "right": [1.57, -0.6, 0.0, 1.8, 0.0, 0.8, 0.0],
}
# INIT_ARM_QPOS = {
#     "left": [-0.323465, 1.963841, 0.553907, 2.263673, 2.451158, 0.345103, 0.0],
#     "right": [0.180512, 1.993543, -0.712481, 2.561961, -2.593623, 0.773324, -1.348068],
# }
TRIGGER_PRESS_ANGLE = np.deg2rad(-40.)
TRIGGER_RELEASE_ANGLE = np.deg2rad(-30.)
PWM_MIN = 500
PWM_MAX = 2500
ANGLE_RANGE_DEG = 270
BAUDRATE = 115200
NUM_SERVOS = 8
# Indexed by follower joint1..joint7. Adjust each side after hardware checks.
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


def _arm_joint_indices(agent):
    return np.array([
        agent.robot.active_joints_map[name].active_index[0].item()
        for name in agent.arm_joint_names
    ], dtype=int)


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
                or self.signs.shape != (7,)
                or not np.isin(self.signs, [-1, 1]).all()
                or self.scales.shape != (7,)
                or not np.isfinite(self.scales).all()):
            raise ValueError(
                "Invalid arm mapping: expected seven channels, signs and finite scales"
            )
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
        self.arm_indices = _arm_joint_indices(agent)
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
    """Read one UArm leader and publish calibrated angle samples.

    The simulation thread owns calibration requests; this leader's worker
    thread owns serial I/O and publishes the latest sample through a queue.
    """

    def __init__(self, side, port, rate, stop_event, errors, debug_angles=False):
        self.side = side
        self.rate = rate
        self.stop_event = stop_event
        self.errors = errors
        self.debug_angles = debug_angles
        self.last_debug_time = -float("inf")
        self.calibration_requested = Event()
        self.calibrated = Event()
        self.samples = Queue(maxsize=1)
        self.zero_angles = [0.] * NUM_SERVOS
        self.ser = serial.Serial(port, BAUDRATE, timeout=0.01, write_timeout=0.5)
        self.thread = Thread(target=self._loop, name=f"{side}-serial", daemon=True)
        try:
            self._init_servos()
        except Exception:
            self.ser.close()
            raise

    def _init_servos(self):
        self.send_command("#000PVER!")
        for i in range(NUM_SERVOS):
            self.send_command("#000PCSK!")
            self.send_command(f"#{i:03d}PULK!")

    def send_command(self, command):
        self.ser.write(command.encode("ascii"))
        time.sleep(0.008)
        return self.ser.read_all().decode("ascii", errors="ignore")

    def read_angles(self):
        angles = []
        for i in range(NUM_SERVOS):
            response = self.send_command(f"#{i:03d}PRAD!")
            angles.append(self.pwm_to_angle(response, i))
        return np.asarray(angles)

    def pwm_to_angle(self, response, index):
        match = re.search(rf"#{index:03d}P(\d{{4}})", response)
        if match is None:
            raise ValueError(
                f"{self.side} servo {index:03d}: invalid response {response!r}"
            )
        pwm = int(match.group(1))
        return (pwm - PWM_MIN) / (PWM_MAX - PWM_MIN) * ANGLE_RANGE_DEG

    def request_calibration(self):
        if not self.calibrated.is_set():
            self.calibration_requested.set()

    def publish_arm_pos(self, arm_pos):
        self.latest()
        self.samples.put_nowait(np.asarray(arm_pos))

    def latest(self):
        try:
            return self.samples.get_nowait()
        except Empty:
            return None

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                tick = time.monotonic()
                if not self.calibrated.is_set():
                    if not self.calibration_requested.wait(0.1):
                        continue
                    if self.stop_event.is_set():
                        break
                    self.calibration_requested.clear()
                    try:
                        reference = self.read_angles()
                    except ValueError as exc:
                        print(f"[ERROR] {exc}; hold both leaders still and press Enter to retry.")
                        continue
                    self.zero_angles = reference
                    self.calibrated.set()
                    print(f"[INFO] {self.side}: all eight channels calibrated")
                else:
                    offsets = np.radians(self.read_angles() - self.zero_angles)
                    self.publish_arm_pos(offsets)
                    if self.debug_angles and time.monotonic() - self.last_debug_time >= 0.5:
                        values = " ".join(
                            f"{i:03d}={value:+.1f}"
                            for i, value in enumerate(np.degrees(offsets))
                        )
                        print(f"[ANGLES] {self.side} offsets (deg): {values}")
                        self.last_debug_time = time.monotonic()
                self.stop_event.wait(max(0., 1. / self.rate - (time.monotonic() - tick)))
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
    """Run two independent UArm leaders against a dual Flexiv simulation.

    The main thread owns environment stepping and calibration input. Each leader
    owns its serial connection in a dedicated worker thread.
    """

    def __init__(self, scene="FlexivDualArm-v1", left_port="/dev/ttyUSB0",
                 right_port="/dev/ttyUSB1", rate=50., headless=False,
                 mapping_mode="joint", debug_angles=False):
        if mapping_mode not in ("single-world", "joint"):
            raise ValueError("Unknown mapping mode")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("--rate must be finite and positive")
        if Path(left_port).resolve() == Path(right_port).resolve():
            raise ValueError("Left and right leaders require different serial ports")
        self.stop_event = Event()
        self.errors = Queue()
        self.leaders = {}
        self.env = None
        self.headless = headless
        try:
            self.env = gym.make(
                scene, robot_uids=("flexiv_rizon4", "flexiv_rizon4"),
                obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
                render_backend="none" if headless else "sapien_cuda",
                render_mode=None if headless else "human",
                sim_config=dict(default_materials_config=dict(
                    static_friction=10., dynamic_friction=10., restitution=0.)),
            )
            self.env.reset(seed=0)
            agents = getattr(self.env.unwrapped.agent, "agents_dict", {})
            if set(agents) != set(AGENT_IDS.values()):
                raise ValueError(f"Scene {scene!r} must support two Flexiv Rizon4 agents")
            self._set_custom_init_pose(agents)
            mapper_type = SingleArmWorldMapping if mapping_mode == "single-world" else ArmMapping
            self.mappings = {
                side: mapper_type(agents[AGENT_IDS[side]], ARM_MAPPING[side]) for side in SIDES
            }
            print(f"[INFO] Arm mapping mode: {mapping_mode}")
            for side, port in zip(SIDES, (left_port, right_port)):
                self.leaders[side] = SerialLeader(
                    side, port, rate, self.stop_event, self.errors,
                    debug_angles=debug_angles,
                )
        except Exception:
            self.close()
            raise

    def _set_custom_init_pose(self, agents):
        """Set each arm's configured pose while preserving its gripper joints."""
        for side in SIDES:
            agent = agents[AGENT_IDS[side]]
            arm_controller = agent.controller.controllers["arm"]
            init_qpos = np.asarray(INIT_ARM_QPOS[side], dtype=np.float64)
            low = arm_controller.single_action_space.low
            high = arm_controller.single_action_space.high
            if (init_qpos.shape != (7,) or not np.isfinite(init_qpos).all()
                    or np.any(init_qpos < low) or np.any(init_qpos > high)):
                raise ValueError(
                    f"INIT_ARM_QPOS[{side!r}] must contain seven finite values "
                    "within the arm controller joint limits"
                )
            qpos = agent.robot.get_qpos()[0].detach().cpu().numpy().copy()
            qpos[_arm_joint_indices(agent)] = init_qpos.astype(qpos.dtype)
            agent.reset(qpos)
            # agent.reset leaves the controller's cached start/target qpos at the scene pose.
            agent.controller.reset()

    def request_calibration(self):
        if all(leader.calibrated.is_set() for leader in self.leaders.values()):
            print("[INFO] Already calibrated; restart to choose new references")
            return
        for leader in self.leaders.values():
            leader.request_calibration()

    def step_if_ready(self):
        # Both full references are required before advancing physics.
        if not all(leader.calibrated.is_set() for leader in self.leaders.values()):
            return False
        fresh_sample = False
        for side, leader in self.leaders.items():
            pose = leader.latest()
            if pose is not None:
                self.mappings[side].convert(pose)
                fresh_sample = True
        if not fresh_sample:
            return False
        self.env.step({AGENT_IDS[side]: self.mappings[side].action.copy() for side in SIDES})
        return True

    def run(self):
        try:
            for leader in self.leaders.values():
                leader.thread.start()
            print("Position BOTH leaders, release BOTH triggers, and keep both still.")
            print("Press Enter IN THIS TERMINAL to calibrate. Type q + Enter or Ctrl+C to stop.")
            print("Left → flexiv_rizon4-0 (+Y); right → flexiv_rizon4-1 (-Y).")
            period = 1. / self.env.unwrapped.control_freq
            while not self.stop_event.is_set():
                tick = time.monotonic()
                # Poll input on the simulation thread so worker errors can stop it.
                if select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if not line or line.strip().lower() == "q":
                        break
                    if not line.strip():
                        self.request_calibration()
                if self.stop_event.is_set():
                    break
                self.step_if_ready()
                if not self.headless:
                    self.env.render()
                self.stop_event.wait(max(0., period - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
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
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene", "-s", default="FlexivDualArm-v1",
        help="Scene supporting two Rizon4 agents",
    )
    parser.add_argument(
        "--left-port", default="/dev/ttyUSB0",
        help="Serial port for the left UArm leader",
    )
    parser.add_argument(
        "--right-port", default="/dev/ttyUSB1",
        help="Serial port for the right UArm leader",
    )
    parser.add_argument(
        "--rate", type=float, default=50.,
        help="Target serial scan rate per leader; limited by serial latency",
    )
    parser.add_argument(
        "--mapping-mode", choices=["joint", "single-world"], default="joint",
        help="joint: same joint-offset strategy as single-arm teleop; "
             "single-world: optional TCP retargeting",
    )
    parser.add_argument(
        "--debug-angles", action="store_true",
        help="Print calibrated servo offsets in degrees per side, twice per second",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run CPU physics without a window (still requires terminal and leaders)",
    )
    args = parser.parse_args()
    try:
        DualServoTeleoperatorSim(**vars(args)).run()
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
