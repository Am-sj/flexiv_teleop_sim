"""View the CAD-mounted dual Rizon4 assembly, or run it without a display."""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import numpy as np


def hold_actions(env):
    """Return independent absolute arm targets with both Grav grippers open."""
    return {
        uid: np.r_[agent.controller.controllers["arm"].qpos[0].cpu().numpy(), 1.].astype(np.float32)
        for uid, agent in env.unwrapped.agent.agents_dict.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=None, help="Seconds to run (default: until interrupted)")
    parser.add_argument("--headless", action="store_true", help="Run CPU physics without opening a window")
    parser.add_argument("--output", type=Path, help="Save a rendered PNG; requires Vulkan even with --headless")
    args = parser.parse_args()
    if args.headless and args.duration is None:
        args.duration = 2.
    render = not args.headless or args.output is not None
    env = gym.make(
        "FlexivDualArm-v1", obs_mode="state", control_mode="pd_joint_pos",
        sim_backend="cpu", render_backend="sapien_cuda" if render else "none",
        render_mode=("rgb_array" if args.output is not None else None) if args.headless else "human",
    )
    try:
        env.reset(seed=0)
        actions = hold_actions(env)
        print("Action keys (left, right):", list(actions))
        print("Each action: 7 absolute arm angles in radians + Grav command (+1 open, -1 closed)")
        period = 1. / env.unwrapped.control_freq
        started = time.monotonic()
        while args.duration is None or time.monotonic() - started < args.duration:
            tick = time.monotonic()
            env.step(actions)
            if not args.headless:
                env.render()
            time.sleep(max(0., period - (time.monotonic() - tick)))
        if args.output is not None:
            from PIL import Image
            frame = env.unwrapped.render_rgb_array()[0].cpu().numpy()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(args.output)
            print("Saved", args.output)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
