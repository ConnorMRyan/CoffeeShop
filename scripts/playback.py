"""
scripts/playback.py — CoffeeShop Visualization Utility (Headless Pygame Version)
==================================================================================
Renders a saved checkpoint to a GIF by manually scraping Pygame surfaces.

Bug fixes over previous version
---------------------------------
1.  SDL_VIDEODRIVER=dummy set after pygame.init() — had no effect
    The env var must be exported BEFORE pygame.init() to prevent pygame
    from attempting to open a real display.  In headless / WSL environments
    the old ordering caused an immediate crash on init.

2.  horizon=4000 is 10× the trained episode length
    Agents are trained with horizon=400.  Running for 4000 steps produces
    4000 - 400 = 3600 frames of policy behaviour outside the training
    distribution.  Default changed to 400 to match training.

3.  Final state never rendered
    states.append() was called before env.step(), so the state produced
    by the last action was never captured.  The append now happens after
    the step so every resulting state is included.

4.  Episode termination used all() instead of any()
    all(terminated.values()) is accidentally correct for Overcooked (all
    agents terminate together) but semantically wrong and would silently
    continue past episode end in any env where agents can terminate
    independently.  Changed to any() to match the rest of the codebase.

5.  Redundant pygame.display.set_mode(1,1)
    StateVisualizer creates its own surface and does not draw to the
    display.  The set_mode call was unnecessary and could interfere with
    the visualizer's surface allocation on some pygame versions.  Removed.

6.  Hardcoded checkpoint path and output filename
    Added argparse so the script is usable without editing source.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

# FIX (bug 1): set dummy video driver BEFORE any pygame import or init call.
# This must happen at module level so it takes effect even if pygame is
# imported transitively by a dependency.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import torch
import numpy as np
import imageio.v2 as imageio
import pygame
from torch.distributions import Categorical

from envs.overcooked.wrapper import OvercookedSocialWrapper
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from agents.ppo import PPOAgent
from core_marl.mediator import CoffeeShopMediator
from utils.torch_utils import select_action


# ---------------------------------------------------------------------------
# Main Playback Logic
# ---------------------------------------------------------------------------

def record_peak_performance(
        checkpoint_path: str,
        output_path:     str  = "playback.gif",
        layout_name:     str  = "cramped_room",
        horizon:         int  = 400,          # FIX (bug 2): match training horizon
        fps:             int  = 10,
        device:          str  = "cpu",
        deterministic:   bool = True,
        loop:            int  = 0,            # 0 = infinite loop in GIF
) -> None:

    # ── 1. Load Checkpoint (early, so meta can configure the env) ─────────
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Read architecture / env metadata written by train.py (post-meta checkpoints).
    # Fall back gracefully for older checkpoints that lack the key.
    meta = checkpoint.get("meta", {})
    ckpt_env     = meta.get("env",       "overcooked")
    ckpt_layout  = meta.get("layout",    layout_name)
    ckpt_encoder = meta.get("encoder",   "mlp")
    ckpt_imgshp  = meta.get("img_shape", None)
    ckpt_hidden  = meta.get("hidden",    128)

    if meta:
        print(f"  meta → env={ckpt_env}, layout={ckpt_layout}, "
              f"encoder={ckpt_encoder}, hidden={ckpt_hidden}")

    # Use the layout from meta if not overridden by the caller.
    effective_layout = ckpt_layout if layout_name == "cramped_room" and ckpt_layout else layout_name

    # ── 2. Environment & Mediator ─────────────────────────────────────────
    env = OvercookedSocialWrapper(layout_name=effective_layout, horizon=horizon)
    mediator = CoffeeShopMediator(global_obs_dim=env.global_obs_dim, device=device)

    # ── 3. Agents (configured from checkpoint meta) ───────────────────────
    # Agent ids use local names here; the PPOAgent mediator machinery is
    # never invoked during playback so global naming is not required.
    agent0 = PPOAgent(
        agent_id       = "agent_0",
        obs_dim        = env.obs_dim,
        action_dim     = env.action_dim,
        global_obs_dim = env.global_obs_dim,
        mediator       = mediator,
        hidden         = ckpt_hidden,
        encoder        = ckpt_encoder,
        img_shape      = ckpt_imgshp,
        device         = device,
    )
    agent1 = PPOAgent(
        agent_id       = "agent_1",
        obs_dim        = env.obs_dim,
        action_dim     = env.action_dim,
        global_obs_dim = env.global_obs_dim,
        mediator       = mediator,
        hidden         = ckpt_hidden,
        encoder        = ckpt_encoder,
        img_shape      = ckpt_imgshp,
        device         = device,
    )

    # ── 4. Load Weights ───────────────────────────────────────────────────
    # Support both current format ('env0_agent_0') and legacy ('agent_0')
    a0_key = "env0_agent_0" if "env0_agent_0" in checkpoint else "agent_0"
    a1_key = "env0_agent_1" if "env0_agent_1" in checkpoint else "agent_1"

    agent0.ac.load_state_dict(checkpoint[a0_key])
    agent1.ac.load_state_dict(checkpoint[a1_key])
    agent0.ac.eval()
    agent1.ac.eval()

    # ── 5. Simulation ─────────────────────────────────────────────────────
    obs, _            = env.reset()
    states            = []
    total_team_reward = 0.0

    print(f"Simulating up to {horizon} steps in layout '{layout_name}' ...")

    for t in range(horizon):
        a0 = select_action(agent0.ac, obs["agent_0"], deterministic)
        a1 = select_action(agent1.ac, obs["agent_1"], deterministic)

        obs, rewards, terminated, truncated, infos = env.step(
            {"agent_0": a0, "agent_1": a1}
        )

        # FIX (bug 3): capture the state AFTER the step so every resulting
        # state is rendered, including the frame produced by the last action.
        states.append(env._env.state.deepcopy())

        total_team_reward += infos.get("team_reward", 0.0)

        # FIX (bug 4): any() is the correct semantic for "episode over"
        if any(terminated.values()) or any(truncated.values()):
            break

    print(f"Simulation complete — {len(states)} frames, "
          f"team reward: {total_team_reward:.3f}")

    # ── 6. Render frames ──────────────────────────────────────────────────
    # FIX (bug 1): pygame.init() is called AFTER the SDL env vars are set
    # (they were set at module import time above).
    pygame.init()
    # FIX (bug 5): do NOT call pygame.display.set_mode(); StateVisualizer
    # manages its own surface and does not draw to the display.

    visualizer = StateVisualizer()
    frames: list[np.ndarray] = []

    print(f"Rendering {len(states)} frames ...")
    for state in states:
        surface   = visualizer.render_state(state, grid=env._mdp.terrain_mtx)
        img_array = pygame.surfarray.array3d(surface)
        # pygame returns (W, H, 3); imageio expects (H, W, 3)
        frames.append(img_array.transpose(1, 0, 2))

    pygame.quit()

    # ── 7. Save GIF ───────────────────────────────────────────────────────
    print(f"Writing GIF to {output_path} at {fps} FPS ...")
    imageio.mimsave(output_path, frames, fps=fps, loop=loop)

    print(f"Saved → {os.path.abspath(output_path)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoffeeShop checkpoint playback → GIF")
    p.add_argument("checkpoint",          help="Path to .pt checkpoint file")
    p.add_argument("--output",   "-o",    default="playback.gif",
                   help="Output GIF path (default: playback.gif)")
    p.add_argument("--layout",            default="cramped_room",
                   help="Overcooked layout name")
    p.add_argument("--horizon",  type=int, default=400,
                   help="Max steps to simulate (default: 400, matching training)")
    p.add_argument("--fps",      type=int, default=10)
    p.add_argument("--device",            default="cpu")
    p.add_argument("--stochastic",        action="store_true",
                   help="Sample actions instead of greedy argmax")
    p.add_argument("--no-loop",           action="store_true",
                   help="GIF plays once instead of looping")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_peak_performance(
        checkpoint_path = args.checkpoint,
        output_path     = args.output,
        layout_name     = args.layout,
        horizon         = args.horizon,
        fps             = args.fps,
        device          = args.device,
        deterministic   = not args.stochastic,
        loop            = 1 if args.no_loop else 0,
    )