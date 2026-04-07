#!/usr/bin/env python3
"""Unified test script to verify all CoffeeShop environment pipelines."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import traceback

def test_environment(env_name: str, module_path: str, class_name: str, kwargs: dict, get_dummy_actions_fn):
    print(f"\n{'-'*40}")
    print(f"🧪 Testing Environment: {env_name.upper()}")
    print(f"{'-'*40}")

    # 1. Try to import the wrapper and dependencies
    try:
        module = importlib.import_module(module_path)
        WrapperClass = getattr(module, class_name)
    except ImportError as e:
        print(f"⚠️  Skipped: Wrapper or library not installed. ({e})")
        return
    except AttributeError:
        print(f"⚠️  Skipped: Class '{class_name}' not found in {module_path}.")
        return

    # 2. Run the pipeline tests
    try:
        print("Initializing...")
        try:
            env = WrapperClass(**kwargs)
        except ImportError as e:
            print(f"⚠️  Skipped: Required library not installed. ({e})")
            return

        print(f"Agent IDs detected: {env.agent_ids}")

        # Test Reset
        obs, infos = env.reset()
        assert isinstance(obs, dict), f"Reset Failed: obs must be a dict, got {type(obs)}"
        assert isinstance(infos, dict), f"Reset Failed: infos must be a dict, got {type(infos)}"

        # Verify all agents are in the observation dict
        for aid in env.agent_ids:
            assert aid in obs, f"Reset Failed: '{aid}' missing from obs dict"
        print("✅ Reset passed: Returned valid {agent_id: value} dictionaries.")

        # Get actions and Test Step
        dummy_actions = get_dummy_actions_fn(env.agent_ids)
        next_obs, rewards, terminated, truncated, step_infos = env.step(dummy_actions)

        # Verify return structures
        assert isinstance(rewards, dict), "Step Failed: rewards must be a dict"
        assert isinstance(terminated, dict), "Step Failed: terminated must be a dict"
        print(f"✅ Step passed: Returned valid dictionaries. Sample rewards: {rewards}")

    except Exception as e:
        print(f"❌ Test Failed for {env_name}!")
        traceback.print_exc()

# --- Dummy Action Generators ---
# Each environment has different action spaces, so we define how to generate a dummy action for them.

def overcooked_actions(agent_ids):
    return {aid: 0 for aid in agent_ids}

def discrete_actions(agent_ids):
    # Standard fallback for Crafter/NLE (just passing 0 or 'no-op')
    return {aid: 0 for aid in agent_ids}

# --- Test Registry ---
if __name__ == "__main__":
    print("Starting Multi-Environment Pipeline Verification...")

    ENV_REGISTRY = [
        {
            "env_name": "Overcooked",
            "module_path": "envs.overcooked.wrapper",
            "class_name": "OvercookedSocialWrapper",
            "kwargs": {"layout_name": "cramped_room"},
            "get_dummy_actions_fn": overcooked_actions
        },
        {
            "env_name": "Crafter",
            "module_path": "envs.crafter.wrapper",
            "class_name": "CrafterSocialWrapper",
            "kwargs": {}, # Future kwargs
            "get_dummy_actions_fn": discrete_actions
        },
        {
            "env_name": "NetHack (NLE)",
            "module_path": "envs.nethack.wrapper",
            "class_name": "NLESocialWrapper",
            "kwargs": {}, # Future kwargs
            "get_dummy_actions_fn": discrete_actions
        },
        {
            "env_name": "AIsaac",
            "module_path": "envs.aisaac.wrapper",
            "class_name": "AIsaacWrapper",
            "kwargs": {}, # Future kwargs
            "get_dummy_actions_fn": discrete_actions
        }
    ]

    for test_config in ENV_REGISTRY:
        test_environment(**test_config)

    print(f"\n{'-'*40}")
    print("🏁 Pipeline verification complete.")