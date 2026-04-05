#!/usr/bin/env python3
"""Quick test script to verify the CoffeeShop environment pipeline."""

from envs.overcooked.wrapper import OvercookedSocialWrapper

try:
    from overcooked_ai_py.mdp.actions import Action
except ImportError:
    print("❌ Overcooked-AI not installed. Cannot run test.")
    exit(1)

# Instantiate the wrapper
env = OvercookedSocialWrapper()

# Test reset
obs, infos = env.reset()
assert isinstance(obs, dict), "Observations must be a dictionary"
assert "agent_0" in obs, "Observations must contain 'agent_0'"
print("✅ Reset successful: Observations are in dictionary format with agent keys")

# Create dummy actions
actions = {"agent_0": "interact", "agent_1": "interact"}

# Test step
next_obs, rewards, terminated, truncated, infos = env.step(actions)
assert isinstance(rewards, dict), "Rewards must be a dictionary"
print("✅ Step successful: Rewards are in dictionary format")

# Close the environment
env.close()

print("✅ All tests passed! Environment pipeline is working.")
