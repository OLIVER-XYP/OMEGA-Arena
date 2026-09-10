#!/usr/bin/env python3
"""Test halite_pyenv module."""

import sys
sys.path.insert(0, 'game_engine/build_pyenv')

import halite_pyenv

print('✓ halite_pyenv loaded successfully')

# Create environment
env = halite_pyenv.HaliteVecEnv('', 150)
print(f'✓ Created env with turn_limit={env.turn_limit}')

# Reset with 2 parallel games
env.reset([12345, 67890])
print(f'✓ Reset with 2 envs, size={env.size()}')

# Observe
obs = env.observe(0)
print(f'✓ Observed player 0, got {len(obs)} observations')
print(f'  Observation keys: {list(obs[0].keys())}')
print(f'  Turn: {obs[0]["turn"]}, Ships: {obs[0]["own_ship_count"]}')

# Test empty step (no commands for each env)
env.step([[], []], [[], []])  # 2 envs, each with empty command lists
print(f'✓ Stepped (empty commands)')

# Check if done
dones = env.dones()
print(f'✓ Dones: {dones}')

# Get scores
scores = env.scores()
print(f'✓ Scores: {scores}')

print('\n✅ halite_pyenv works perfectly!')
