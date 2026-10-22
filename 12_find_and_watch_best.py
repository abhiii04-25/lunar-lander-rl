from pathlib import Path
import time
import statistics

import gymnasium as gym
from stable_baselines3 import PPO

CANDIDATES = [
    ("Final 300k model", Path("models/ppo_lunar_lander_300k.zip")),
    ("Best 300k checkpoint", Path("models/best_300k/best_model.zip")),
]

available = [(name, path) for name, path in CANDIDATES if path.exists()]

if not available:
    raise FileNotFoundError("No trained 300k model was found.")

env = gym.make("LunarLander-v3")
best_run = None

try:
    for name, path in available:
        model = PPO.load(str(path), device="cpu")
        rewards = []

        print(f"\nTesting {name}...")

        for seed in range(100, 120):
            observation, info = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0

            while True:
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            rewards.append(total_reward)

            if best_run is None or total_reward > best_run["reward"]:
                best_run = {
                    "name": name,
                    "path": path,
                    "seed": seed,
                    "reward": total_reward,
                    "steps": steps,
                }

        print(f"Mean reward: {statistics.mean(rewards):.2f}")
        print(f"Best reward: {max(rewards):.2f}")
        print(f"Episodes scoring 200+: {sum(r >= 200 for r in rewards)}/20")

finally:
    env.close()

print("\nBest attempt found:")
print("Model:", best_run["name"])
print("Seed:", best_run["seed"])
print("Reward:", round(best_run["reward"], 2))

if best_run["reward"] < 200:
    print("Note: No tested attempt reached the 200-point benchmark.")

# Replay the same starting seed in a visible window.
model = PPO.load(str(best_run["path"]), device="cpu")
render_env = gym.make("LunarLander-v3", render_mode="human")

try:
    observation, info = render_env.reset(seed=best_run["seed"])
    total_reward = 0.0
    steps = 0

    print("\nReplaying the selected attempt...")

    while True:
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = (
            render_env.step(action)
        )

        total_reward += reward
        steps += 1
        time.sleep(0.03)

        if terminated or truncated:
            break

    print("Replay reward:", round(total_reward, 2))
    print("Steps:", steps)
    print("Time limit reached:", truncated)

finally:
    render_env.close()
