from pathlib import Path
import statistics

import gymnasium as gym
from stable_baselines3 import PPO

MODELS = {
    "Baseline (100k)": Path("models/ppo_lunar_lander_baseline.zip"),
    "Continued (300k)": Path("models/ppo_lunar_lander_300k.zip"),
}

for name, path in MODELS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path.resolve()}")

env = gym.make("LunarLander-v3")
results = {}

# Use identical initial seeds for both models.
seeds = range(100, 120)

for name, path in MODELS.items():
    model = PPO.load(str(path), device="cpu")
    rewards = []

    print(f"\nTesting {name}...")

    for seed in seeds:
        observation, info = env.reset(seed=seed)
        total_reward = 0.0

        while True:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    results[name] = rewards

    print(f"Average reward: {statistics.mean(rewards):.2f}")
    print(f"Reward standard deviation: {statistics.pstdev(rewards):.2f}")
    print(f"Episodes scoring 200 or more: {sum(r >= 200 for r in rewards)}/20")

env.close()

print("\n--- FINAL COMPARISON ---")
for name, rewards in results.items():
    print(
        f"{name}: mean={statistics.mean(rewards):.2f}, "
        f"successful episodes={sum(r >= 200 for r in rewards)}/20"
    )

difference = (
    statistics.mean(results["Continued (300k)"])
    - statistics.mean(results["Baseline (100k)"])
)

print(f"\nChange in average reward: {difference:+.2f}")
