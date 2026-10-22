from pathlib import Path
import statistics

import gymnasium as gym
from stable_baselines3 import PPO

MODELS = {
    "Baseline 100k": "models/ppo_lunar_lander_baseline.zip",
    "Continued 300k": "models/ppo_lunar_lander_300k.zip",
    "Time penalty 100k": "models/ppo_lunar_lander_time_penalty_100k.zip",
}

# Evaluate every model in the ORIGINAL environment.
env = gym.make("LunarLander-v3")
seeds = range(100, 120)

try:
    for name, filename in MODELS.items():
        path = Path(filename)

        if not path.exists():
            print(f"\nSkipping {name}: file not found: {path}")
            continue

        model = PPO.load(str(path), device="cpu")
        rewards = []
        lengths = []
        timeouts = 0

        print(f"\nTesting {name}...")

        for seed in seeds:
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
                    if truncated:
                        timeouts += 1
                    break

            rewards.append(total_reward)
            lengths.append(steps)

        print(f"Average reward: {statistics.mean(rewards):.2f}")
        print(
            "Reward standard deviation:",
            f"{statistics.pstdev(rewards):.2f}",
        )
        print(f"Episodes scoring 200+: {sum(r >= 200 for r in rewards)}/20")
        print(f"Average episode length: {statistics.mean(lengths):.1f}")
        print(f"Time-limit endings: {timeouts}/20")

finally:
    env.close()
