from pathlib import Path
import statistics

import gymnasium as gym
from stable_baselines3 import PPO

model_path = Path("models/ppo_lunar_lander_reward_hack_100k.zip")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

# IMPORTANT: no HoverBonusWrapper here!
env = gym.make("LunarLander-v3")
model = PPO.load(str(model_path), device="cpu")

rewards = []
lengths = []
timeouts = 0

print("Testing the reward-hacking agent with ORIGINAL rewards...")

try:
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
                if truncated:
                    timeouts += 1
                break

        rewards.append(total_reward)
        lengths.append(steps)

        print(
            f"Episode {seed - 99:02d} | "
            f"Original reward: {total_reward:.2f} | "
            f"Steps: {steps} | "
            f"Timeout: {truncated}"
        )

finally:
    env.close()

print("\n--- FINAL RESULTS ---")
print(f"Mean original reward: {statistics.mean(rewards):.2f}")
print(f"Reward standard deviation: {statistics.pstdev(rewards):.2f}")
print(f"Episodes scoring 200+: {sum(r >= 200 for r in rewards)}/20")
print(f"Average episode length: {statistics.mean(lengths):.1f}")
print(f"Time-limit endings: {timeouts}/20")
