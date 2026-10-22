import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

model_path = Path("models/ppo_lunar_lander_baseline.zip")

if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found: {model_path.resolve()}"
    )

# Load the trained PPO policy.
model = PPO.load(str(model_path), device="cpu")
print("Trained PPO model loaded successfully!")

# Open the game window.
env = gym.make("LunarLander-v3", render_mode="human")

# Evaluate five attempts.
for episode in range(1, 6):
    observation, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        # Ask the trained policy to select an action.
        action, _ = model.predict(
            observation, deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        steps += 1

        # Slow the visual playback.
        time.sleep(0.03)

        if terminated or truncated:
            break

    print(
        f"Episode {episode} | "
        f"Reward: {total_reward:.2f} | "
        f"Steps: {steps} | "
        f"Terminated: {terminated} | "
        f"Truncated: {truncated}"
    )

env.close()
print("Evaluation finished!")
