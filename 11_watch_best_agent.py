import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = Path("models/ppo_lunar_lander_300k.zip")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

# Load the strongest model from our comparison.
model = PPO.load(str(MODEL_PATH), device="cpu")
print("Loaded the 300k PPO model!")

# Open the visible Lunar Lander game.
env = gym.make("LunarLander-v3", render_mode="human")

try:
    for episode in range(1, 6):
        observation, info = env.reset()
        total_reward = 0.0
        steps = 0

        print(f"\nStarting landing attempt {episode}...")

        while True:
            # Use the policy learned during training.
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            steps += 1

            # Slow the display so we can watch the rocket.
            time.sleep(0.03)

            if terminated or truncated:
                break

        if terminated:
            outcome = "Episode ended naturally (inspect the landing or crash)"
        else:
            outcome = "Time limit reached"

        print(f"Episode: {episode}")
        print(f"Reward: {total_reward:.2f}")
        print(f"Steps: {steps}")
        print(f"Outcome: {outcome}")

        # Brief pause between attempts.
        time.sleep(1)

finally:
    env.close()

print("\nAll five attempts finished!")
