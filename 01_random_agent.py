import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset(seed=42)

print("Observation shape:", observation.shape)
print("Initial observation:", np.round(observation, 3))
print("Action space:", env.action_space)

total_reward = 0.0

for step in range(1000):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step < 5:
        print(f"Step: {step + 1}, Action: {action}, Reward: {reward:.2f}")

    if terminated or truncated:
        print("Episode ended!")
        print("Steps:", step + 1)
        print("Total reward:", round(total_reward, 2))
        break

env.close()
