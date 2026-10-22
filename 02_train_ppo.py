from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

env = Monitor(gym.make("LunarLander-v3"))
eval_env = Monitor(gym.make("LunarLander-v3"))

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(MODELS_DIR / "best"),
    log_path=str(LOGS_DIR / "evaluation"),
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42,
    device="cpu",
)

print("Starting PPO training...")

model.learn(
    total_timesteps=100_000,
    callback=eval_callback,
)

model.save(str(MODELS_DIR / "ppo_lunar_lander_baseline"))

env.close()
eval_env.close()

print("Training complete!")
print("Model saved in the models folder.")
