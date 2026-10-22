from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

models_dir = Path("models")
logs_dir = Path("logs")
models_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

original_model = models_dir / "ppo_lunar_lander_baseline.zip"

if not original_model.exists():
    raise FileNotFoundError(
        f"Could not find the original model: {original_model}"
    )

env = Monitor(gym.make("LunarLander-v3"))
eval_env = Monitor(gym.make("LunarLander-v3"))

callback = EvalCallback(
    eval_env,
    best_model_save_path=str(models_dir / "best_300k"),
    log_path=str(logs_dir / "evaluation_300k"),
    eval_freq=20_000,
    n_eval_episodes=5,
    deterministic=True,
)

model = PPO.load(
    str(original_model),
    env=env,
    device="cpu",
)

print("Resuming from", model.num_timesteps, "steps")

model.learn(
    total_timesteps=200_000,
    callback=callback,
    reset_num_timesteps=False,
)

model.save(str(models_dir / "ppo_lunar_lander_300k"))

env.close()
eval_env.close()

print("Continued training complete!")
print("Saved: models/ppo_lunar_lander_300k.zip")
