from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class TimePenaltyWrapper(gym.Wrapper):
    """Subtract a small amount from the reward at every step."""

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        reward = float(reward) - 0.01

        return observation, reward, terminated, truncated, info


def make_env():
    env = gym.make("LunarLander-v3")
    env = TimePenaltyWrapper(env)
    return Monitor(env)


env = make_env()
eval_env = make_env()

callback = EvalCallback(
    eval_env,
    best_model_save_path=str(MODELS_DIR / "time_penalty_best"),
    log_path=str(LOGS_DIR / "time_penalty_eval"),
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

# Train a fresh model so we can compare equal training budgets.
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42,
    device="cpu",
)

print("Training PPO with a small time penalty...")

model.learn(
    total_timesteps=100_000,
    callback=callback,
)

model.save(
    str(MODELS_DIR / "ppo_lunar_lander_time_penalty_100k")
)

env.close()
eval_env.close()

print("Training complete!")
print("Saved: models/ppo_lunar_lander_time_penalty_100k.zip")
