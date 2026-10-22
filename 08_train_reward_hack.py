from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class HoverBonusWrapper(gym.Wrapper):
    """A deliberately flawed reward: reward slow, upright hovering."""

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # LunarLander observation:
        # 0=x, 1=y, 2=vx, 3=vy, 4=angle,
        # 5=angular velocity, 6=left leg, 7=right leg.
        vx = observation[2]
        vy = observation[3]
        angle = observation[4]
        angular_velocity = observation[5]
        left_leg = observation[6]
        right_leg = observation[7]

        airborne = left_leg < 0.5 and right_leg < 0.5

        upright_and_slow = (
            abs(vx) < 0.2
            and abs(vy) < 0.2
            and abs(angle) < 0.2
            and abs(angular_velocity) < 0.2
        )

        hover_bonus = (
            1.0 if airborne and upright_and_slow else 0.0
        )

        info = dict(info)
        info["hover_bonus"] = hover_bonus

        return (
            observation,
            float(reward) + hover_bonus,
            terminated,
            truncated,
            info,
        )


def make_env():
    env = gym.make("LunarLander-v3")
    env = HoverBonusWrapper(env)
    return Monitor(env)


env = make_env()
eval_env = make_env()

callback = EvalCallback(
    eval_env,
    best_model_save_path=str(MODELS_DIR / "reward_hack_best"),
    log_path=str(LOGS_DIR / "reward_hack_eval"),
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

print("Training with the deliberately flawed hover bonus...")

model.learn(
    total_timesteps=100_000,
    callback=callback,
)

model.save(str(MODELS_DIR / "ppo_lunar_lander_reward_hack_100k"))

env.close()
eval_env.close()

print("Reward-hacking experiment complete!")
print("Saved: models/ppo_lunar_lander_reward_hack_100k.zip")
