"""Record representative best/worst LunarLander episodes for the ORBITAL website.

Run from the repository root with Python 3.11 after training both PPO models:
    python 15_record_demo_videos.py

The script evaluates seeds 100–119, then renders only the chosen episode to MP4.
This keeps memory use much lower than saving frames from every evaluation episode.
"""
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "docs" / "assets"
SEEDS = range(100, 120)

MODELS = [
    {
        "path": ROOT / "models" / "ppo_lunar_lander_300k.zip",
        "output": OUTPUT_DIR / "success.mp4",
        "choose": max,
        "label": "best-reward trained-agent run",
    },
    {
        "path": ROOT / "models" / "ppo_lunar_lander_baseline.zip",
        "output": OUTPUT_DIR / "failure.mp4",
        "choose": min,
        "label": "lowest-reward baseline run",
    },
]


def evaluate_seed(model: PPO, seed: int) -> float:
    """Evaluate one seed without rendering or storing video frames."""
    env = gym.make("LunarLander-v3")
    try:
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += float(reward)

        return total_reward
    finally:
        env.close()


def render_episode(model: PPO, seed: int, output_path: Path) -> None:
    """Render one episode directly to MP4 rather than keeping frames in RAM."""
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False

        with imageio.get_writer(
            output_path,
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as writer:
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
                frame = env.render()
                if frame is not None:
                    writer.append_data(frame)
    finally:
        env.close()


def record(config: dict) -> None:
    model_path = config["path"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train the relevant model first, or put its .zip checkpoint "
            "in the models/ folder."
        )

    print(f"Loading {model_path.relative_to(ROOT)} ...")
    model = PPO.load(str(model_path))

    scores = []
    for seed in SEEDS:
        reward = evaluate_seed(model, seed)
        scores.append((reward, seed))
        print(f"  seed {seed}: reward {reward:.1f}")

    choose = config["choose"]
    reward, seed = choose(scores, key=lambda item: item[0])
    output_path = config["output"]

    print(f"\nRecording {config['label']}: seed {seed}, reward {reward:.1f}")
    render_episode(model, seed, output_path)
    print(f"Saved: {output_path.relative_to(ROOT)}")


def main() -> None:
    for config in MODELS:
        record(config)
        print("-" * 60)

    print("\nDone! Commit docs/assets/success.mp4 and docs/assets/failure.mp4")
    print("to GitHub so the website can display the recordings.")


if __name__ == "__main__":
    main()
