from pathlib import Path
import csv

import matplotlib.pyplot as plt

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

rows = [
    {
        "model": "Baseline 100k",
        "training_reward_design": "Original reward",
        "mean_original_reward": -17.55,
        "reward_std": 86.21,
        "episodes_scoring_200_plus": 0,
        "episodes_tested": 20,
        "average_episode_length": 896.2,
        "timeouts": 11,
    },
    {
        "model": "Continued 300k",
        "training_reward_design": "Original reward",
        "mean_original_reward": 104.87,
        "reward_std": 128.00,
        "episodes_scoring_200_plus": 8,
        "episodes_tested": 20,
        "average_episode_length": 430.9,
        "timeouts": 0,
    },
    {
        "model": "Time penalty 100k",
        "training_reward_design": "Original reward minus 0.01 per step",
        "mean_original_reward": -2.09,
        "reward_std": 44.79,
        "episodes_scoring_200_plus": 0,
        "episodes_tested": 20,
        "average_episode_length": 996.9,
        "timeouts": 19,
    },
    {
        "model": "Hover bonus 100k",
        "training_reward_design": "Bonus for slow, upright airborne states",
        "mean_original_reward": -145.56,
        "reward_std": 27.09,
        "episodes_scoring_200_plus": 0,
        "episodes_tested": 20,
        "average_episode_length": 972.0,
        "timeouts": 17,
    },
]

# Save the experiment data.
csv_path = output_dir / "experiment_results.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Plot every model using the original game reward.
labels = [row["model"] for row in rows]
scores = [row["mean_original_reward"] for row in rows]

plt.figure(figsize=(10, 5))
plt.bar(labels, scores)
plt.axhline(0, linewidth=0.8)
plt.ylabel("Mean reward (original environment)")
plt.title("Lunar Lander: Reward Design Comparison")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

chart_path = output_dir / "reward_comparison.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print("Results saved:")
print(csv_path.resolve())
print(chart_path.resolve())
