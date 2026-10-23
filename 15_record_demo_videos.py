"""Record the actual cinematic Pygame viewers for the ORBITAL website.

Run from the repository root after model checkpoints and visual assets exist:
    python 15_record_demo_videos.py

This launcher runs file 13 first (successful-landing cinematic) and file 14 second
(failure cinematic). Each viewer records its own rendered Pygame window directly
to docs/assets/*.mp4, including the HUD, artwork, effects, and terminal overlay.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main() -> None:
    viewers = [
        ("13_cinematic_lander.py", "docs/assets/success.mp4"),
        ("14_cinematic_failure.py", "docs/assets/failure.mp4"),
    ]

    for script, output in viewers:
        print(f"\n=== Recording output from {script} ===")
        print(f"Expected video: {output}")
        subprocess.run(
            [sys.executable, str(ROOT / script), "--record"],
            cwd=ROOT,
            check=True,
        )

    print("\nBoth cinematic replays recorded.")
    print("Next upload docs/assets/success.mp4 and docs/assets/failure.mp4 to GitHub.")


if __name__ == "__main__":
    main()
