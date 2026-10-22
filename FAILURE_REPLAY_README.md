# Lunar Lander Failure Replay

This adds a cinematic failure replay for comparison with `13_cinematic_lander.py`.

## Install
1. Copy `14_cinematic_failure.py` into your existing `lunar-lander-rl` project folder, next to `13_cinematic_lander.py`.
2. Ensure your models folder contains `models/ppo_lunar_lander_baseline.zip` (the original 100k model).
3. Run from the project folder:

```powershell
& ".\.venv\Scripts\python.exe" .\14_cinematic_failure.py
```

The script evaluates the baseline model over seeds 100–119, chooses an actual unsuccessful attempt (preferring a crash, otherwise the worst failed/timeout attempt), then replays it in the same cinematic Pygame style with red danger effects and a `MISSION FAILED` or `CRASH DETECTED` overlay. It does not fake the outcome or retrain the model.

If no failed attempt is found in those seeds, change `SEEDS_TO_CHECK` near the top of the script to `range(120, 160)` and run it again.
