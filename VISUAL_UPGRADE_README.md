# Cinematic Lunar Lander Viewer

This add-on visualizes your existing PPO Lunar Lander agent in a cinematic sunset landing scene. It does **not** alter the trained model or the environment's physics. On a successful Gymnasium touchdown, it launches animated confetti and displays a touchdown banner.

## Install

From the root of your existing `lunar-lander-rl` project, activate the virtual environment and install the extra display dependency:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install pygame
```

Extract the ZIP contents directly into:

`C:\Users\Abhishek\lunar-lander-rl`

Keep the folder structure:

```text
lunar-lander-rl/
├── 13_cinematic_lander.py
├── assets/
│   ├── landing_base.png
│   └── rocket_lander.png
└── models/
    └── ppo_lunar_lander_300k.zip
```

The script searches candidate models across seeds 100–119, prefers a confirmed safe landing, then replays that attempt in the new viewer. A successful landing triggers a 5.5-second confetti celebration. Press SPACE to replay the selected seed or ESC to quit.

Run it from the project root:

```powershell
python .\13_cinematic_lander.py
```
