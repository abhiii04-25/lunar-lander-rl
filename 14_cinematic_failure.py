"""Cinematic Lunar Lander FAILURE replay matching the success viewer.

Uses the same sunset background, rocket art, HUD, and original Gymnasium physics.
It searches for a real failed attempt from the baseline PPO agent, then replays it.
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
import pygame
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
BACKGROUND_PATH = ASSETS / "landing_base.png"
ROCKET_PATH = ASSETS / "rocket_lander.png"

# Use the weaker baseline policy for a clear comparison against the successful 300k viewer.
MODEL_CANDIDATES = [
    ("PPO baseline 100k", ROOT / "models" / "ppo_lunar_lander_baseline.zip"),
]
SEEDS_TO_SEARCH = range(100, 140)
WIDTH, HEIGHT = 1152, 768
FPS = 60

# Visual target center: aligned with the helipad in the background artwork.
PAD_X = WIDTH * 0.50
PAD_Y = HEIGHT * 0.782

INK = (15, 20, 42)
WHITE = (247, 249, 255)
MINT = (65, 242, 174)
GOLD = (255, 188, 63)
CORAL = (255, 103, 120)
AQUA = (65, 225, 232)
CONFETTI_COLORS = [GOLD, CORAL, AQUA, MINT, (192, 140, 255), WHITE, (255, 130, 48)]


def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    preferred = "bahnschrift" if bold else "segoeui"
    return pygame.font.SysFont(preferred, size, bold=bold)


def simulate_episode(model: PPO, env: gym.Env, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    last_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        last_reward = float(reward)
        steps += 1
        if terminated or truncated:
            # LunarLander returns a positive terminal reward for a safe landing
            # and a negative terminal reward for a crash. A timeout is not a win.
            success = bool(terminated and not truncated and last_reward > 0)
            return {
                "seed": seed,
                "reward": total_reward,
                "steps": steps,
                "success": success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "terminal_reward": float(last_reward),
            }


def find_failed_attempt() -> tuple[PPO, str, dict]:
    """Find a genuine non-successful baseline episode; prefer a real crash."""
    available = [(name, path) for name, path in MODEL_CANDIDATES if path.exists()]
    if not available:
        expected = ROOT / "models" / "ppo_lunar_lander_baseline.zip"
        raise FileNotFoundError(
            f"Baseline model not found: {expected}. Keep the models folder next to this script."
        )

    env = gym.make("LunarLander-v3")
    crashes: list[tuple[str, Path, dict]] = []
    other_failures: list[tuple[str, Path, dict]] = []
    try:
        print("Searching the baseline model for a REAL failed attempt (seeds 100–139)...")
        for name, path in available:
            model = PPO.load(str(path), device="cpu")
            for seed in SEEDS_TO_SEARCH:
                result = simulate_episode(model, env, seed)
                # Success is confirmed only when the episode terminates naturally
                # with the positive terminal reward used by LunarLander.
                if result["success"]:
                    continue
                candidate = (name, path, result)
                if result["terminated"] and not result["truncated"] and result["terminal_reward"] < 0:
                    crashes.append(candidate)
                else:
                    other_failures.append(candidate)
            print(f"Checked {len(SEEDS_TO_SEARCH)} deterministic episodes for {name}.")
    finally:
        env.close()

    # A crash is visually clearest. If none exists in this seed set, select a real
    # timeout/other failed attempt and label it as a mission failure, never a crash.
    pool = crashes if crashes else other_failures
    if not pool:
        raise RuntimeError(
            "No failed attempt was found in seeds 100–139. Increase SEEDS_TO_SEARCH "
            "to range(140, 220) and run again; this viewer never fabricates a failure."
        )
    name, path, chosen = min(pool, key=lambda entry: entry[2]["reward"])
    chosen_model = PPO.load(str(path), device="cpu")
    chosen["is_crash"] = bool(chosen["terminated"] and not chosen["truncated"] and chosen["terminal_reward"] < 0)
    chosen["failure_title"] = "CRASH DETECTED" if chosen["is_crash"] else "MISSION FAILED"
    chosen["failure_subtitle"] = (
        "UNSAFE IMPACT  |  BASELINE POLICY" if chosen["is_crash"]
        else "TIME LIMIT REACHED  |  NO SAFE LANDING"
    )
    print("\nSelected failure replay:")
    print(f"  Model: {name}")
    print(f"  Seed: {chosen['seed']}")
    print(f"  Original reward: {chosen['reward']:.2f}")
    print(f"  Steps: {chosen['steps']}")
    print(f"  Result: {chosen['failure_title']}")
    return chosen_model, name, chosen


def draw_panel(screen: pygame.Surface, rect: pygame.Rect, alpha: int = 205,
               color: tuple[int, int, int] = INK, radius: int = 20) -> None:
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (*color, alpha), panel.get_rect(), border_radius=radius)
    pygame.draw.rect(panel, (255, 255, 255, 35), panel.get_rect(), width=1, border_radius=radius)
    screen.blit(panel, rect.topleft)


def draw_text(screen: pygame.Surface, text: str, pos: tuple[int, int], size: int,
              color: tuple[int, int, int] = WHITE, bold: bool = False,
              shadow: bool = False) -> None:
    font = get_font(size, bold)
    rendered = font.render(text, True, color)
    if shadow:
        shade = font.render(text, True, (8, 10, 24))
        screen.blit(shade, (pos[0] + 2, pos[1] + 3))
    screen.blit(rendered, pos)


def draw_meter(screen: pygame.Surface, x: int, y: int, w: int, label: str,
               value: float, limit: float, color: tuple[int, int, int]) -> None:
    draw_text(screen, label, (x, y), 14, (203, 214, 239), True)
    pygame.draw.rect(screen, (50, 57, 83), (x, y + 23, w, 7), border_radius=4)
    fraction = max(0.0, min(1.0, abs(value) / max(limit, 1e-6)))
    pygame.draw.rect(screen, color, (x, y + 23, int(w * fraction), 7), border_radius=4)


def draw_hud(screen: pygame.Surface, episode_reward: float, step: int,
             obs, action: int, model_name: str) -> None:
    # Top-left brand and live score: reminiscent of a premium game HUD.
    draw_panel(screen, pygame.Rect(24, 22, 330, 132), alpha=218)
    draw_text(screen, "ORBITAL  /  AI FLIGHT SYSTEM", (44, 39), 14, AQUA, True)
    draw_text(screen, "MISSION SCORE", (44, 66), 14, (206, 218, 242), True)
    score_color = MINT if episode_reward >= 0 else CORAL
    draw_text(screen, f"{episode_reward:+.0f}", (42, 84), 43, score_color, True)
    pygame.draw.line(screen, (61, 234, 183), (44, 137), (314, 137), 2)

    draw_panel(screen, pygame.Rect(WIDTH - 300, 22, 276, 132), alpha=205)
    draw_text(screen, "AUTOPILOT  /  PPO", (WIDTH - 278, 40), 17, WHITE, True)
    draw_text(screen, model_name[:29], (WIDTH - 278, 66), 13, (204, 215, 239))
    draw_text(screen, f"STEP  {step:04d}", (WIDTH - 278, 91), 14, GOLD, True)
    status = ["COAST", "LEFT THRUST", "MAIN ENGINE", "RIGHT THRUST"][int(action)]
    draw_text(screen, status, (WIDTH - 278, 117), 12, AQUA, True)

    # Compact telemetry card at the bottom left.
    draw_panel(screen, pygame.Rect(24, HEIGHT - 116, 280, 90), alpha=195)
    draw_meter(screen, 42, HEIGHT - 100, 106, "HORIZONTAL SPEED", float(obs[2]), 1.0, AQUA)
    draw_meter(screen, 166, HEIGHT - 100, 106, "VERTICAL SPEED", float(obs[3]), 1.0, GOLD)


def draw_engine_flame(screen: pygame.Surface, center: tuple[float, float],
                      angle: float, ship_height: int, action: int, t: float) -> None:
    """Draw an animated, angle-aware main-engine plume below the lander."""
    if action != 2:
        return
    # Observation angle zero means the rocket points upright. The engine is
    # below the center, along the rocket's local down axis.
    down_x, down_y = math.sin(angle), math.cos(angle)
    side_x, side_y = math.cos(angle), -math.sin(angle)
    nozzle_x = center[0] + down_x * ship_height * 0.30
    nozzle_y = center[1] + down_y * ship_height * 0.30
    plume_length = 35 + 14 * (0.5 + 0.5 * math.sin(t * 22))
    tip = (nozzle_x + down_x * plume_length, nozzle_y + down_y * plume_length)
    base_left = (nozzle_x + side_x * 10, nozzle_y + side_y * 10)
    base_right = (nozzle_x - side_x * 10, nozzle_y - side_y * 10)
    pygame.draw.polygon(screen, (255, 105, 34), [base_left, base_right, tip])
    inner_left = (nozzle_x + side_x * 5, nozzle_y + side_y * 5)
    inner_right = (nozzle_x - side_x * 5, nozzle_y - side_y * 5)
    inner_tip = (nozzle_x + down_x * plume_length * 0.76,
                 nozzle_y + down_y * plume_length * 0.76)
    pygame.draw.polygon(screen, (255, 218, 104), [inner_left, inner_right, inner_tip])


def draw_ship(screen: pygame.Surface, rocket: pygame.Surface, obs, action: int,
              t: float) -> tuple[float, float]:
    h = 184
    w = int(h * rocket.get_width() / rocket.get_height())
    ship = pygame.transform.smoothscale(rocket, (w, h))
    # x=0 and y=0 are near the lander's target pad in Gymnasium's observation.
    cx = PAD_X + float(obs[0]) * WIDTH * 0.47
    cy = PAD_Y - h * 0.48 - float(obs[1]) * HEIGHT * 0.50
    angle = float(obs[4])
    draw_engine_flame(screen, (cx, cy), angle, h, action, t)
    rotated = pygame.transform.rotozoom(ship, -math.degrees(angle), 1.0)
    screen.blit(rotated, rotated.get_rect(center=(round(cx), round(cy))))
    return cx, cy


class Confetti:
    def __init__(self, x: float, y: float, amount: int = 220):
        self.particles = []
        for _ in range(amount):
            angle = random.uniform(-math.pi * 0.98, -math.pi * 0.02)
            speed = random.uniform(90, 430)
            self.particles.append({
                "x": x + random.uniform(-10, 10),
                "y": y + random.uniform(-5, 8),
                "vx": math.cos(angle) * speed + random.uniform(-45, 45),
                "vy": math.sin(angle) * speed,
                "gravity": random.uniform(180, 330),
                "size": random.randint(4, 10),
                "color": random.choice(CONFETTI_COLORS),
                "angle": random.uniform(0, 360),
                "spin": random.uniform(-500, 500),
            })

    def update(self, dt: float) -> None:
        for p in self.particles:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["vy"] += p["gravity"] * dt
            p["vx"] *= 0.994
            p["angle"] += p["spin"] * dt

    def draw(self, screen: pygame.Surface) -> None:
        for p in self.particles:
            x, y, size = int(p["x"]), int(p["y"]), p["size"]
            if x < -20 or x > WIDTH + 20 or y < -20 or y > HEIGHT + 20:
                continue
            # Alternating little rectangles and triangles look like party confetti.
            if int(p["angle"] // 30) % 2 == 0:
                piece = pygame.Surface((size * 2, size), pygame.SRCALPHA)
                pygame.draw.rect(piece, p["color"], (0, 0, size * 2, size), border_radius=2)
                piece = pygame.transform.rotate(piece, p["angle"])
                screen.blit(piece, piece.get_rect(center=(x, y)))
            else:
                points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
                pygame.draw.polygon(screen, p["color"], points)


def draw_celebration(screen: pygame.Surface, confetti: Confetti,
                     reward: float, elapsed: float) -> None:
    # Subtle dark veil keeps the announcement legible without hiding the scene.
    veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    veil.fill((8, 11, 28, 45))
    screen.blit(veil, (0, 0))
    confetti.draw(screen)

    panel = pygame.Rect(WIDTH // 2 - 260, 168, 520, 158)
    draw_panel(screen, panel, alpha=224, color=(12, 23, 50), radius=26)
    pygame.draw.rect(screen, (61, 234, 183), panel, width=2, border_radius=26)
    title = get_font(47, True).render("TOUCHDOWN!", True, (255, 211, 94))
    screen.blit(title, title.get_rect(center=(WIDTH // 2, 220)))
    subtitle = get_font(20, True).render("SAFE LANDING  |  PPO AUTOPILOT", True, WHITE)
    screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, 265)))
    result = get_font(17, True).render(f"MISSION REWARD  {reward:+.1f}", True, MINT)
    screen.blit(result, result.get_rect(center=(WIDTH // 2, 298)))

    # Tiny golden sparkles around the headline.
    for i in range(14):
        phase = elapsed * 3 + i * 0.7
        x = WIDTH / 2 + math.cos(phase) * (245 + 10 * math.sin(phase * 2))
        y = 245 + math.sin(phase * 1.4) * 84
        r = 2 + int((1 + math.sin(phase * 5)) * 2)
        pygame.draw.circle(screen, GOLD, (int(x), int(y)), r)
    draw_text(screen, "Landing confirmed. Celebration protocol engaged.",
              (WIDTH // 2 - 200, HEIGHT - 58), 16, WHITE, True)



class FailureFX:
    """Visual-only sparks and smoke used at the confirmed failure endpoint."""
    def __init__(self, x: float, y: float, crash: bool):
        self.crash = crash
        self.particles = []
        count = 100 if crash else 42
        for _ in range(count):
            angle = random.uniform(-math.pi, math.pi)
            speed = random.uniform(50, 300 if crash else 150)
            self.particles.append({
                "x": x + random.uniform(-18, 18),
                "y": y + random.uniform(-18, 18),
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed - random.uniform(0, 90),
                "life": random.uniform(0.8, 2.8),
                "size": random.randint(3, 10 if crash else 7),
                "smoke": random.random() < (0.36 if crash else 0.10),
                "color": random.choice([CORAL, GOLD, (255, 70, 45), WHITE]),
            })

    def update(self, dt: float) -> None:
        for p in self.particles:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["vy"] += (70 if p["smoke"] else 190) * dt
            p["vx"] *= 0.985
            p["life"] -= dt
            if p["smoke"]:
                p["size"] += dt * 3

    def draw(self, screen: pygame.Surface) -> None:
        for p in self.particles:
            if p["life"] <= 0:
                continue
            radius = max(2, int(p["size"]))
            if p["smoke"]:
                puff = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
                alpha = max(0, min(110, int(110 * min(1, p["life"]))))
                pygame.draw.circle(puff, (41, 43, 57, alpha), (radius * 2, radius * 2), radius)
                screen.blit(puff, puff.get_rect(center=(int(p["x"]), int(p["y"]))))
            else:
                pygame.draw.circle(screen, p["color"], (int(p["x"]), int(p["y"])), radius // 2 + 1)


def draw_failure_overlay(screen: pygame.Surface, title: str, subtitle: str,
                         reward: float, elapsed: float, is_crash: bool) -> None:
    """Same centered glass-panel style as the success viewer, but danger-red."""
    veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    veil.fill((28, 4, 15, 82 if is_crash else 58))
    screen.blit(veil, (0, 0))

    panel = pygame.Rect(WIDTH // 2 - 275, 168, 550, 166)
    draw_panel(screen, panel, alpha=232, color=(34, 13, 28), radius=26)
    pulse = int(185 + 55 * (0.5 + 0.5 * math.sin(elapsed * 5.0)))
    pygame.draw.rect(screen, (255, 66, 92), panel, width=3, border_radius=26)
    pygame.draw.line(screen, (pulse, 48, 68), (panel.x + 26, panel.y + 12),
                     (panel.right - 26, panel.y + 12), 3)

    title_font = get_font(37 if len(title) > 15 else 42, True)
    title_surface = title_font.render(title, True, (255, 91, 120))
    screen.blit(title_surface, title_surface.get_rect(center=(WIDTH // 2, 218)))
    sub = get_font(18, True).render(subtitle, True, WHITE)
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, 265)))
    score_color = GOLD if reward >= 0 else CORAL
    result = get_font(16, True).render(f"ORIGINAL MISSION REWARD  {reward:+.1f}", True, score_color)
    screen.blit(result, result.get_rect(center=(WIDTH // 2, 300)))

    # Small warning beacons pulse around the result card.
    for side in (-1, 1):
        bx = WIDTH // 2 + side * 302
        by = 244
        r = 5 + int(3 * (0.5 + 0.5 * math.sin(elapsed * 6 + side)))
        pygame.draw.circle(screen, (255, 48, 76), (bx, by), r + 4, 1)
        pygame.draw.circle(screen, (255, 70, 92), (bx, by), r)

    hint = get_font(15, True).render("PRESS SPACE TO REPLAY  |  ESC TO EXIT", True, WHITE)
    screen.blit(hint, hint.get_rect(center=(WIDTH // 2, HEIGHT - 28)))

def main() -> None:
    if not BACKGROUND_PATH.exists() or not ROCKET_PATH.exists():
        raise FileNotFoundError(
            "Visual assets not found. Put the same assets/ folder used by "
            "13_cinematic_lander.py next to this script. Missing: "
            f"{BACKGROUND_PATH if not BACKGROUND_PATH.exists() else ROCKET_PATH}"
        )

    model, model_name, chosen = find_failed_attempt()
    pygame.init()
    pygame.display.set_caption("ORBITAL // Cinematic Lunar Lander — FAILURE REPLAY")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    background = pygame.image.load(str(BACKGROUND_PATH)).convert()
    background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
    rocket_image = pygame.image.load(str(ROCKET_PATH)).convert_alpha()

    env = gym.make("LunarLander-v3")
    replay_seed = int(chosen["seed"])
    obs, _ = env.reset(seed=replay_seed)
    total_reward = 0.0
    steps = 0
    action = 0
    last_reward = 0.0
    terminated = truncated = False
    done = False
    success = False
    failure_started = False
    failure_fx = None
    failure_start_time = 0.0
    running = True
    start_clock = time.perf_counter()

    print("\nStarting cinematic failure replay with matching success-viewer visuals...")
    print(f"Model: {model_name} | seed: {replay_seed} | expected: {chosen['failure_title']}")

    try:
        while running:
            now = time.perf_counter()
            dt = min(clock.tick(FPS) / 1000.0, 0.05)
            elapsed = now - start_clock

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and done:
                    obs, _ = env.reset(seed=replay_seed)
                    total_reward = 0.0
                    steps = 0
                    action = 0
                    last_reward = 0.0
                    terminated = truncated = done = success = False
                    failure_started = False
                    failure_fx = None
                    failure_start_time = 0.0
                    start_clock = time.perf_counter()

            if not running:
                break

            if not done:
                action_arr, _ = model.predict(obs, deterministic=True)
                action = int(action_arr)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                last_reward = float(reward)
                steps += 1
                done = bool(terminated or truncated)
                success = bool(terminated and not truncated and last_reward > 0)
                if done:
                    if success:
                        print(f"This replay unexpectedly landed safely: reward={total_reward:.2f}")
                    elif truncated:
                        print(f"MISSION FAILED (timeout). reward={total_reward:.2f}")
                    else:
                        print(f"CRASH DETECTED. reward={total_reward:.2f}")

            screen.blit(background, (0, 0))

            # Match the success viewer's golden helipad glow, then tint it red
            # in the failure viewer to signal a dangerous approach.
            glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            danger_pulse = 18 + int(12 * (0.5 + 0.5 * math.sin(elapsed * 4.0)))
            pygame.draw.ellipse(glow, (255, 35, 67, danger_pulse),
                                (PAD_X - 270, PAD_Y - 58, 540, 116))
            pygame.draw.ellipse(glow, (255, 75, 92, 62),
                                (PAD_X - 190, PAD_Y - 35, 380, 70), 2)
            screen.blit(glow, (0, 0))

            cx, cy = draw_ship(screen, rocket_image, obs, action, elapsed)
            draw_hud(screen, total_reward, steps, obs, action, model_name + " • FAILURE REPLAY")

            if done and not success:
                is_crash = bool(terminated and not truncated and last_reward < 0)
                if not failure_started:
                    failure_fx = FailureFX(cx, cy + 24, is_crash)
                    failure_started = True
                    failure_start_time = time.perf_counter()
                    failure_title = "CRASH DETECTED" if is_crash else "MISSION FAILED"
                    failure_subtitle = (
                        "UNSAFE IMPACT  |  BASELINE POLICY" if is_crash
                        else "TIME LIMIT REACHED  |  NO SAFE LANDING"
                    )
                if failure_fx is not None:
                    failure_fx.update(dt)
                    failure_fx.draw(screen)
                draw_failure_overlay(
                    screen, failure_title, failure_subtitle, total_reward,
                    time.perf_counter() - failure_start_time, is_crash
                )
            elif done and success:
                # This should not happen because seed search selects a failed attempt.
                draw_panel(screen, pygame.Rect(WIDTH // 2 - 260, 168, 520, 158),
                           alpha=224, color=(12, 23, 50), radius=26)
                draw_text(screen, "UNEXPECTED SAFE LANDING", (WIDTH // 2 - 220, 218), 30, MINT, True)
                draw_text(screen, "This replay did not fail; choose another seed.",
                          (WIDTH // 2 - 205, 264), 15, WHITE)
                draw_text(screen, "PRESS SPACE TO REPLAY  |  ESC TO EXIT",
                          (WIDTH // 2 - 185, HEIGHT - 28), 14, WHITE, True)

            pygame.display.flip()

    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Could not launch cinematic viewer: {exc}")
        raise
