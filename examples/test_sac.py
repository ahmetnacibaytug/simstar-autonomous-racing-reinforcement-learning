#Run 5 evaluation episodes on Austria: python test_sac.py austria 5

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EX_DIR = os.path.dirname(os.path.abspath(__file__))
if EX_DIR not in sys.path:
    sys.path.insert(0, EX_DIR)

from simstarEnv import SimstarEnv
import simstar


ACTION_DIM = 3
LOG_STD_MIN = -20
LOG_STD_MAX = 2

PORT = 8080
HZ = 10
SPEED_UP = 6
LOWER_SPEED_LIMIT = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_track(name: str):
    key = name.lower().replace("_", "").replace("-", "")

    mapping = {
        "austria": ["Austria", "AustriaGrandPrix"],
        "dutch": ["Dutch", "DutchGrandPrix"],
        "hungary": ["Hungary", "HungaryGrandPrix"],
    }

    if key not in mapping:
        raise ValueError("track must be one of: austria, dutch, hungary")

    for attr in mapping[key]:
        if hasattr(simstar.TrackName, attr):
            return getattr(simstar.TrackName, attr)

    raise AttributeError(f"TrackName not found for {name}: {mapping[key]}")


def scalar(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def build_state(obs) -> np.ndarray:
    return np.hstack((
        obs.angle,
        obs.track,
        obs.trackPos,
        obs.speedX,
        obs.speedY,
    )).astype(np.float32)


def make_action(raw: np.ndarray) -> np.ndarray:
    steer = float(np.clip(raw[0], -1.0, 1.0))
    throttle = float(np.clip((raw[1] + 1.0) / 2.0, 0.0, 1.0))
    brake = float(np.clip((raw[2] + 1.0) / 2.0, 0.0, 0.35))

    if throttle > 0.2 and brake > 0.2:
        if throttle >= brake:
            brake = 0.0
        else:
            throttle = 0.0

    return np.array([steer, throttle, brake], dtype=np.float32)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        with torch.no_grad():
            self.mu.bias[0] = 0.0
            self.mu.bias[1] = 0.4
            self.mu.bias[2] = -2.0

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, x):
        mu, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


def load_actor(model_path: str, state_dim: int) -> Actor:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    actor = Actor(state_dim, ACTION_DIM).to(device)

    if isinstance(ckpt, dict) and "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    elif isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    else:
        actor.load_state_dict(ckpt)

    actor.eval()
    return actor


@torch.no_grad()
def select_action(actor: Actor, state: np.ndarray) -> np.ndarray:
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    _, _, det_action = actor.sample(s)
    return det_action.squeeze(0).cpu().numpy()


def run_test(track_name: str, episodes: int, model_path: str, hard_limit: int, out_csv: str):
    print("=" * 80)
    print("SAC DETERMINISTIC TEST")
    print(f"Model   : {model_path}")
    print(f"Track   : {track_name}")
    print(f"Episodes: {episodes}")
    print("Noise   : OFF / deterministic tanh(mu)")
    print(f"Limit   : env done'a kadar | safety hard_limit={hard_limit}")
    print("=" * 80)

    env = SimstarEnv(
        track=get_track(track_name),
        port=PORT,
        synronized_mode=True,
        speed_up=SPEED_UP,
        hz=HZ,
        lower_speed_limit=LOWER_SPEED_LIMIT,
        add_agent=False,
    )

    rows = []

    try:
        obs = env.reset()
        state_dim = build_state(obs).shape[0]
        actor = load_actor(model_path, state_dim)
        print(f"[INFO] state_dim={state_dim}")
        print(f"[INFO] device={device}")
        print("[INFO] actor loaded\n")

        for ep in range(1, episodes + 1):
            obs = env.reset()
            state = build_state(obs)

            prev_lap = env.get_lap_progress()
            lap_accum = 0.0
            ep_env_r = 0.0

            speeds = []
            steers = []
            throttles = []
            brakes = []
            track_positions = []
            angles = []

            done_reason = "hard_limit"
            t0 = time.time()

            for step in range(1, hard_limit + 1):
                raw_action = select_action(actor, state)
                action = make_action(raw_action)

                next_obs, env_r, done, info = env.step(action)
                next_state = build_state(next_obs)

                cur_lap = env.get_lap_progress()
                delta = cur_lap - prev_lap
                if delta < -0.5:
                    delta += 1.0
                elif delta > 0.5:
                    delta -= 1.0

                lap_accum += delta
                prev_lap = cur_lap
                ep_env_r += float(env_r)

                speeds.append(scalar(next_obs.speedX) * 130.0)
                steers.append(float(action[0]))
                throttles.append(float(action[1]))
                brakes.append(float(action[2]))
                track_positions.append(scalar(next_obs.trackPos))
                angles.append(scalar(next_obs.angle))

                state = next_state

                if done:
                    done_reason = "env_done"
                    break

            elapsed = time.time() - t0

            row = {
                "track": track_name,
                "episode": ep,
                "lap_percent": lap_accum * 100.0,
                "env_reward": ep_env_r,
                "steps": step,
                "done_reason": done_reason,
                "avg_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
                "max_speed_kmh": float(np.max(speeds)) if speeds else 0.0,
                "avg_abs_steer": float(np.mean(np.abs(steers))) if steers else 0.0,
                "avg_throttle": float(np.mean(throttles)) if throttles else 0.0,
                "avg_brake": float(np.mean(brakes)) if brakes else 0.0,
                "max_abs_trackPos": float(np.max(np.abs(track_positions))) if track_positions else 0.0,
                "avg_abs_angle": float(np.mean(np.abs(angles))) if angles else 0.0,
                "elapsed_sec": elapsed,
            }
            rows.append(row)

            print(
                f"Ep {ep:02d}/{episodes} | "
                f"Lap {row['lap_percent']:7.1f}% | "
                f"EnvR {row['env_reward']:10.2f} | "
                f"Steps {row['steps']:5d} | "
                f"AvgSpd {row['avg_speed_kmh']:5.1f} | "
                f"MaxSpd {row['max_speed_kmh']:5.1f} | "
                f"Thr {row['avg_throttle']:.2f} | "
                f"Brk {row['avg_brake']:.2f} | "
                f"Max|TP| {row['max_abs_trackPos']:.2f} | "
                f"{row['done_reason']}"
            )

    finally:
        try:
            env.end()
        except Exception:
            pass

    df = pd.DataFrame(rows)

    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\n[CSV SAVED] {out_csv}")

    if len(df) > 0:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"MeanLap   : {df['lap_percent'].mean():.1f}%")
        print(f"MedianLap : {df['lap_percent'].median():.1f}%")
        print(f"BestLap   : {df['lap_percent'].max():.1f}%")
        print(f"WorstLap  : {df['lap_percent'].min():.1f}%")
        print(f"MeanSteps : {df['steps'].mean():.1f}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("track", type=str, choices=["austria", "dutch", "hungary"])
    parser.add_argument("episodes", type=int, nargs="?", default=5)
    parser.add_argument("--model", type=str, default=os.path.join("checkpoints", "sac_best.pt"))
    parser.add_argument("--hard-limit", type=int, default=20000)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.out is None:
        base = os.path.splitext(os.path.basename(args.model))[0]
        args.out = os.path.join("checkpoints", f"test_{base}_{args.track}.csv")

    run_test(
        track_name=args.track,
        episodes=args.episodes,
        model_path=args.model,
        hard_limit=args.hard_limit,
        out_csv=args.out,
    )


if __name__ == "__main__":
    main()
