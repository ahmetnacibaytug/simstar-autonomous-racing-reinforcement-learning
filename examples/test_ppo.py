"""
Ahmet Naci Baytuğ - 256102005 - RIS502
PPOTEST.py — PPO

Usage examples:
  python PPOTEST.py austria 5 --model checkpoints/v2_lastppo_austria_best.pt
  python PPOTEST.py dutch 5 --model checkpoints/v2_lastppo_dutch_best.pt
  python PPOTEST.py hungary 5 --model checkpoints/v2_lastppo_best.pt
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EX_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EX_DIR)
if EX_DIR not in sys.path:
    sys.path.insert(0, EX_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simstarEnv import SimstarEnv
import simstar

ACTION_DIM = 3
PORT = 8080
HZ = 10
SPEED_UP = 6
LOWER_SPEED_LIMIT = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_track(name: str):
    mapping = {
        "austria": ["Austria", "AustriaGrandPrix"],
        "dutch": ["Dutch", "DutchGrandPrix"],
        "hungary": ["Hungary", "HungaryGrandPrix"],
    }

    key = name.lower()

    if key not in mapping:
        raise ValueError("track must be one of: austria, dutch, hungary")

    for attr in mapping[key]:
        if hasattr(simstar.TrackName, attr):
            return getattr(simstar.TrackName, attr)

    raise AttributeError(f"TrackName not found for {name}: {mapping[key]}")


def build_state(obs) -> np.ndarray:
    return np.hstack((
        obs.angle,
        obs.track,
        obs.trackPos,
        obs.speedX,
        obs.speedY,
    )).astype(np.float32)


def scalar(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def make_action(raw: np.ndarray) -> np.ndarray:
    """raw [-1,1]^3 -> SimStar env action [steer, throttle, brake]."""
    raw = np.asarray(raw, dtype=np.float32)
    steer = float(np.clip(raw[0], -1.0, 1.0))
    throttle = float(np.clip((raw[1] + 1.0) / 2.0, 0.0, 1.0))
    brake = float(np.clip((raw[2] + 1.0) / 2.0, 0.0, 0.35))
    if throttle > 0.2 and brake > 0.2:
        if throttle >= brake:
            brake = 0.0
        else:
            throttle = 0.0
    return np.array([steer, throttle, brake], dtype=np.float32)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU()
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        with torch.no_grad():
            self.mu.bias[0] = 0.0
            self.mu.bias[1] = 0.4
            self.mu.bias[2] = -2.0

    def forward(self, state):
        h = self.actor(state)
        mu = torch.tanh(self.mu(h))
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std.expand_as(mu))
        dist = torch.distributions.Normal(mu, std)
        value = self.critic(state)
        return dist, value, mu

    def deterministic_action(self, state):
        h = self.actor(state)
        return torch.tanh(self.mu(h))


def load_policy(model_path: str, state_dim: int) -> ActorCritic:
    policy = ActorCritic(state_dim, ACTION_DIM).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "ac" in ckpt:
        policy.load_state_dict(ckpt["ac"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        policy.load_state_dict(ckpt["state_dict"])
    else:
        policy.load_state_dict(ckpt)
    policy.eval()
    return policy


@torch.no_grad()
def select_deterministic_action(policy: ActorCritic, state: np.ndarray) -> np.ndarray:
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    raw = policy.deterministic_action(s)
    return raw.squeeze(0).cpu().numpy().astype(np.float32)


def evaluate_episode(env, policy, hard_limit: int):
    obs = env.reset()
    state = build_state(obs)
    prev_lap = env.get_lap_progress()
    lap_accum = 0.0
    env_return = 0.0
    steps = 0
    speeds, throttles, brakes, steers, track_positions = [], [], [], [], []
    end_reason = "hard_limit"

    for _ in range(hard_limit):
        raw_action = select_deterministic_action(policy, state)
        sim_action = make_action(raw_action)
        next_obs, env_r, done, info = env.step(sim_action)
        next_state = build_state(next_obs)

        cur_lap = env.get_lap_progress()
        delta = cur_lap - prev_lap
        if delta < -0.5:
            delta += 1.0
        elif delta > 0.5:
            delta -= 1.0
        lap_accum += delta
        prev_lap = cur_lap

        speed_kmh = scalar(next_obs.speedX) * 130.0
        track_pos = scalar(next_obs.trackPos)
        env_return += float(env_r)
        speeds.append(speed_kmh)
        steers.append(float(sim_action[0]))
        throttles.append(float(sim_action[1]))
        brakes.append(float(sim_action[2]))
        track_positions.append(abs(track_pos))
        state = next_state
        steps += 1

        if isinstance(info, dict) and "end_reason" in info:
            end_reason = info.get("end_reason", "done")
        elif done:
            end_reason = "env_done"
        if done:
            break

    return {
        "lap_percent": lap_accum * 100.0,
        "env_return": env_return,
        "steps": steps,
        "avg_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "max_speed_kmh": float(np.max(speeds)) if speeds else 0.0,
        "mean_steer": float(np.mean(steers)) if steers else 0.0,
        "mean_throttle": float(np.mean(throttles)) if throttles else 0.0,
        "mean_brake": float(np.mean(brakes)) if brakes else 0.0,
        "max_abs_track_pos": float(np.max(track_positions)) if track_positions else 0.0,
        "termination_reason": end_reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("track", type=str, choices=["austria", "dutch", "hungary"])
    parser.add_argument("episodes", type=int, nargs="?", default=5)
    parser.add_argument("--model", type=str, default=os.path.join("checkpoints", "v2_lastppo_best.pt"))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--hard-limit", type=int, default=20000)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--hz", type=int, default=HZ)
    parser.add_argument("--speed-up", type=int, default=SPEED_UP)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print("\n" + "=" * 72)
    print("PPO deterministic evaluation")
    print(f"Track       : {args.track}")
    print(f"Episodes    : {args.episodes}")
    print(f"Model       : {args.model}")
    print(f"Hard limit  : {args.hard_limit}")
    print("=" * 72 + "\n")

    env = SimstarEnv(
        track=get_track(args.track),
        port=args.port,
        synronized_mode=True,
        speed_up=args.speed_up,
        hz=args.hz,
        lower_speed_limit=LOWER_SPEED_LIMIT,
        add_agent=False,
    )

    try:
        obs = env.reset()
        state_dim = build_state(obs).shape[0]
        policy = load_policy(args.model, state_dim)
        results = []
        t0 = time.time()

        for ep in range(1, args.episodes + 1):
            result = evaluate_episode(env, policy, args.hard_limit)
            result["episode"] = ep
            result["track"] = args.track
            result["model"] = os.path.basename(args.model)
            results.append(result)
            print(
                f"Ep {ep:02d} | "
                f"Lap {result['lap_percent']:8.1f}% | "
                f"EnvR {result['env_return']:10.1f} | "
                f"Steps {result['steps']:6d} | "
                f"AvgSpeed {result['avg_speed_kmh']:5.1f} km/h | "
                f"Max|TP| {result['max_abs_track_pos']:.3f} | "
                f"Reason {result['termination_reason']}"
            )

        df = pd.DataFrame(results)
        print("\n" + "-" * 72)
        print("SUMMARY")
        print(f"Mean lap      : {df['lap_percent'].mean():.1f}%")
        print(f"Median lap    : {df['lap_percent'].median():.1f}%")
        print(f"Best lap      : {df['lap_percent'].max():.1f}%")
        print(f"Worst lap     : {df['lap_percent'].min():.1f}%")
        print(f"Mean EnvR     : {df['env_return'].mean():.1f}")
        print(f"Mean steps    : {df['steps'].mean():.1f}")
        print(f"Mean speed    : {df['avg_speed_kmh'].mean():.1f} km/h")
        print(f"Mean Max|TP|  : {df['max_abs_track_pos'].mean():.3f}")
        print(f"Elapsed       : {(time.time() - t0) / 60.0:.1f} min")
        print("-" * 72)

        if args.out is None:
            safe_model = os.path.splitext(os.path.basename(args.model))[0]
            args.out = f"test_ppo_{safe_model}_{args.track}.csv"
        df.to_csv(args.out, index=False)
        print(f"\n[CSV] Saved results to: {args.out}")

    finally:
        try:
            env.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
