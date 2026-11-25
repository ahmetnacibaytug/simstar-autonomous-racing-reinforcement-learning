import os
import sys
import time
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX_DIR     = os.path.dirname(os.path.abspath(__file__)) 

if EX_DIR not in sys.path:
    sys.path.insert(0, EX_DIR)

from simstarEnv import SimstarEnv
import simstar

# ── Device Kurulumu ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dosya Yolları ────────
SAVE_FOLDER = os.path.join(EX_DIR, "checkpoints")
os.makedirs(SAVE_FOLDER, exist_ok=True)

AUSTRIA_CKPT = os.path.join(SAVE_FOLDER, "sac_austria.pt")
AUSTRIA_BEST = os.path.join(SAVE_FOLDER, "sac_austria_best.pt")
AUSTRIA_BUF  = os.path.join(SAVE_FOLDER, "sac_austria_buffer.pkl")
AUSTRIA_CSV  = os.path.join(SAVE_FOLDER, "sac_austria.csv") 

DUTCH_CKPT   = os.path.join(SAVE_FOLDER, "sac_dutch.pt")
DUTCH_BEST   = os.path.join(SAVE_FOLDER, "sac_dutch_best.pt")
DUTCH_BUF    = os.path.join(SAVE_FOLDER, "sac_dutch_buffer.pkl")
DUTCH_CSV    = os.path.join(SAVE_FOLDER, "sac_dutch.csv")   

HUNGARY_CKPT = os.path.join(SAVE_FOLDER, "sac_hungary.pt")
HUNGARY_BEST = os.path.join(SAVE_FOLDER, "sac_best.pt")  
HUNGARY_BUF  = os.path.join(SAVE_FOLDER, "sac_hungary_buffer.pkl")
HUNGARY_CSV  = os.path.join(SAVE_FOLDER, "sac_hungary.csv")  

# ── Training Config ───────────────────────────────────────────────────────────
AUSTRIA_EPISODES = 1000
DUTCH_EPISODES   = 400
HUNGARY_EPISODES = 300
MAX_STEPS        = 4000  
PORT             = 8080
HZ               = 10    
SPEED_UP         = 6     

# ── SAC Hyperparameters ───────────────────────────────────────────────────────
ACTION_DIM       = 3
LOG_STD_MIN      = -20
LOG_STD_MAX      = 2
ACTOR_LR         = 3e-4
CRITIC_LR        = 3e-4
ALPHA_LR         = 3e-4
GAMMA            = 0.99
TAU              = 0.005

BUFFER_SIZE      = 500_000 
BATCH_SIZE       = 256     
UPDATES_PER_STEP = 1       
WARMUP_STEPS     = 5000    

# Alpha Sabitleme Koruması
FREEZE_ALPHA     = True    
INIT_ALPHA       = 0.2

FINETUNE_ACTOR_LR  = 1e-4
FINETUNE_CRITIC_LR = 1e-4

# Ödül Fonksiyonu Katsayısı
W_PROGRESS  = 100.0        


def get_track(name: str):
    mapping = {
        "austria": ["Austria", "AustriaGrandPrix"],
        "dutch":   ["Dutch",   "DutchGrandPrix"],
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
    steer    = float(np.clip(raw[0], -1.0, 1.0))
    throttle = float(np.clip((raw[1] + 1.0) / 2.0, 0.0, 1.0))
    brake    = float(np.clip((raw[2] + 1.0) / 2.0, 0.0, 0.35)) 

    if throttle > 0.2 and brake > 0.2:
        if throttle >= brake: brake = 0.0
        else: throttle = 0.0

    return np.array([steer, throttle, brake], dtype=np.float32)


def get_racing_target_track_pos(obs) -> float:
    return 0.0


def estimate_turn_strength(obs) -> float:
    track = np.asarray(obs.track).reshape(-1).astype(np.float32)
    if track.size < 19:
        return 0.0

    left = np.mean(track[2:7])
    center = np.mean(track[7:12])
    right = np.mean(track[12:17])

    side_diff = abs(right - left)
    front_close = max(0.0, 0.45 - center)

    return float(np.clip(1.8 * side_diff + 1.2 * front_close, 0.0, 1.0))


def warmup_action_from_obs(obs) -> np.ndarray:
    track = np.asarray(obs.track).reshape(-1).astype(np.float32)
    if track.size < 19:
        return np.array([0.0, 0.25, -0.9], dtype=np.float32)

    left = np.mean(track[2:7])
    right = np.mean(track[12:17])
    turn = right - left

    steer = np.clip(1.5 * turn, -0.45, 0.45)
    throttle_raw = 0.25  
    brake_raw = -0.90

    return np.array([
        steer + np.random.normal(0.0, 0.08),
        throttle_raw + np.random.normal(0.0, 0.08),
        brake_raw + np.random.normal(0.0, 0.04)
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ödül Fonksiyonu
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reward(env_reward, next_obs, raw_action, delta_progress, done, lap_accum) -> float:
    if done:
        speed_kmh = scalar(next_obs.speedX) * 130.0
        if lap_accum >= 1.0:
            return float(env_reward) + 200.0  
        if speed_kmh < 40.0:
            return -200.0
        return float(env_reward) - 20.0

    track_pos   = scalar(next_obs.trackPos)
    angle       = scalar(next_obs.angle)
    speed_kmh   = scalar(next_obs.speedX) * 130.0
    steer       = float(raw_action[0])
    target_tp   = get_racing_target_track_pos(next_obs)

    r  = float(env_reward)
    r += W_PROGRESS * delta_progress

    # Viraj Yaklaşım Kontrolü ve Dinamik Hedef Hız Modeli
    turn_strength = estimate_turn_strength(next_obs)
    target_speed = 85.0 - 45.0 * turn_strength
    target_speed = np.clip(target_speed, 35.0, 85.0)

    if speed_kmh <= target_speed:
        r += 0.015 * speed_kmh
    else:
        overspeed = speed_kmh - target_speed
        r -= 0.08 * overspeed

    if turn_strength > 0.35 and speed_kmh > target_speed + 10.0:
        r -= 0.12 * (speed_kmh - target_speed)

    # Geometrik Sürüş Cezaları
    r -= 0.60 * abs(track_pos - target_tp)
    r -= 0.40 * abs(angle)
    r -= 0.20 * abs(steer)

    # Fren Ödüllendirme Stratejisi
    brake = float(np.clip((raw_action[2] + 1.0) / 2.0, 0.0, 0.35))
    if turn_strength > 0.35 and speed_kmh > target_speed:
        r += 0.15 * max(brake, 0.0)
    else:
        r -= 0.10 * max(brake, 0.0)

    if abs(track_pos) > 0.82:
        r -= 5.0
        
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#  Networks
# ═══════════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU(),
        )
        self.mu      = nn.Linear(256, action_dim)
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
        std    = log_std.exp()
        dist   = torch.distributions.Normal(mu, std)
        z      = dist.rsample()
        action = torch.tanh(z)
        log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),                    nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.q(torch.cat([s, a], dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
#  Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.cap = capacity
        self.ptr = self.size = 0
        self.s  = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.a  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r  = np.zeros((capacity, 1),          dtype=np.float32)
        self.ns = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.d  = np.zeros((capacity, 1),          dtype=np.float32)

    def push(self, s, a, r, ns, d):
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.ns[self.ptr] = ns
        self.d[self.ptr]  = d
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda x: torch.tensor(x[idx], dtype=torch.float32, device=device)
        return to(self.s), to(self.a), to(self.r), to(self.ns), to(self.d)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                "ptr": self.ptr, "size": self.size,
                "s": self.s, "a": self.a, "r": self.r, "ns": self.ns, "d": self.d
            }, f)
        print(f"[BUFFER] Hafıza diske kilitlendi: {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.ptr = data["ptr"]
            self.size = data["size"]
            self.s = data["s"]
            self.a = data["a"]
            self.r = data["r"]
            self.ns = data["ns"]
            self.d = data["d"]
            print(f"[BUFFER] Tecrübe hafızası geri yüklendi: {filepath} ({self.size} adım)")

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════════════════
#  SAC Agent Sınıfı
# ═══════════════════════════════════════════════════════════════════════════════

class SACAgent:
    def __init__(self, state_dim, action_dim=ACTION_DIM, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR):
        self.actor          = Actor(state_dim, action_dim).to(device)
        self.critic1        = Critic(state_dim, action_dim).to(device)
        self.critic2        = Critic(state_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(INIT_ALPHA), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt      = optim.Adam([self.log_alpha], lr=ALPHA_LR)
        self.target_entropy = -float(action_dim)
        self.memory         = ReplayBuffer(BUFFER_SIZE, state_dim, action_dim)

    @property
    def alpha(self):
        return torch.tensor(INIT_ALPHA, dtype=torch.float32, device=device) if FREEZE_ALPHA else self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(s)
            else:
                action, _, _ = self.actor.sample(s)
        return action.squeeze(0).cpu().numpy()

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        s, a, r, ns, d = self.memory.sample(BATCH_SIZE)

        with torch.no_grad():
            na, nlogp, _ = self.actor.sample(ns)
            nq1 = self.target_critic1(ns, na)
            nq2 = self.target_critic2(ns, na)
            target_q = r + (1 - d) * GAMMA * (torch.min(nq1, nq2) - self.alpha.detach() * nlogp)

        q1, q2 = self.critic1(s, a), self.critic2(s, a)
        c1_loss = F.mse_loss(q1, target_q)
        c2_loss = F.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad(); c1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 5.0)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad(); c2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 5.0)
        self.critic2_opt.step()

        new_a, logp, _ = self.actor.sample(s)
        q_new = torch.min(self.critic1(s, new_a), self.critic2(s, new_a))
        actor_loss = (self.alpha.detach() * logp - q_new).mean()

        self.actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        if not FREEZE_ALPHA:
            aloss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); aloss.backward(); self.alpha_opt.step()

        for src, tgt in [(self.critic1, self.target_critic1), (self.critic2, self.target_critic2)]:
            for sp, tp in zip(src.parameters(), tgt.parameters()):
                tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)

        return {"actor": float(actor_loss), "c1": float(c1_loss), "c2": float(c2_loss), "alpha": float(self.alpha.item())}

    def save(self, path, meta: dict):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().item(),
            **meta,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_critic1.load_state_dict(ckpt["target_critic1"])
        self.target_critic2.load_state_dict(ckpt["target_critic2"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic1_opt.load_state_dict(ckpt["critic1_opt"])
        self.critic2_opt.load_state_dict(ckpt["critic2_opt"])
        if "log_alpha" in ckpt:
            self.log_alpha.data = torch.tensor(ckpt["log_alpha"], device=device)
        return {k: ckpt[k] for k in ["step", "episode", "ep_count", "best_score", "best_env_rew", "best_lap"] if k in ckpt}

    def load_weights_only(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_critic1.load_state_dict(ckpt["target_critic1"])
        self.target_critic2.load_state_dict(ckpt["target_critic2"])
        print(f"[TRANSFER] SAC Master Geometri Ağı Yüklendi: {path}")


def write_csv(columns, values, filepath):
    pd.DataFrame([values], columns=columns).to_csv(filepath, mode="a", index=False, header=not os.path.isfile(filepath))


# ═══════════════════════════════════════════════════════════════════════════════
#  Ana Kesintisiz Sürüş Döngüsü
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(track_str, episodes, ckpt_path, best_path, csv_path, buf_path, transfer_path=None, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR):
    print(f"\n{'='*70}")
    print(f"SAC MASTER ENGINE fazı: {track_str.upper()} | Hedef: {episodes} episode")
    print(f"{'='*70}\n")

    env = SimstarEnv(
        track=get_track(track_str), port=PORT, synronized_mode=True,
        speed_up=SPEED_UP, hz=HZ, lower_speed_limit=5, add_agent=False,
    )

    obs       = env.reset()
    state_dim = build_state(obs).shape[0]

    print("\n[KALİBRASYON] İlk Gözlem (Observation) Analizi:")
    print(f"  -> angle (Açı): {obs.angle}")
    print(f"  -> trackPos (Pozisyon): {obs.trackPos}")
    print(f"  -> speedX (Ham Hız): {obs.speedX}")
    print(f"  -> track (Sensör Min/Max): {np.min(obs.track)} / {np.max(obs.track)}\n")

    agent = SACAgent(state_dim=state_dim, action_dim=ACTION_DIM, actor_lr=actor_lr, critic_lr=critic_lr)

    step          = 0
    start_ep      = 0
    ep_count      = 0
    best_score    = -1e18
    best_env_rew  = -1e18
    best_lap      = -1e18
    datalog       = defaultdict(list)
    current_alpha = INIT_ALPHA

    if os.path.exists(ckpt_path):
        print(f"[LOAD] Checkpoint bulundu -> {ckpt_path}")
        meta = agent.load(ckpt_path)
        step         = meta.get("step", 0)
        start_ep     = meta.get("episode", -1) + 1
        ep_count     = meta.get("ep_count", 0)
        best_score   = meta.get("best_score", -1e18)
        best_env_rew = meta.get("best_env_rew", -1e18)
        best_lap     = meta.get("best_lap", -1e18)
        agent.memory.load(buf_path)
        
        if os.path.exists(csv_path):
            try:
                df_old = pd.read_csv(csv_path)
                datalog["env_r"] = df_old["env_reward"].tolist()
                datalog["lap"] = df_old["lap_progress"].tolist()
                print(f"[RESUME] Datalog geçmişi başarıyla bağlandı ({len(datalog['lap'])} ep)")
            except Exception: pass
    elif transfer_path and os.path.exists(transfer_path):
        agent.load_weights_only(transfer_path)

    interrupted   = False
    t0            = time.time()
    episode       = start_ep - 1

    try:
        for episode in range(start_ep, episodes):
            obs   = env.reset()
            state = build_state(obs)

            ep_env_r = ep_train_r = 0.0
            actor_losses = []
            prev_lap = env.get_lap_progress()
            lap_accum = 0.0
            ep_step = 0 

            for _ in range(MAX_STEPS):
                ep_step += 1
                
                if step < WARMUP_STEPS and transfer_path is None:
                    action = warmup_action_from_obs(obs)
                else:
                    action = agent.select_action(state, evaluate=False)
                    
                sim_action = make_action(action)

                next_obs, env_r, done, _ = env.step(sim_action)
                next_state = build_state(next_obs)

                cur_lap = env.get_lap_progress()
                delta   = cur_lap - prev_lap
                if   delta < -0.5: delta += 1.0
                elif delta >  0.5: delta -= 1.0
                lap_accum += delta
                prev_lap   = cur_lap

                train_r = compute_reward(env_r, next_obs, action, delta, done, lap_accum)

                # Tur geçiş kontrolü
                prev_lap_count = int((lap_accum - delta) // 1.0)
                cur_lap_count  = int(lap_accum // 1.0)
                if cur_lap_count > prev_lap_count:
                    train_r += 100.0

                agent.memory.push(state, action, train_r, next_state, float(done))

                if len(agent.memory) >= BATCH_SIZE:
                    for _ in range(UPDATES_PER_STEP):
                        losses = agent.update()
                        if losses:
                            actor_losses.append(losses["actor"])
                            current_alpha = losses.get("alpha", current_alpha)

                ep_env_r   += float(env_r)
                ep_train_r += float(train_r)
                state       = next_state
                obs         = next_obs 
                step       += 1
                if done:
                    break

            ep_count += 1

            def avg20(key, val):
                h = datalog[key][-19:]
                return float(np.mean(h + [val])) if h else float(val)

            avg_env = avg20("env_r",  ep_env_r)
            avg_lap = avg20("lap",    lap_accum)
            datalog["env_r"].append(ep_env_r)
            datalog["lap"].append(lap_accum)

            a_loss = float(np.mean(actor_losses)) if actor_losses else 0.0

            if lap_accum > best_lap + 0.005:
                best_score   = ep_train_r
                best_env_rew = ep_env_r
                best_lap     = lap_accum
                agent.save(best_path, dict(step=step, episode=episode, ep_count=ep_count, best_score=best_score, best_env_rew=best_env_rew, best_lap=best_lap))
                print(f"\n★ SAC Rekor Tur | Lap={best_lap*100:.1f}% | EnvRew={best_env_rew:.2f} | Alpha={current_alpha:.4f}")

            agent.save(ckpt_path, dict(step=step, episode=episode, ep_count=ep_count, best_score=best_score, best_env_rew=best_env_rew, best_lap=best_lap))

            elapsed_min = (time.time() - t0) / 60.0
            print(f"Ep {ep_count}/{episodes} | EnvR {ep_env_r:7.2f} | Avg20Env {avg_env:7.2f} | Lap {lap_accum*100:5.1f}% | Avg20Lap {avg_lap*100:5.1f}% | Best {best_lap*100:5.1f}% | Alpha {current_alpha:.4f} | Steps {step} | {elapsed_min:.1f}m")

            write_csv(["episode","track","env_reward","avg20_env","lap_progress","avg20_lap","actor_loss","steps","best_lap"],
                      [ep_count, track_str, ep_env_r, avg_env, lap_accum, avg_lap, a_loss, step, best_lap], csv_path)

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[INTERRUPTED] Kesme sinyali alındı. Hafıza diske kilitleniyor...")
    finally:
        agent.save(ckpt_path, dict(step=step, episode=episode, ep_count=ep_count, best_score=best_score, best_env_rew=best_env_rew, best_lap=best_lap))
        agent.memory.save(buf_path)
        try: env.end()
        except Exception: pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("track", type=str, choices=["austria", "dutch", "hungary"])
    args = parser.parse_args()

    if args.track == "austria":
        run_training(track_str="austria", episodes=AUSTRIA_EPISODES, ckpt_path=AUSTRIA_CKPT, best_path=AUSTRIA_BEST, csv_path=AUSTRIA_CSV, buf_path=AUSTRIA_BUF)
    elif args.track == "dutch":
        run_training(track_str="dutch", episodes=DUTCH_EPISODES, ckpt_path=DUTCH_CKPT, best_path=DUTCH_BEST, csv_path=DUTCH_CSV, buf_path=DUTCH_BUF, transfer_path=AUSTRIA_BEST, actor_lr=FINETUNE_ACTOR_LR, critic_lr=FINETUNE_CRITIC_LR)
    elif args.track == "hungary":
        run_training(track_str="hungary", episodes=HUNGARY_EPISODES, ckpt_path=HUNGARY_CKPT, best_path=HUNGARY_BEST, csv_path=HUNGARY_CSV, buf_path=HUNGARY_BUF, transfer_path=DUTCH_BEST, actor_lr=FINETUNE_ACTOR_LR, critic_lr=FINETUNE_CRITIC_LR)