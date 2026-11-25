"""
Ahmet Naci Baytuğ - 256102005 - RIS502
PPOPPO.py — PPO
"""
import os
import sys
import time
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

# ── Cihaz Yapılandırması (Device Setup) ───────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dosya Yolları ve Log Yapılandırması ────────────────────────────────────────
SAVE_FOLDER = os.path.join(EX_DIR, "checkpoints")
os.makedirs(SAVE_FOLDER, exist_ok=True)

AUSTRIA_CKPT = os.path.join(SAVE_FOLDER, "v2_lastppo_austria.pt")
AUSTRIA_BEST = os.path.join(SAVE_FOLDER, "v2_lastppo_austria_best.pt")
AUSTRIA_CSV  = os.path.join(SAVE_FOLDER, "v2_lastppo_austria.csv")

DUTCH_CKPT   = os.path.join(SAVE_FOLDER, "v2_lastppo_dutch.pt")
DUTCH_BEST   = os.path.join(SAVE_FOLDER, "v2_lastppo_dutch_best.pt")
DUTCH_CSV    = os.path.join(SAVE_FOLDER, "v2_lastppo_dutch.csv")

HUNGARY_CKPT = os.path.join(SAVE_FOLDER, "v2_lastppo_hungary.pt")
HUNGARY_BEST = os.path.join(SAVE_FOLDER, "v2_lastppo_best.pt") 
HUNGARY_CSV  = os.path.join(SAVE_FOLDER, "v2_lastppo_hungary.csv")

# ── Eğitim ve Simülasyon Konfigürasyonu ────────────────────────────────────────
AUSTRIA_EPISODES = 3000
DUTCH_EPISODES   = 3000
HUNGARY_EPISODES = 3000
MAX_STEPS        = 4000  
PORT             = 8080
HZ               = 10    
SPEED_UP         = 6     

# ── PPO Hiperparametreleri (Hyperparameters) ──────────────────────────────────
ACTION_DIM        = 3
LR                = 3e-4
FINETUNE_LR       = 1e-4   
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
PPO_CLIP          = 0.2
PPO_EPOCHS        = 10
BATCH_SIZE        = 256     
TRAIN_EVERY_STEPS = 4096   # MADDE 4 DÜZELTMESİ: Daha geniş ve kararlı on-policy rollout

W_PROGRESS        = 100.0   


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Ödül Fonksiyonu (Reward Shaping)
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

    r -= 0.60 * abs(track_pos - target_tp)
    r -= 0.40 * abs(angle)
    r -= 0.20 * abs(steer)

    brake = float(np.clip((raw_action[2] + 1.0) / 2.0, 0.0, 0.35))
    if turn_strength > 0.35 and speed_kmh > target_speed:
        r += 0.15 * max(brake, 0.0)
    else:
        r -= 0.10 * max(brake, 0.0)

    if abs(track_pos) > 0.82:
        r -= 5.0
        
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#  PPO Network Mimarisi (Actor-Critic Framework)
# ═══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU()
        )
        self.mu      = nn.Linear(256, action_dim)
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
        h_act = self.actor(state)
        mu = torch.tanh(self.mu(h_act))
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std.expand_as(mu))
        dist = torch.distributions.Normal(mu, std)
        
        # MADDE 2 DÜZELTMESİ: Olasılık kayıplarını ve çelişkileri azaltmak için aksiyonu [-1,1] aralığına kırpıyoruz
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        value = self.critic(state)
        return action, log_prob, value


# ═══════════════════════════════════════════════════════════════════════════════
#  On-Policy Geçici Yörünge Belleği (Trajectory Buffer)
# ═══════════════════════════════════════════════════════════════════════════════

class PPOBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states      = []
        self.actions     = []
        self.rewards     = []
        self.log_probs   = []
        self.values      = []
        self.dones       = []

    def push(self, s, a, r, log_p, v, d):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.log_probs.append(log_p)
        self.values.append(v)
        self.dones.append(d)

    def get_tensors(self):
        s = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        a = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        r = torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device).unsqueeze(1)
        log_p = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device).unsqueeze(1)
        v = torch.tensor(np.array(self.values), dtype=torch.float32, device=device).unsqueeze(1)
        d = torch.tensor(np.array(self.dones), dtype=torch.float32, device=device).unsqueeze(1)
        return s, a, r, log_p, v, d

    def __len__(self):
        return len(self.states)


# ═══════════════════════════════════════════════════════════════════════════════
#  PPO Ajan Yapısı (Agent Core)
# ═══════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    def __init__(self, state_dim, action_dim=ACTION_DIM, lr=LR):
        self.ac = ActorCritic(state_dim, action_dim).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)
        self.buffer = PPOBuffer()

    def select_action(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.ac(s)
        return action.squeeze(0).cpu().numpy(), log_prob.cpu().item(), value.cpu().item()

    def update(self, next_state_rollout=None):
        s, a, r, old_log_p, v, d = self.buffer.get_tensors()
        
        advantages = torch.zeros_like(r, device=device)
        returns    = torch.zeros_like(r, device=device)
        gae = 0.0
        
        with torch.no_grad():
            if next_state_rollout is not None:
                ns_tensor = torch.tensor(next_state_rollout, dtype=torch.float32, device=device).unsqueeze(0)
                _, _, next_value = self.ac(ns_tensor)
                next_value = next_value.squeeze(0)
            else:
                _, _, next_value = self.ac(s[-1].unsqueeze(0))
                next_value = next_value.squeeze(0)
            
        for t in reversed(range(len(r))):
            next_non_terminal = 1.0 - d[t]
            next_val = v[t+1] if t < len(r) - 1 else next_value
            delta = r[t] + GAMMA * next_val * next_non_terminal - v[t]
            gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
            advantages[t] = gae
            returns[t]    = advantages[t] + v[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(s)
        indices = np.arange(dataset_size)
        
        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = indices[start:end]
                
                b_s = s[batch_idx]
                b_a = a[batch_idx]
                b_old_log_p = old_log_p[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]

                h_act = self.ac.actor(b_s)
                mu = torch.tanh(self.ac.mu(h_act))
                log_std = torch.clamp(self.ac.log_std, -20, 2)
                std = torch.exp(log_std.expand_as(mu))
                dist = torch.distributions.Normal(mu, std)

                new_log_p = dist.log_prob(b_a).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                val = self.ac.critic(b_s)

                ratio = torch.exp(new_log_p - b_old_log_p)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(val, b_ret)
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()

        self.buffer.clear() 
        return actor_loss.item(), critic_loss.item()

    def save(self, path, meta: dict):
        torch.save({"ac": self.ac.state_dict(), "opt": self.opt.state_dict(), **meta}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.ac.load_state_dict(ckpt["ac"])
        self.opt.load_state_dict(ckpt["opt"])
        return {k: ckpt[k] for k in ["step", "episode", "ep_count", "best_score", "best_env_rew", "best_lap"] if k in ckpt}

    def load_weights_only(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.ac.load_state_dict(ckpt["ac"])
        print(f"[TRANSFER] PPO Geometri Ağı Yüklendi: {path}")


def write_csv(columns, values, filepath):
    pd.DataFrame([values], columns=columns).to_csv(filepath, mode="a", index=False, header=not os.path.isfile(filepath))


# ═══════════════════════════════════════════════════════════════════════════════
#  Eğitim Kontrol Döngüsü (Training Loop)
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(track_str, episodes, ckpt_path, best_path, csv_path, transfer_path=None, lr=LR):
    print(f"\n{'='*70}")
    print(f"PPO DENEYSEL MATRİS: {track_str.upper()} | Hedef: {episodes} episode")
    print(f"{'='*70}\n")

    env = SimstarEnv(
        track=get_track(track_str), port=PORT, synronized_mode=True,
        speed_up=SPEED_UP, hz=HZ, lower_speed_limit=5, add_agent=False,
    )

    obs       = env.reset()
    state_dim = build_state(obs).shape[0]

    agent = PPOAgent(state_dim=state_dim, action_dim=ACTION_DIM, lr=lr)

    step          = 0
    start_ep      = 0
    ep_count      = 0
    best_score    = -1e18
    best_env_rew  = -1e18
    best_lap      = -1e18
    datalog       = defaultdict(list)

    if os.path.exists(ckpt_path):
        print(f"[LOAD] Checkpoint yüklendi -> {ckpt_path}")
        meta = agent.load(ckpt_path)
        step         = meta.get("step", 0)
        start_ep     = meta.get("episode", -1) + 1
        ep_count     = meta.get("ep_count", 0)
        best_score   = meta.get("best_score", -1e18)
        best_env_rew = meta.get("best_env_rew", -1e18)
        best_lap     = meta.get("best_lap", -1e18)
        
        if os.path.exists(csv_path):
            try:
                df_old = pd.read_csv(csv_path)
                datalog["env_r"] = df_old["env_reward"].tolist()
                datalog["lap"] = df_old["lap_progress"].tolist()
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
            act_losses = []
            prev_lap = env.get_lap_progress()
            lap_accum = 0.0
            end_reason = "unknown" # MADDE 3 ENTEGRASYONU: Başlangıç elenme log etiket tanımı

            for _ in range(MAX_STEPS):
                
                action, log_prob, val = agent.select_action(state)
                sim_action = make_action(action)

                next_obs, env_r, done, info = env.step(sim_action)
                next_state = build_state(next_obs)

                cur_lap = env.get_lap_progress()
                delta   = cur_lap - prev_lap
                if   delta < -0.5: delta += 1.0
                elif delta >  0.5: delta -= 1.0
                lap_accum += delta
                prev_lap   = cur_lap

                train_r = compute_reward(env_r, next_obs, action, delta, done, lap_accum)

                prev_lap_count = int((lap_accum - delta) // 1.0)
                cur_lap_count  = int(lap_accum // 1.0)
                if cur_lap_count > prev_lap_count:
                    train_r += 100.0

                agent.buffer.push(state, action, train_r, log_prob, val, float(done))

                ep_env_r   += float(env_r)
                ep_train_r += float(train_r)
                state       = next_state
                step       += 1

                # MADDE 3 DÜZELTMESİ: SimStar elenme sebebi (info) dinamik olarak yakalanıyor
                if isinstance(info, dict) and "end_reason" in info:
                    end_reason = info.get("end_reason", "done")
                elif done:
                    end_reason = "env_done"

                if len(agent.buffer) >= TRAIN_EVERY_STEPS:
                    a_loss, c_loss = agent.update(next_state_rollout=next_state)
                    act_losses.append(a_loss)

                if done:
                    break

            # MADDE 1 DÜZELTMESİ: Episode kapandığında içeride kalan artık verileri batch'i geçiyorsa anında işliyoruz!
            if len(agent.buffer) >= BATCH_SIZE:
                a_loss, c_loss = agent.update(next_state_rollout=state)
                act_losses.append(a_loss)

            ep_count += 1

            def avg20(key, val):
                h = datalog[key][-19:]
                return float(np.mean(h + [val])) if h else float(val)

            avg_env = avg20("env_r",  ep_env_r)
            avg_lap = avg20("lap",    lap_accum)
            datalog["env_r"].append(ep_env_r)
            datalog["lap"].append(lap_accum)

            a_loss = float(np.mean(act_losses)) if act_losses else 0.0

            if lap_accum > best_lap + 0.005:
                best_score   = ep_train_r
                best_env_rew = ep_env_r
                best_lap     = lap_accum
                agent.save(best_path, dict(step=step, episode=episode, ep_count=ep_count, best_score=best_score, best_env_rew=best_env_rew, best_lap=best_lap))
                print(f"\n★ PPO Rekor Gelişim | Lap={best_lap*100:.1f}% | Ödül={best_env_rew:.2f} | Sebep={end_reason}")

            agent.save(ckpt_path, dict(step=step, episode=episode, ep_count=ep_count, best_score=best_score, best_env_rew=best_env_rew, best_lap=best_lap))

            elapsed_min = (time.time() - t0) / 60.0
            print(f"Ep {ep_count}/{episodes} | EnvR {ep_env_r:7.2f} | Avg20Env {avg_env:7.2f} | Lap {lap_accum*100:5.1f}% | Avg20Lap {avg_lap*100:5.1f}% | Best {best_lap*100:5.1f}% | ActorL {a_loss:.4f} | Steps {step} | {end_reason}")

            # MADDE 3 DÜZELTMESİ: Veri setine elenme sebebi (termination_reason) kolonu eksiksiz işleniyor
            write_csv(["episode","track","env_reward","avg20_env","lap_progress","avg20_lap","actor_loss","steps","best_lap","termination_reason"],
                      [ep_count, track_str, ep_env_r, avg_env, lap_accum, avg_lap, a_loss, step, best_lap, end_reason], csv_path)

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[INTERRUPTED] Süreç sonlandırıldı. Mevcut ağırlıklar diske yazılıyor...")
    finally:
        agent.save(ckpt_path, dict(step=step, episode=episode, ep_count=ep_count, best_score=best_score, best_env_rew=best_env_rew, best_lap=best_lap))
        try: env.end()
        except Exception: pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("track", type=str, choices=["austria", "dutch", "hungary"])
    args = parser.parse_args()

    if args.track == "austria":
        run_training(track_str="austria", episodes=AUSTRIA_EPISODES, ckpt_path=AUSTRIA_CKPT, best_path=AUSTRIA_BEST, csv_path=AUSTRIA_CSV, lr=LR)
    elif args.track == "dutch":
        run_training(track_str="dutch", episodes=DUTCH_EPISODES, ckpt_path=DUTCH_CKPT, best_path=DUTCH_BEST, csv_path=DUTCH_CSV, transfer_path=AUSTRIA_BEST, lr=FINETUNE_LR)
    elif args.track == "hungary":
        run_training(track_str="hungary", episodes=HUNGARY_EPISODES, ckpt_path=HUNGARY_CKPT, best_path=HUNGARY_BEST, csv_path=HUNGARY_CSV, transfer_path=DUTCH_BEST, lr=FINETUNE_LR)