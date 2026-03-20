"""Batched DRQN training for Pacman using the vectorized engine.

Runs N environments simultaneously. Each game step produces N transitions,
giving an N× throughput boost. The neural network uses an LSTM for temporal
reasoning — the agent can learn multi-step strategies, ghost tracking, and
route planning instead of purely reactive play.

Usage:
    python -m training.train                        # 16 envs, 5000 episodes
    python -m training.train --n-envs 64            # 64 parallel envs
    python -m training.train --device mps           # Force MPS device
    python -m training.train --resume               # Resume from checkpoint
"""

import os
import time
import csv
import argparse
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from engine.batched_game import BatchedGame
from engine.constants import MAX_STEPS, NUM_STATE_CHANNELS
from models.pacman_model import PacmanDRQN
from utils.replay_buffer import SequenceReplayBuffer


def _auto_device() -> torch.device:
    """Select best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


class BatchedPacmanAgent:
    """DRQN agent for batched (N-env) training.

    Uses Double-DQN with a target network, epsilon-greedy exploration,
    and an LSTM for temporal reasoning. Action selection always runs a
    forward pass on all N environments to keep LSTM hidden states current.
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_actions: int = 4,
        lr: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 200_000,
        batch_size: int = 32,
        seq_len: int = 16,
        target_update_freq: int = 5000,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = _auto_device()
        else:
            self.device = torch.device(device)

        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.model = PacmanDRQN(in_channels, num_actions).to(self.device)
        self.target_model = PacmanDRQN(in_channels, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.buffer = SequenceReplayBuffer(buffer_size, seq_len)
        self.steps = 0

    def init_hidden(self, batch_size: int):
        """Initialize LSTM hidden state for N environments."""
        return self.model.init_hidden(batch_size, self.device)

    def select_actions_batched(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
        hidden,
    ):
        """Select actions for all N environments AND update LSTM hidden state.

        Always runs the full forward pass on all envs to keep hidden states
        current, then applies epsilon-greedy action selection.

        Args:
            states: (N, C, H, W) float32 tensor on device.
            action_masks: (N, 4) bool tensor — True = valid.
            hidden: (h, c) tuple from previous step.

        Returns:
            actions: (N,) int64 tensor.
            hidden: updated (h, c) tuple.
        """
        N = states.shape[0]

        with torch.no_grad():
            q_values, hidden = self.model(states, hidden)  # (N, 4), hidden

        # Greedy actions with mask
        q_masked = q_values.clone()
        q_masked[~action_masks] = float('-inf')
        actions = q_masked.argmax(dim=1)

        # Exploration: random valid action for some envs
        explore_mask = torch.rand(N, device=self.device) < self.epsilon
        if explore_mask.any():
            for i in torch.where(explore_mask)[0]:
                valid = torch.where(action_masks[i])[0]
                if len(valid) == 0:
                    valid = torch.arange(self.num_actions, device=self.device)
                actions[i] = valid[torch.randint(len(valid), (1,), device=self.device)]

        return actions, hidden

    def train_step(self) -> Optional[float]:
        """Sample a batch of sequences and perform one gradient step (Double DQN).

        Forwards the full (L+1)-length state sequence through both online and
        target models:
          - Online at t=0..L-1 → Q(s_t) for current actions
          - Online at t=1..L   → Q(s_{t+1}) for Double-DQN action selection
          - Target at t=1..L   → Q_target(s_{t+1}) for value estimation
        """
        if len(self.buffer) < self.batch_size * self.seq_len:
            return None

        states_full, actions, rewards, dones, masks = self.buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states_full).to(self.device)   # (B, L+1, C, H, W)
        actions_t = torch.from_numpy(actions).to(self.device)      # (B, L)
        rewards_t = torch.from_numpy(rewards).to(self.device)      # (B, L)
        dones_t = torch.from_numpy(dones).to(self.device)          # (B, L)
        masks_t = torch.from_numpy(masks).to(self.device)          # (B, L)

        # Forward online model through full sequence
        q_all, _ = self.model(states_t)  # (B, L+1, num_actions)
        q_current = q_all[:, :-1, :]                 # (B, L, num_actions)
        q_next_online = q_all[:, 1:, :].detach()     # (B, L, num_actions)

        # Forward target model (no grad)
        with torch.no_grad():
            q_target_all, _ = self.target_model(states_t)
            q_next_target = q_target_all[:, 1:, :]   # (B, L, num_actions)

        # Double DQN: online selects, target evaluates
        q_selected = q_current.gather(2, actions_t.unsqueeze(2)).squeeze(2)  # (B, L)
        best_next = q_next_online.argmax(dim=2)                              # (B, L)
        next_q = q_next_target.gather(2, best_next.unsqueeze(2)).squeeze(2)  # (B, L)

        # TD target
        target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # Masked Huber loss
        elementwise_loss = F.smooth_l1_loss(q_selected, target, reduction='none')
        loss = (elementwise_loss * masks_t).sum() / masks_t.sum().clamp(min=1)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.detach().item()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "seq_len": self.seq_len,
                "arch": "drqn",
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]


def _push_episodes_to_buffer(
    buffer: SequenceReplayBuffer,
    step_states: list,
    step_actions: list,
    step_rewards: list,
    step_dones: list,
    step_active: list,
    n_envs: int,
) -> None:
    """Split per-step data by environment and push to sequence buffer.

    Each env ran for a contiguous prefix of steps (active from step 0 until
    done), producing an independent trajectory.
    """
    T = len(step_actions)
    if T == 0:
        return

    # Number of active steps per env
    active_all = np.stack(step_active)  # (T, N)
    T_per_env = active_all.sum(axis=0).astype(np.int32)  # (N,)

    for i in range(n_envs):
        T_i = int(T_per_env[i])
        if T_i < 1:
            continue

        # Extract per-env trajectory (list comprehension avoids stacking full arrays)
        ep_states = np.stack([step_states[t][i] for t in range(T_i + 1)])
        ep_actions = np.array([step_actions[t][i] for t in range(T_i)], dtype=np.int64)
        ep_rewards = np.array([step_rewards[t][i] for t in range(T_i)], dtype=np.float32)
        ep_dones = np.array([step_dones[t][i] for t in range(T_i)], dtype=np.float32)

        buffer.add_episode(ep_states, ep_actions, ep_rewards, ep_dones)


def train(
    n_envs: int = 16,
    num_episodes: int = 5000,
    save_dir: str = "checkpoints",
    resume: bool = False,
    device: str = "auto",
    lr: float = 0.0001,
    gamma: float = 0.99,
    eps_start: float = None,
    eps_end: float = None,
    eps_decay: float = None,
    buffer_size: int = 200_000,
    batch_size: int = 32,
    seq_len: int = 16,
    target_update: int = 5_000,
    train_every: int = 4,
    no_reverse: bool = True,
    stage: int = 1,
) -> None:
    """Train Pacman DRQN agent using N parallel environments."""

    maze_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.pardir, "levels", "level_1.txt")

    # Resolve device
    if device == "auto":
        model_dev = _auto_device()
    else:
        model_dev = torch.device(device)

    # Apply defaults for epsilon params
    _eps_start = eps_start if eps_start is not None else 0.5
    _eps_end = eps_end if eps_end is not None else 0.05
    _eps_decay = eps_decay if eps_decay is not None else 0.99997

    # Game engine always on CPU
    game = BatchedGame(n_envs=n_envs, maze_file=maze_file, device="cpu")
    game.configure_stage(stage)
    total_pellets = game.maze.total_pellets

    # Create agent
    agent = BatchedPacmanAgent(
        in_channels=NUM_STATE_CHANNELS,
        num_actions=4,
        lr=lr,
        gamma=gamma,
        epsilon_start=_eps_start,
        epsilon_end=_eps_end,
        epsilon_decay=_eps_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        seq_len=seq_len,
        target_update_freq=target_update,
        device=str(model_dev),
    )

    # Count params
    n_params = sum(p.numel() for p in agent.model.parameters())

    # Resume
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "pacman_agent.pt")
    start_episode = 0

    if resume and os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        if eps_start is not None:
            agent.epsilon = eps_start
        if eps_end is not None:
            agent.epsilon_end = eps_end
        if eps_decay is not None:
            agent.epsilon_decay = eps_decay
        meta_path = os.path.join(save_dir, "meta.npy")
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True).item()
            start_episode = meta.get("episode", 0)

    # Banner
    print()
    print("=" * 70)
    print("  Vectorized Pac-Man — Dueling DRQN Training (LSTM + Per-Ghost)")
    print("=" * 70)
    stage_desc = {1: "No ghosts", 2: "Blinky slow", 3: "Blinky fast", 4: "Blinky+Pinky slow",
                  5: "3 ghosts slow", 6: "4 ghosts slow", 7: "Blinky slow + Pinky fast",
                  8: "3 ghosts mixed"}
    print(f"  Stage          : {stage} ({stage_desc.get(stage, '?')})")
    print(f"  Parallel envs  : {n_envs}")
    print(f"  Episodes       : {num_episodes} (starting from {start_episode})")
    print(f"  Game device    : cpu")
    print(f"  Model device   : {model_dev}")
    print(f"  LR             : {lr}")
    print(f"  Gamma          : {gamma}")
    print(f"  Epsilon        : {agent.epsilon:.4f} -> {_eps_end} (decay {_eps_decay})")
    print(f"  Architecture   : Dueling DRQN, 9ch, Conv→512→LSTM(512)→Dueling")
    print(f"  Parameters     : {n_params:,}")
    print(f"  LSTM seq_len   : {seq_len}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Buffer size    : {buffer_size:,}")
    print(f"  Target update  : every {target_update} steps")
    print(f"  Train every    : {train_every} game steps")
    print(f"  Save schedule  : episodes 1-10 individually, then every 75")
    print(f"  No-reverse     : {no_reverse} (anti-oscillation)")
    print(f"  Save dir       : {os.path.abspath(save_dir)}")
    if resume and start_episode > 0:
        print(f"  Resumed from   : episode {start_episode}")
        print(f"  Epsilon        : {agent.epsilon:.4f}")
    print("=" * 70)
    print()
    print("  Ep   | Score |  Pellets | Result   | Reward | Eps    | Loss    | Steps | Time")
    print("-" * 95, flush=True)

    # CSV log file
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_s{stage}_{log_timestamp}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "timestamp", "episode", "stage", "n_envs", "avg_score", "pellets_eaten",
        "total_pellets", "pac_wins", "ghost_wins", "timeouts", "win_pct",
        "avg_reward", "epsilon", "avg_loss", "steps", "elapsed_sec",
        "env_steps_per_sec", "buffer_size",
    ])
    print(f"  Log file       : {os.path.abspath(log_path)}")

    # Tracking
    all_scores = []
    all_rewards = []
    ep_win_rates = []
    recent_losses = []
    pac_wins = 0
    ghost_wins = 0
    timeouts = 0
    best_score = 0
    total_game_steps = 0

    t0 = time.time()

    for episode in range(start_episode, start_episode + num_episodes):
        game.reset()
        state = game.get_state()  # (N, C, H, W) tensor on CPU
        hidden = agent.init_hidden(n_envs)

        # Per-step data collection for episode buffer
        step_states = [state.numpy().copy()]  # initial state for all envs
        step_actions = []
        step_rewards = []
        step_dones = []
        step_active = []

        ep_reward_total = 0.0
        ep_step_count = 0
        ep_t0 = time.time()

        while True:
            # Check if all envs are done
            dones_all = game.game_over | game.level_complete
            if dones_all.all().item():
                break
            if ep_step_count >= MAX_STEPS:
                break

            active = ~dones_all  # (N,) bool — envs still playing

            # Get action masks
            action_masks = game.get_action_mask(no_reverse=no_reverse)

            # Select actions (always forwards all envs to update LSTM hidden)
            actions, hidden = agent.select_actions_batched(
                state.to(agent.device), action_masks.to(agent.device), hidden)
            actions_cpu = actions.cpu()

            # Step all environments
            rewards, dones, infos = game.step(actions_cpu)

            # Get next state
            next_state = game.get_state()

            # Get rewards (unclipped — see README)
            pac_rewards = game.get_reward_pacman()

            # Record step data (copy because get_state uses double-buffered tensors)
            step_states.append(next_state.numpy().copy())
            step_actions.append(actions_cpu.numpy().copy())
            step_rewards.append(pac_rewards.numpy().copy())
            step_dones.append(dones.float().numpy().copy())
            step_active.append(active.numpy().copy())

            # Train on sequences from previous episodes
            if ep_step_count % train_every == 0:
                loss = agent.train_step()
                if loss is not None:
                    recent_losses.append(loss)

            # Accumulate rewards for logging
            ep_reward_total += pac_rewards.sum().item()
            ep_step_count += 1
            total_game_steps += 1
            state = next_state

            # Mid-episode progress
            if ep_step_count % 500 == 0:
                done_mask = game.game_over | game.level_complete
                alive = (~done_mask).sum().item()
                wins = game.level_complete.sum().item()
                remaining = (game.pellets.sum(dim=(1, 2)) + game.power_pellets.sum(dim=(1, 2))).float()
                avg_eaten = (total_pellets - remaining.mean()).item()
                ep_elapsed = time.time() - ep_t0
                sps = ep_step_count * n_envs / ep_elapsed if ep_elapsed > 0 else 0
                print(f"    ... step {ep_step_count}/{MAX_STEPS} | "
                      f"avg {avg_eaten:.0f}/{total_pellets} pellets | "
                      f"{wins} won, {alive} playing | "
                      f"{sps:.0f} env-steps/s", flush=True)

        # Push per-env trajectories to sequence replay buffer
        _push_episodes_to_buffer(
            agent.buffer, step_states, step_actions, step_rewards,
            step_dones, step_active, n_envs)

        # Free episode data
        del step_states, step_actions, step_rewards, step_dones, step_active

        # Episode results
        avg_score = game.score.float().mean().item()
        remaining_all = (game.pellets.sum(dim=(1, 2)) + game.power_pellets.sum(dim=(1, 2))).float()
        avg_pellets_eaten = int(total_pellets - remaining_all.mean().item())

        ep_pac_wins = game.level_complete.sum().item()
        ep_ghost_wins = (game.game_over & ~game.level_complete).sum().item()
        ep_timeouts = n_envs - ep_pac_wins - ep_ghost_wins

        all_scores.append(avg_score)
        all_rewards.append(ep_reward_total / n_envs)

        if avg_score > best_score:
            best_score = avg_score

        pac_wins += ep_pac_wins
        ghost_wins += ep_ghost_wins
        timeouts += ep_timeouts
        ep_win_rates.append(ep_pac_wins / n_envs)

        # Print
        ep_num = episode + 1
        eps_done = ep_num - start_episode
        elapsed = time.time() - t0
        eta = (elapsed / eps_done) * (num_episodes - eps_done) if eps_done > 0 else 0
        avg_loss = np.mean(recent_losses[-100:]) if recent_losses else 0.0
        avg_reward = ep_reward_total / n_envs
        win_pct = 100 * ep_pac_wins / n_envs

        print(
            f"  {ep_num:>4d} | "
            f"Score {avg_score:>5.0f} | "
            f"{avg_pellets_eaten:>3d}/{total_pellets} pellets | "
            f"W {ep_pac_wins} / L {ep_ghost_wins} / TO {ep_timeouts} ({n_envs} games) "
            f"[{win_pct:4.0f}% win] | "
            f"R {avg_reward:>+6.1f} | "
            f"Eps {agent.epsilon:.4f} | "
            f"Loss {avg_loss:.4f} | "
            f"{ep_step_count} steps | "
            f"{_fmt_time(elapsed)} (ETA {_fmt_time(eta)})"
        )

        # Log to CSV
        sps_now = total_game_steps * n_envs / elapsed if elapsed > 0 else 0
        log_writer.writerow([
            datetime.now().isoformat(), ep_num, stage, n_envs, f"{avg_score:.0f}",
            avg_pellets_eaten, total_pellets, ep_pac_wins, ep_ghost_wins,
            ep_timeouts, f"{win_pct:.1f}", f"{avg_reward:.1f}",
            f"{agent.epsilon:.6f}", f"{avg_loss:.4f}", ep_step_count,
            f"{elapsed:.1f}", f"{sps_now:.0f}", len(agent.buffer),
        ])
        log_file.flush()

        # Running summary every 5 episodes
        if eps_done % 5 == 0 or eps_done == 1:
            total_games = eps_done * n_envs
            sps = total_game_steps * n_envs / elapsed if elapsed > 0 else 0
            recent_5_wr = ep_win_rates[-5:]
            avg_wr_5 = 100 * np.mean(recent_5_wr)
            overall_wr = 100 * pac_wins / total_games
            wr_bar = " ".join(f"{100*r:2.0f}%" for r in recent_5_wr)
            print(f"  >>> Last 5 win%: [{wr_bar}] avg {avg_wr_5:.0f}% | "
                  f"Overall {overall_wr:.0f}% ({pac_wins}W/{ghost_wins}L/{timeouts}TO) | "
                  f"Avg Score {np.mean(all_scores):.0f} | Best {best_score:.0f} | "
                  f"Avg R {np.mean(all_rewards):.1f} | "
                  f"Buf {len(agent.buffer):,} ({agent.buffer.num_episodes} eps) | {sps:.0f} sps")

        # Save: every episode for first 10, then every 25
        should_save = eps_done <= 10 or eps_done % 25 == 0
        if should_save:
            tag = f"s{stage}_ep{ep_num}"
            versioned_path = os.path.join(save_dir, f"pacman_{tag}.pt")
            agent.save(versioned_path)
            agent.save(ckpt_path)
            np.save(os.path.join(save_dir, "meta.npy"), {"episode": ep_num, "stage": stage, "epsilon": agent.epsilon})
            print(f"\033[92m  >>> Saved: {os.path.basename(versioned_path)} (+ latest)\033[0m")

    # Final save
    final_ep = start_episode + num_episodes
    final_tag = f"s{stage}_ep{final_ep}"
    final_path = os.path.join(save_dir, f"pacman_{final_tag}.pt")
    agent.save(final_path)
    agent.save(ckpt_path)
    np.save(os.path.join(save_dir, "meta.npy"), {"episode": final_ep, "stage": stage, "epsilon": agent.epsilon})

    # Close log file
    log_file.close()
    print(f"\n  Training log saved to: {os.path.abspath(log_path)}")

    # Summary
    elapsed = time.time() - t0
    total = len(all_scores)
    print()
    print("=" * 70)
    print("  Training Complete!")
    print("=" * 70)
    if total > 0:
        sps = total_game_steps * n_envs / elapsed if elapsed > 0 else 0
        total_games = total * n_envs
        print(f"  Total episodes : {total} ({total_games:,} games)")
        print(f"  Total time     : {_fmt_time(elapsed)}")
        print(f"  Throughput     : {sps:.0f} env-steps/sec")
        print(f"  Avg score      : {np.mean(all_scores):.0f}")
        print(f"  Best score     : {best_score}")
        last_n = min(100, total)
        print(f"  Last {last_n} avg   : {np.mean(all_scores[-last_n:]):.0f}")
        print(f"  Pacman wins    : {pac_wins} ({100 * pac_wins / total_games:.1f}%)")
        print(f"  Ghost wins     : {ghost_wins} ({100 * ghost_wins / total_games:.1f}%)")
        print(f"  Timeouts       : {timeouts} ({100 * timeouts / total_games:.1f}%)")
    print("=" * 70)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched DRQN training for Pac-Man")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/mps/cuda)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (sequences)")
    parser.add_argument("--seq-len", type=int, default=16, help="LSTM sequence length for training")
    parser.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer capacity (transitions)")
    parser.add_argument("--eps-start", type=float, default=None, help="Starting epsilon (default: 0.5)")
    parser.add_argument("--eps-end", type=float, default=None, help="Final epsilon (default: 0.05)")
    parser.add_argument("--eps-decay", type=float, default=None, help="Epsilon decay per train step (default: 0.99997)")
    parser.add_argument("--no-reverse", action="store_true", default=True,
                        help="Anti-oscillation: proximity-based reverse masking (default: on)")
    parser.add_argument("--allow-reverse", action="store_true",
                        help="Disable no-reverse masking")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Curriculum stage (1=no ghosts, 2=Blinky slow, 3=Blinky fast, 4=B+P slow, 5=3 ghosts slow, 6=B slow+P fast, 7=3 ghosts mixed)")
    args = parser.parse_args()
    no_reverse = not args.allow_reverse

    # Build kwargs
    kwargs = dict(
        n_envs=args.n_envs,
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        resume=args.resume,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        buffer_size=args.buffer_size,
        no_reverse=no_reverse,
        stage=args.stage,
    )
    if args.eps_start is not None:
        kwargs["eps_start"] = args.eps_start
    if args.eps_end is not None:
        kwargs["eps_end"] = args.eps_end
    if args.eps_decay is not None:
        kwargs["eps_decay"] = args.eps_decay
    train(**kwargs)
