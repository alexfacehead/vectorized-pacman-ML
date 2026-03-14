"""Batched DQN training for Pacman using the vectorized engine.

Runs N environments simultaneously. Each game step produces N transitions,
giving an N× throughput boost for replay buffer filling. The neural network
forward pass is batched across all N environments for action selection.

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
import torch.optim as optim

from engine.batched_game import BatchedGame
from engine.constants import MAX_STEPS, NUM_STATE_CHANNELS
from models.pacman_model import PacmanDQN
from utils.replay_buffer import ReplayBuffer


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
    """DQN agent adapted for batched (N-env) training.

    Uses Double-DQN with a target network and epsilon-greedy exploration.
    Action selection is batched: one forward pass for all N environments.
    """

    def __init__(
        self,
        in_channels: int = 7,
        num_actions: int = 4,
        lr: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = _auto_device()
        else:
            self.device = torch.device(device)

        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.model = PacmanDQN(in_channels, num_actions).to(self.device)
        self.target_model = PacmanDQN(in_channels, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def select_actions_batched(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select actions for all N environments in one forward pass.

        Args:
            states: (N, C, H, W) float32 tensor on device.
            action_masks: (N, 4) bool tensor — True = valid.

        Returns:
            actions: (N,) int64 tensor.
        """
        N = states.shape[0]
        actions = torch.zeros(N, dtype=torch.int64, device=self.device)

        # Determine which envs explore vs exploit
        explore_mask = torch.rand(N, device=self.device) < self.epsilon

        # Exploration: random valid action
        if explore_mask.any():
            # For each exploring env, pick a random valid action
            for i in torch.where(explore_mask)[0]:
                valid = torch.where(action_masks[i])[0]
                if len(valid) == 0:
                    valid = torch.arange(self.num_actions, device=self.device)
                actions[i] = valid[torch.randint(len(valid), (1,), device=self.device)]

        # Exploitation: greedy from Q-values
        exploit_mask = ~explore_mask
        if exploit_mask.any():
            with torch.no_grad():
                exploit_states = states[exploit_mask]
                q_values = self.model(exploit_states)
                # Mask invalid actions
                exploit_action_masks = action_masks[exploit_mask]
                q_values[~exploit_action_masks] = float('-inf')
                exploit_actions = q_values.argmax(dim=1)
                actions[exploit_mask] = exploit_actions

        return actions

    def store_transitions_batched(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add N transitions to the replay buffer at once."""
        self.buffer.add_batch(states, actions, rewards, next_states, dones)

    def train_step(self) -> Optional[float]:
        """Sample a batch and perform one gradient step (Double DQN)."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Current Q-values
        q_values = self.model(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_q_online = self.model(next_states_t)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_model(next_states_t)
            next_q = next_q_target.gather(1, best_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_selected, target)

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
    batch_size: int = 64,
    target_update: int = 10_000,
    train_every: int = 4,
    no_reverse: bool = True,
    stage: int = 1,
) -> None:
    """Train Pacman DQN agent using N parallel environments.

    Args:
        no_reverse: If True, mask the reverse of Pacman's current direction
            using proximity-based logic: reverse is blocked unless a non-house
            ghost is within Manhattan distance 4 (so the agent can flee).
        stage: Curriculum stage (1=no ghosts, 2=Blinky slow, 3=Blinky fast,
            4=Blinky+Pinky).
    """

    maze_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.pardir, "levels", "level_1.txt")

    # Resolve device — model goes on GPU, game stays on CPU for throughput
    if device == "auto":
        model_dev = _auto_device()
    else:
        model_dev = torch.device(device)

    # Apply defaults for epsilon params (None means use defaults)
    _eps_start = eps_start if eps_start is not None else 0.5
    _eps_end = eps_end if eps_end is not None else 0.01
    _eps_decay = eps_decay if eps_decay is not None else 0.99997

    # Game engine always on CPU (14x faster than MPS for small integer ops)
    game = BatchedGame(n_envs=n_envs, maze_file=maze_file, device="cpu")
    game.configure_stage(stage)
    total_pellets = game.maze.total_pellets

    # Create agent (model on GPU)
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
        target_update_freq=target_update,
        device=str(model_dev),
    )

    # Resume
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "pacman_agent.pt")
    start_episode = 0

    if resume and os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        # Only override epsilon if user explicitly passed --eps-start
        # Otherwise keep the checkpoint's epsilon value
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
    print("  Vectorized Pac-Man — Batched DQN Training")
    print("=" * 70)
    stage_desc = {1: "No ghosts", 2: "Blinky slow", 3: "Blinky fast", 4: "Blinky+Pinky slow",
                  5: "3 ghosts slow", 6: "Blinky slow + Pinky fast", 7: "3 ghosts mixed"}
    print(f"  Stage          : {stage} ({stage_desc.get(stage, '?')})")
    print(f"  Parallel envs  : {n_envs}")
    print(f"  Episodes       : {num_episodes} (starting from {start_episode})")
    print(f"  Game device    : cpu")
    print(f"  Model device   : {model_dev}")
    print(f"  LR             : {lr}")
    print(f"  Gamma          : {gamma}")
    print(f"  Epsilon        : {_eps_start} -> {_eps_end} (decay {_eps_decay})")
    print(f"  Batch size     : {batch_size}")
    print(f"  Buffer size    : {buffer_size:,}")
    print(f"  Target update  : every {target_update} steps")
    print(f"  Train every    : {train_every} game steps")
    print(f"  Save schedule  : episodes 1-10 individually, then every 10")
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
    all_results = []
    recent_losses = []
    pac_wins = 0
    ghost_wins = 0
    timeouts = 0
    best_score = 0
    total_game_steps = 0

    t0 = time.time()

    # Episode tracking per env
    ep_rewards = np.zeros(n_envs, dtype=np.float32)
    ep_steps = np.zeros(n_envs, dtype=np.int32)
    episodes_completed = 0

    # Get initial state
    state = game.get_state()  # (N, 7, 31, 28) on device

    for episode in range(start_episode, start_episode + num_episodes):
        game.reset()
        ep_reward_total = 0.0
        ep_step_count = 0
        ep_t0 = time.time()
        ep_rewards[:] = 0
        ep_steps[:] = 0

        state = game.get_state()

        while True:
            # Check if all envs are done
            dones_all = game.game_over | game.level_complete
            if dones_all.all().item():
                break
            if ep_step_count >= MAX_STEPS:
                break

            # Get action masks — proximity-based no-reverse handles per-env
            # ghost distance checks internally (reverse allowed when ghost nearby)
            action_masks = game.get_action_mask(no_reverse=no_reverse)  # (N, 4) bool, CPU

            # Transfer to model device for action selection, then back to CPU for stepping
            actions = agent.select_actions_batched(
                state.to(agent.device), action_masks.to(agent.device))
            actions_cpu = actions.cpu()

            # Step all environments (CPU)
            rewards, dones, infos = game.step(actions_cpu)

            # Get next state
            next_state = game.get_state()

            # Get clipped rewards
            pac_rewards = game.get_reward_pacman()  # (N,) clipped to [-1, 1]

            # Store transitions for all active (non-done-before-step) environments
            active_before = ~dones_all
            if active_before.any():
                active_idx = torch.where(active_before)[0]
                s = state[active_idx].numpy()
                a = actions_cpu[active_idx].numpy()
                r = pac_rewards[active_idx].numpy()
                ns = next_state[active_idx].numpy()
                d = dones[active_idx].float().numpy()
                agent.store_transitions_batched(s, a, r, ns, d)

            # Train
            if ep_step_count % train_every == 0:
                loss = agent.train_step()
                if loss is not None:
                    recent_losses.append(loss)

            # Accumulate rewards for logging
            ep_reward_total += pac_rewards.sum().item()
            ep_step_count += 1
            total_game_steps += 1
            state = next_state

            # Mid-episode progress (long episodes with no ghosts can take minutes)
            if ep_step_count % 500 == 0:
                done_mask = game.game_over | game.level_complete
                alive = (~done_mask).sum().item()
                wins = game.level_complete.sum().item()
                # Average pellets eaten across all envs
                remaining = (game.pellets.sum(dim=(1, 2)) + game.power_pellets.sum(dim=(1, 2))).float()
                avg_eaten = (total_pellets - remaining.mean()).item()
                ep_elapsed = time.time() - ep_t0
                sps = ep_step_count * n_envs / ep_elapsed if ep_elapsed > 0 else 0
                print(f"    ... step {ep_step_count}/{MAX_STEPS} | "
                      f"avg {avg_eaten:.0f}/{total_pellets} pellets | "
                      f"{wins} won, {alive} playing | "
                      f"{sps:.0f} env-steps/s", flush=True)

        # Episode results — aggregate across all N envs
        avg_score = game.score.float().mean().item()
        remaining_all = (game.pellets.sum(dim=(1, 2)) + game.power_pellets.sum(dim=(1, 2))).float()
        avg_pellets_eaten = int(total_pellets - remaining_all.mean().item())

        ep_pac_wins = game.level_complete.sum().item()
        # game_over can include envs that also completed the level; exclude those
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
            # Visual win rate bar for last 5 episodes
            wr_bar = " ".join(f"{100*r:2.0f}%" for r in recent_5_wr)
            print(f"  >>> Last 5 win%: [{wr_bar}] avg {avg_wr_5:.0f}% | "
                  f"Overall {overall_wr:.0f}% ({pac_wins}W/{ghost_wins}L/{timeouts}TO) | "
                  f"Avg Score {np.mean(all_scores):.0f} | Best {best_score:.0f} | "
                  f"Avg R {np.mean(all_rewards):.1f} | "
                  f"Buf {len(agent.buffer):,} | {sps:.0f} sps")

        # Save every 50 episodes
        should_save = eps_done % 50 == 0
        if should_save:
            # Versioned checkpoint (never overwritten)
            tag = f"s{stage}_ep{ep_num}"
            versioned_path = os.path.join(save_dir, f"pacman_{tag}.pt")
            agent.save(versioned_path)
            # Also save as latest (for --resume)
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
    parser = argparse.ArgumentParser(description="Batched DQN training for Pac-Man")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/mps/cuda)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer capacity")
    parser.add_argument("--eps-start", type=float, default=None, help="Starting epsilon (default: 0.5)")
    parser.add_argument("--eps-end", type=float, default=None, help="Final epsilon (default: 0.01)")
    parser.add_argument("--eps-decay", type=float, default=None, help="Epsilon decay per train step (default: 0.99997)")
    parser.add_argument("--no-reverse", action="store_true", default=True,
                        help="Anti-oscillation: proximity-based reverse masking (default: on)")
    parser.add_argument("--allow-reverse", action="store_true",
                        help="Disable no-reverse masking")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Curriculum stage (1=no ghosts, 2=Blinky slow, 3=Blinky fast, 4=B+P slow, 5=3 ghosts slow, 6=B slow+P fast, 7=3 ghosts mixed)")
    parser.add_argument("--n-envs-bench", type=int, nargs="+", help="Benchmark mode: test throughput for these env counts")
    args = parser.parse_args()
    no_reverse = not args.allow_reverse

    if args.n_envs_bench:
        # Quick throughput benchmark
        print("\n  Throughput Benchmark")
        print("  " + "-" * 50)
        maze_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 os.pardir, "levels", "level_1.txt")
        for n in args.n_envs_bench:
            game = BatchedGame(n_envs=n, maze_file=maze_file, device="cpu")
            actions = torch.randint(0, 4, (n,))
            # Warmup
            for _ in range(100):
                game.step(actions)
            # Timed run
            t0 = time.time()
            num_steps = 1000
            for _ in range(num_steps):
                rewards, dones, _ = game.step(actions)
                done_mask = dones
                if done_mask.any():
                    game.reset(done_mask)
            elapsed = time.time() - t0
            sps = num_steps * n / elapsed
            print(f"  n_envs={n:>4d}  |  {sps:>10,.0f} env-steps/sec  |  {elapsed:.2f}s for {num_steps} steps")
        print()
    else:
        # Build kwargs, only passing epsilon overrides if specified
        kwargs = dict(
            n_envs=args.n_envs,
            num_episodes=args.episodes,
            save_dir=args.save_dir,
            resume=args.resume,
            device=args.device,
            lr=args.lr,
            batch_size=args.batch_size,
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
