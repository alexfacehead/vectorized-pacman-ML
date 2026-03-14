"""Batched pellet collection, scoring, and level completion."""

import torch

from .constants import PELLET_SCORE, POWER_PELLET_SCORE, MAX_STEPS


def collect_pellets(
    pacman_pos: torch.Tensor,
    pellets: torch.Tensor,
    power_pellets: torch.Tensor,
    total_pellets_start: int,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check and handle pellet collection for all N environments.

    Modifies pellets and power_pellets tensors in-place.

    Args:
        pacman_pos: (N, 2) int — Pacman positions (x, y).
        pellets: (N, ROWS, COLS) bool — uneaten regular pellets (modified in-place).
        power_pellets: (N, ROWS, COLS) bool — uneaten power pellets (modified in-place).
        total_pellets_start: int — total pellets at level start.
        active: (N,) bool — which envs are active.

    Returns:
        Tuple of:
            reward: (N,) float — pellet-eating reward this step.
            ate_pellet: (N,) bool — whether a regular pellet was eaten.
            ate_power: (N,) bool — whether a power pellet was eaten.
    """
    N = pacman_pos.shape[0]
    device = pacman_pos.device
    reward = torch.zeros(N, dtype=torch.float32, device=device)

    px = pacman_pos[:, 0].long()  # (N,)
    py = pacman_pos[:, 1].long()  # (N,)
    env_idx = torch.arange(N, device=device)

    # Regular pellets
    ate_pellet = pellets[env_idx, py, px] & active  # (N,)
    pellets[env_idx[ate_pellet], py[ate_pellet], px[ate_pellet]] = False

    # Power pellets
    ate_power = power_pellets[env_idx, py, px] & active  # (N,)
    power_pellets[env_idx[ate_power], py[ate_power], px[ate_power]] = False

    # Remaining pellets for progress calculation (after eating)
    remaining = pellets.sum(dim=(1, 2)).float() + power_pellets.sum(dim=(1, 2)).float()  # (N,)
    progress = 1.0 - remaining / max(total_pellets_start, 1)

    # Escalating rewards
    reward = reward + (1.0 + 2.0 * progress) * ate_pellet.float()
    reward = reward + (2.0 + 2.0 * progress) * ate_power.float()

    return reward, ate_pellet, ate_power


def check_level_complete(
    pellets: torch.Tensor,
    power_pellets: torch.Tensor,
    step_count: torch.Tensor,
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check if any environments have completed the level (all pellets eaten).

    Args:
        pellets: (N, ROWS, COLS) bool.
        power_pellets: (N, ROWS, COLS) bool.
        step_count: (N,) int — current step count per env.
        max_steps: int — max steps per episode.

    Returns:
        Tuple of:
            completed: (N,) bool — which envs just completed.
            completion_reward: (N,) float — reward for completing.
    """
    remaining = pellets.sum(dim=(1, 2)) + power_pellets.sum(dim=(1, 2))  # (N,)
    completed = remaining == 0

    # Time bonus: 5.0 base + up to 5.0 for speed
    time_remaining = (max_steps - step_count.float()).clamp(min=0)
    time_bonus = 5.0 * (time_remaining / max_steps)
    completion_reward = (5.0 + time_bonus) * completed.float()

    return completed, completion_reward
