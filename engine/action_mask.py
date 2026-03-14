"""Batched valid action mask computation for Pacman."""

import torch

from .constants import ACTION_DIRS
from .pacman_movement import wrap_tunnel, is_blocked_pacman


# Map direction vector (dx, dy) to action index.
# pacman_dir is (dx, dy) as int tensor, NOT an action index.
# CRITICAL: This avoids the original engine's silent bug where tuple directions
# were compared against integer keys, making reverse masking a no-op.
_DIR_TO_ACTION = {
    (0, -1): 0,   # UP
    (0, 1): 1,    # DOWN
    (-1, 0): 2,   # LEFT
    (1, 0): 3,    # RIGHT
}

# Reverse action: UP<->DOWN, LEFT<->RIGHT
_REVERSE_ACTION = torch.tensor([1, 0, 3, 2], dtype=torch.long)


def get_action_mask(
    pacman_pos: torch.Tensor,      # (N, 2) int — (x, y)
    pacman_dir: torch.Tensor,      # (N, 2) int — (dx, dy)
    pacman_blocked: torch.Tensor,  # (ROWS, COLS) bool — walls | door
    tunnel_y: int,
    no_reverse: bool = False,
    ghost_pos: torch.Tensor | None = None,      # (N, 4, 2) int
    ghost_in_house: torch.Tensor | None = None,  # (N, 4) bool
    proximity_threshold: int = 4,
) -> torch.Tensor:
    """Compute valid action mask for all N environments.

    Actions: UP=0, DOWN=1, LEFT=2, RIGHT=3.
    An action is valid if the adjacent tile is not blocked (wall or ghost house door).

    Args:
        pacman_pos: (N, 2) int — current positions (x, y).
        pacman_dir: (N, 2) int — current directions (dx, dy).
        pacman_blocked: (ROWS, COLS) bool — walls | ghost house door.
        tunnel_y: int — tunnel row index.
        no_reverse: bool — if True, mask the reverse of Pacman's current
            direction as anti-oscillation. When ghost_pos/ghost_in_house are
            provided, reverse is allowed for envs where any non-house ghost is
            within Manhattan distance ``proximity_threshold`` (so the agent can
            flee). When ghost data is *not* provided, reverse is masked for all
            envs unconditionally (legacy Stage 1 behaviour).
        ghost_pos: (N, 4, 2) int — ghost positions. Optional.
        ghost_in_house: (N, 4) bool — True if ghost is in the house. Optional.
        proximity_threshold: Manhattan distance at which a nearby ghost
            disables the reverse mask for that env (default 4).

    Returns:
        mask: (N, 4) bool — True where action is valid.
    """
    N = pacman_pos.shape[0]
    device = pacman_pos.device
    action_dirs = ACTION_DIRS.to(device)  # (4, 2)

    px = pacman_pos[:, 0]  # (N,)
    py = pacman_pos[:, 1]  # (N,)

    # Check each of 4 directions
    mask = torch.zeros(N, 4, dtype=torch.bool, device=device)
    for a in range(4):
        dx, dy = action_dirs[a, 0], action_dirs[a, 1]
        nx = px + dx
        ny = py + dy
        nx = wrap_tunnel(nx, ny, tunnel_y)
        blocked = is_blocked_pacman(nx, ny, pacman_blocked, tunnel_y)
        mask[:, a] = ~blocked

    if no_reverse:
        reverse_action = _REVERSE_ACTION.to(device)

        dir_dx = pacman_dir[:, 0]  # (N,)
        dir_dy = pacman_dir[:, 1]  # (N,)
        has_dir = (dir_dx != 0) | (dir_dy != 0)  # (N,)

        if has_dir.any():
            # Match direction vector to action index
            dir_expanded = pacman_dir.unsqueeze(1)  # (N, 1, 2)
            dirs_expanded = action_dirs.unsqueeze(0)  # (1, 4, 2)
            match = (dir_expanded == dirs_expanded).all(dim=2)  # (N, 4)
            cur_action_idx = match.long().argmax(dim=1)  # (N,)

            rev_idx = reverse_action[cur_action_idx]  # (N,)

            # Per-env: skip reverse masking when a ghost is nearby
            if ghost_pos is not None and ghost_in_house is not None:
                # Manhattan distance to each ghost: (N, 4)
                pac_expanded = pacman_pos.unsqueeze(1)  # (N, 1, 2)
                dist = (pac_expanded - ghost_pos).abs().sum(dim=2)  # (N, 4)
                outside = ~ghost_in_house  # (N, 4)
                ghost_nearby = (outside & (dist <= proximity_threshold)).any(dim=1)  # (N,)
            else:
                ghost_nearby = torch.zeros(N, dtype=torch.bool, device=device)

            # Build reverse mask
            rev_mask = torch.zeros(N, 4, dtype=torch.bool, device=device)
            rev_mask[torch.arange(N, device=device), rev_idx] = True

            other_valid = mask.clone()
            other_valid[torch.arange(N, device=device), rev_idx] = False
            can_mask = has_dir & other_valid.any(dim=1) & ~ghost_nearby  # (N,)

            mask = mask & ~(rev_mask & can_mask.unsqueeze(1))

    return mask
