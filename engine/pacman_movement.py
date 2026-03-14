"""Batched Pacman movement — vectorized across N environments.

All operations are pure tensor ops with no Python loops over the batch dimension.
"""

import torch

from .constants import COLS


def wrap_tunnel(x: torch.Tensor, y: torch.Tensor, tunnel_y: int) -> torch.Tensor:
    """Apply tunnel wrapping: on the tunnel row, x wraps around.

    Args:
        x: (N,) int tensor of x coordinates.
        y: (N,) int tensor of y coordinates.
        tunnel_y: Row index of the tunnel.

    Returns:
        Wrapped x tensor (N,).
    """
    on_tunnel = y == tunnel_y
    x = torch.where(on_tunnel & (x < 0), torch.tensor(COLS - 1, device=x.device, dtype=x.dtype), x)
    x = torch.where(on_tunnel & (x >= COLS), torch.tensor(0, device=x.device, dtype=x.dtype), x)
    return x


def is_blocked_pacman(
    x: torch.Tensor,
    y: torch.Tensor,
    pacman_blocked: torch.Tensor,
    tunnel_y: int,
) -> torch.Tensor:
    """Check if positions are blocked for Pacman (wall or ghost house door).

    Out-of-bounds positions are blocked, except on the tunnel row.

    Args:
        x: (N,) int tensor of x coordinates.
        y: (N,) int tensor of y coordinates.
        pacman_blocked: (ROWS, COLS) bool tensor (walls | door).
        tunnel_y: Row index of the tunnel.

    Returns:
        (N,) bool tensor — True where position is blocked.
    """
    rows, cols = pacman_blocked.shape

    # Out of bounds check (except tunnel row wraps)
    oob = (y < 0) | (y >= rows) | (x < 0) | (x >= cols)
    on_tunnel = y == tunnel_y
    oob = oob & ~on_tunnel  # tunnel row OOB is handled by wrapping before this call

    # Clamp for safe indexing (clamped positions will be masked by oob anyway)
    x_safe = x.clamp(0, cols - 1)
    y_safe = y.clamp(0, rows - 1)

    blocked = pacman_blocked[y_safe, x_safe]
    return oob | blocked


def update_pacman(
    pacman_pos: torch.Tensor,
    pacman_dir: torch.Tensor,
    next_dir: torch.Tensor,
    pacman_blocked: torch.Tensor,
    tunnel_y: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update Pacman positions for all N environments.

    Implements the buffered-direction system:
    1. If next_dir leads to a valid tile, adopt it as the current direction.
    2. Move one tile in the current direction if that tile is valid.

    Args:
        pacman_pos: (N, 2) int — current positions (x, y).
        pacman_dir: (N, 2) int — current movement directions (dx, dy).
        next_dir: (N, 2) int — buffered directions from action (dx, dy).
        pacman_blocked: (ROWS, COLS) bool — walls | ghost house door.
        tunnel_y: int — tunnel row index.

    Returns:
        Tuple of (new_pacman_pos, new_pacman_dir), same shapes as inputs.
    """
    # Step 1: Try buffered direction
    has_next = (next_dir[:, 0] != 0) | (next_dir[:, 1] != 0)  # (N,)

    if has_next.any():
        cand_x = pacman_pos[:, 0] + next_dir[:, 0]
        cand_y = pacman_pos[:, 1] + next_dir[:, 1]
        cand_x = wrap_tunnel(cand_x, cand_y, tunnel_y)
        cand_blocked = is_blocked_pacman(cand_x, cand_y, pacman_blocked, tunnel_y)

        # Where next_dir is set and not blocked, adopt it
        adopt = has_next & ~cand_blocked
        pacman_dir = torch.where(adopt.unsqueeze(1), next_dir, pacman_dir)

    # Step 2: Move in current direction
    has_dir = (pacman_dir[:, 0] != 0) | (pacman_dir[:, 1] != 0)  # (N,)

    if has_dir.any():
        move_x = pacman_pos[:, 0] + pacman_dir[:, 0]
        move_y = pacman_pos[:, 1] + pacman_dir[:, 1]
        move_x = wrap_tunnel(move_x, move_y, tunnel_y)
        move_blocked = is_blocked_pacman(move_x, move_y, pacman_blocked, tunnel_y)

        can_move = has_dir & ~move_blocked
        new_x = torch.where(can_move, move_x, pacman_pos[:, 0])
        new_y = torch.where(can_move, move_y, pacman_pos[:, 1])
        pacman_pos = torch.stack([new_x, new_y], dim=1)

    return pacman_pos, pacman_dir
