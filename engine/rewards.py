"""Visit heatmap, proximity reward, and ghost proximity penalty — batched."""

import torch

from .constants import ROWS, COLS


def update_visit_map_and_penalty(
    visit_map: torch.Tensor,
    pacman_pos: torch.Tensor,
    pellets: torch.Tensor,
    power_pellets: torch.Tensor,
    total_pellets_start: int,
    ghosts_active: torch.Tensor,
    active: torch.Tensor,
    decay: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decay visit map, stamp Pacman position, compute visit penalty.

    Visit penalty is only active when no ghosts are out (Stage 1 curriculum).

    Args:
        visit_map: (N, ROWS, COLS) float32 — modified in-place.
        pacman_pos: (N, 2) int.
        pellets: (N, ROWS, COLS) bool.
        power_pellets: (N, ROWS, COLS) bool.
        total_pellets_start: int.
        ghosts_active: (N,) bool — whether any ghost is out of the house.
        active: (N,) bool — which envs are active.
        decay: float — visit map decay factor.

    Returns:
        visit_penalty: (N,) float — negative penalty for revisiting.
        visit_map: (N, ROWS, COLS) — updated in-place, returned for convenience.
    """
    N = pacman_pos.shape[0]
    device = pacman_pos.device
    visit_penalty = torch.zeros(N, dtype=torch.float32, device=device)

    # Decay
    visit_map *= decay

    px = pacman_pos[:, 0].long()
    py = pacman_pos[:, 1].long()
    env_idx = torch.arange(N, device=device)

    # Visit penalty: stronger when no ghosts (Stage 1 anti-oscillation),
    # weaker when ghosts are active (allows evasion circling).
    heat = visit_map[env_idx, py, px]  # (N,)
    coeff = torch.where(ghosts_active,
                        torch.tensor(0.15, device=device),
                        torch.tensor(0.30, device=device))
    visit_penalty = torch.where(active, -coeff * heat, visit_penalty)

    # Stamp current position
    visit_map[env_idx[active], py[active], px[active]] = 1.0

    return visit_penalty, visit_map


def proximity_reward(
    pacman_pos: torch.Tensor,
    pellets: torch.Tensor,
    power_pellets: torch.Tensor,
    total_pellets_start: int,
    prev_nearest_dist: torch.Tensor,
    dist_matrix: torch.Tensor,
    tile_to_idx: torch.Tensor,
    active: torch.Tensor,
    game_over: torch.Tensor,
    level_complete: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute proximity reward for moving closer to nearest pellet.

    Uses precomputed distance matrix instead of per-step BFS.

    Args:
        pacman_pos: (N, 2) int.
        pellets: (N, ROWS, COLS) bool.
        power_pellets: (N, ROWS, COLS) bool.
        total_pellets_start: int.
        prev_nearest_dist: (N,) float — previous nearest pellet distance.
        dist_matrix: (num_tiles, num_tiles) int16 — precomputed distances.
        tile_to_idx: (ROWS, COLS) int32 — grid position to tile index.
        active: (N,) bool.
        game_over: (N,) bool.
        level_complete: (N,) bool.

    Returns:
        reward: (N,) float.
        new_nearest_dist: (N,) float — updated nearest distance.
    """
    N = pacman_pos.shape[0]
    device = pacman_pos.device
    reward = torch.zeros(N, dtype=torch.float32, device=device)

    # Skip if game is over or complete
    skip = game_over | level_complete | ~active
    if skip.all():
        return reward, prev_nearest_dist

    px = pacman_pos[:, 0].long()
    py = pacman_pos[:, 1].long()

    # Get Pacman's tile index: (N,)
    pac_tile_idx = tile_to_idx[py, px].long()

    # Build uneaten pellet mask over all tiles: (N, num_tiles)
    # tile_to_idx maps (y, x) -> tile_idx. We need the reverse: for each tile,
    # check if there's a pellet at its position.
    num_tiles = dist_matrix.shape[0]

    # For all pellet positions, we need to know which tile indices have uneaten pellets.
    # Strategy: flatten pellet grids and index using precomputed mapping.
    # pellets is (N, ROWS, COLS) and tile_to_idx is (ROWS, COLS)
    # For valid tiles (tile_to_idx >= 0), check if pellet exists

    valid_tile_mask = tile_to_idx >= 0  # (ROWS, COLS)
    valid_yx = valid_tile_mask.nonzero()  # (num_tiles, 2) as (y, x)
    valid_y = valid_yx[:, 0]
    valid_x = valid_yx[:, 1]
    valid_idx = tile_to_idx[valid_y, valid_x].long()  # (num_tiles,)

    # Check pellets at each valid tile for each env
    has_pellet = pellets[:, valid_y, valid_x]  # (N, num_tiles)
    has_power = power_pellets[:, valid_y, valid_x]  # (N, num_tiles)
    uneaten = has_pellet | has_power  # (N, num_tiles)

    # Get distances from Pacman to all tiles: (N, num_tiles)
    # dist_matrix is indexed by tile index
    all_dists = dist_matrix[pac_tile_idx]  # (N, num_tiles)
    all_dists = all_dists.float()

    # Mask unreachable/no-pellet tiles
    all_dists = torch.where(uneaten, all_dists, torch.tensor(9999.0, device=device))

    # Nearest pellet distance: (N,)
    nearest_dist = all_dists.min(dim=1).values

    # Handle case where no pellets remain (should be caught by level_complete)
    no_pellets = nearest_dist >= 9999
    nearest_dist = torch.where(no_pellets, torch.zeros_like(nearest_dist), nearest_dist)

    # Compute reward delta
    has_prev = prev_nearest_dist > 0
    delta = prev_nearest_dist - nearest_dist  # positive = moved closer

    remaining = pellets.sum(dim=(1, 2)).float() + power_pellets.sum(dim=(1, 2)).float()
    remaining_frac = remaining / max(total_pellets_start, 1)
    prox_scale = 0.1 + 0.2 * (1.0 - remaining_frac)

    apply = active & has_prev & ~skip
    reward = torch.where(apply, prox_scale * delta, reward)

    # Update nearest dist (only for active envs)
    new_nearest_dist = torch.where(active & ~skip, nearest_dist, prev_nearest_dist)

    return reward, new_nearest_dist


def ghost_proximity_penalty(
    pacman_pos: torch.Tensor,
    ghost_pos: torch.Tensor,
    ghost_in_house: torch.Tensor,
    ghost_is_frightened: torch.Tensor,
    ghost_is_eaten: torch.Tensor,
    active: torch.Tensor,
    game_over: torch.Tensor,
    level_complete: torch.Tensor,
) -> torch.Tensor:
    """Penalize Pacman for being near non-frightened ghosts.

    Args:
        pacman_pos: (N, 2) int.
        ghost_pos: (N, 4, 2) int.
        ghost_in_house: (N, 4) bool.
        ghost_is_frightened: (N, 4) bool.
        ghost_is_eaten: (N, 4) bool.
        active: (N,) bool.
        game_over: (N,) bool.
        level_complete: (N,) bool.

    Returns:
        penalty: (N,) float — negative values.
    """
    N = pacman_pos.shape[0]
    device = pacman_pos.device
    penalty = torch.zeros(N, dtype=torch.float32, device=device)

    skip = game_over | level_complete | ~active
    if skip.all():
        return penalty

    # Manhattan distance to each ghost: (N, 4)
    pac_expanded = pacman_pos.unsqueeze(1)  # (N, 1, 2)
    dist = (pac_expanded - ghost_pos).abs().sum(dim=2)  # (N, 4)

    # Exclude ghosts in house, frightened, or eaten
    excluded = ghost_in_house | ghost_is_frightened | ghost_is_eaten  # (N, 4)

    # Penalty for ghosts within distance 4
    close = (dist <= 4) & ~excluded  # (N, 4)
    per_ghost_penalty = -0.075 * (5 - dist.float())  # (N, 4)
    per_ghost_penalty = torch.where(close, per_ghost_penalty, torch.zeros_like(per_ghost_penalty))

    # Sum across ghosts
    penalty = per_ghost_penalty.sum(dim=1)  # (N,)
    penalty = torch.where(skip, torch.zeros_like(penalty), penalty)

    return penalty
