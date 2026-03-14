"""Batched ghost movement, AI targeting, house exit, and collision detection.

All operations are vectorized across N environments and 4 ghosts.
No Python loops over N — ghost index loops (4 iterations) are acceptable
since 4 is constant and tiny.
"""

import torch

from .constants import (
    COLS, ROWS,
    SCATTER, CHASE, FRIGHTENED, EATEN,
    SCATTER_DURATION, CHASE_DURATION, FRIGHTENED_DURATION,
    GHOST_HOUSE_TARGET, GHOST_HOUSE_DOOR,
    GHOST_SCATTER_TARGETS, GHOST_SCORE,
    POWERUP_DURATION,
)
from .pacman_movement import wrap_tunnel


# Direction vectors for the 4 movement options: UP, DOWN, LEFT, RIGHT
_DIRS = torch.tensor([
    [0, -1],  # UP
    [0, 1],   # DOWN
    [-1, 0],  # LEFT
    [1, 0],   # RIGHT
], dtype=torch.int32)


def ghost_house_step(
    ghost_pos: torch.Tensor,       # (N, 4, 2)
    ghost_in_house: torch.Tensor,  # (N, 4) bool
    ghost_exit_timer: torch.Tensor,  # (N, 4) int
    door_pos: torch.Tensor,        # (2,) int — (x, y) of door
    active: torch.Tensor,          # (N,) bool
    walls: torch.Tensor | None = None,  # (ROWS, COLS) bool — for pathfinding
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Handle ghost house exit logic for in-house ghosts.

    For each ghost that is in_house:
    - If exit_timer > 0: decrement and skip
    - If at door position: set in_house=False, move up one tile
    - Otherwise: move toward door (vertical first if passable, matching A*)

    Returns updated ghost_pos, ghost_in_house, ghost_exit_timer.
    """
    N = ghost_pos.shape[0]
    device = ghost_pos.device

    for gi in range(4):
        in_house = ghost_in_house[:, gi] & active  # (N,)

        if not in_house.any():
            continue

        # Check timer BEFORE decrementing: ghosts with timer already <= 0
        # can move. Original engine decrements and returns (no movement that step).
        timer_done = in_house & (ghost_exit_timer[:, gi] <= 0)

        # Decrement exit timer for in-house ghosts with timer > 0
        has_timer = in_house & ~timer_done
        ghost_exit_timer[:, gi] = torch.where(
            has_timer, ghost_exit_timer[:, gi] - 1, ghost_exit_timer[:, gi]
        )

        if not timer_done.any():
            continue

        gx = ghost_pos[:, gi, 0]  # (N,)
        gy = ghost_pos[:, gi, 1]  # (N,)
        dx_door = door_pos[0]
        dy_door = door_pos[1]

        # Check if at door
        at_door = timer_done & (gx == dx_door) & (gy == dy_door)

        # At door: exit — move up one tile, set in_house=False
        ghost_pos[:, gi, 1] = torch.where(at_door, gy - 1, ghost_pos[:, gi, 1])
        ghost_in_house[:, gi] = torch.where(at_door, torch.zeros_like(ghost_in_house[:, gi]),
                                             ghost_in_house[:, gi])

        # Not at door, timer done: move toward door
        # Vertical first (UP) if passable, else horizontal — matches A*'s
        # direction ordering: [(0,-1), (0,1), (-1,0), (1,0)]
        moving = timer_done & ~at_door
        if moving.any():
            moved = torch.zeros(N, dtype=torch.bool, device=device)

            # Try UP first if ghost needs to go up and cell above is passable
            want_up = (gy > dy_door) & moving
            if want_up.any() and walls is not None:
                up_y = (gy - 1).clamp(0, ROWS - 1).long()
                gx_safe = gx.clamp(0, COLS - 1).long()
                up_passable = ~walls[up_y, gx_safe]
                go_up = want_up & up_passable
                ghost_pos[:, gi, 1] = torch.where(go_up, gy - 1, ghost_pos[:, gi, 1])
                moved = go_up

            # Horizontal: for ghosts that couldn't move up (or don't need to)
            need_x = moving & ~moved & (gx != dx_door)
            if need_x.any():
                dx = torch.sign(dx_door - gx).int()
                ghost_pos[:, gi, 0] = torch.where(need_x, gx + dx, ghost_pos[:, gi, 0])
                moved = moved | need_x

            # Vertical fallback: for ghosts already x-aligned but still need y movement
            need_y = moving & ~moved & (gy != dy_door)
            if need_y.any():
                dy = torch.sign(dy_door - gy).int()
                ghost_pos[:, gi, 1] = torch.where(need_y, gy + dy, ghost_pos[:, gi, 1])

    return ghost_pos, ghost_in_house, ghost_exit_timer


def compute_chase_targets(
    pacman_pos: torch.Tensor,   # (N, 2)
    pacman_dir: torch.Tensor,   # (N, 2)
    ghost_pos: torch.Tensor,    # (N, 4, 2)
    scatter_targets: torch.Tensor,  # (4, 2) — static scatter corners
    device: torch.device,
) -> torch.Tensor:
    """Compute chase targets for all 4 ghost personalities.

    Returns: (N, 4, 2) int — chase target (x, y) per ghost per env.
    """
    N = pacman_pos.shape[0]
    targets = torch.zeros(N, 4, 2, dtype=torch.int32, device=device)

    # Blinky (0): chase Pacman directly
    targets[:, 0] = pacman_pos

    # Pinky (1): target 4 tiles ahead of Pacman
    targets[:, 1] = pacman_pos + pacman_dir * 4

    # Inky (2): 2 * (pacman + 2*dir) - blinky_pos
    ahead = pacman_pos + pacman_dir * 2
    blinky_pos = ghost_pos[:, 0]  # (N, 2)
    targets[:, 2] = 2 * ahead - blinky_pos

    # Clyde (3): chase if dist > 8, else scatter
    clyde_pos = ghost_pos[:, 3]  # (N, 2)
    dist = (clyde_pos - pacman_pos).abs().sum(dim=1)  # (N,) manhattan
    far = dist > 8  # (N,)
    clyde_scatter = scatter_targets[3].unsqueeze(0).expand(N, 2).to(device)  # (N, 2)
    targets[:, 3] = torch.where(far.unsqueeze(1), pacman_pos, clyde_scatter)

    return targets


def ghost_ai_move(
    ghost_pos: torch.Tensor,        # (N, 4, 2)
    ghost_dir: torch.Tensor,        # (N, 4, 2)
    ghost_state: torch.Tensor,      # (N, 4) int
    ghost_in_house: torch.Tensor,   # (N, 4) bool
    ghost_move_timer: torch.Tensor, # (N, 4) int
    ghost_speed: torch.Tensor,      # (N, 4) int
    pacman_pos: torch.Tensor,       # (N, 2)
    pacman_dir: torch.Tensor,       # (N, 2)
    walls: torch.Tensor,            # (ROWS, COLS) bool
    door_mask: torch.Tensor,        # (ROWS, COLS) bool
    tunnel_y: int,
    scatter_targets: torch.Tensor,  # (4, 2)
    active: torch.Tensor,           # (N,) bool
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move ghosts using built-in AI. Handles speed timer, targeting, and pathfinding.

    Returns updated ghost_pos, ghost_dir, ghost_move_timer.
    """
    N = ghost_pos.shape[0]
    device = ghost_pos.device
    dirs = _DIRS.to(device)

    # Frightened timer decrement (happens before speed check, outside can_move)
    frightened = ghost_state == FRIGHTENED
    # Timer is managed in the caller (batched_game.py)

    # Check eaten ghosts reaching ghost house target
    # (handled in caller before this function)

    # Speed timer: can_move check
    # Increment timer for all active non-in-house ghosts
    outside = active.unsqueeze(1) & ~ghost_in_house  # (N, 4)
    ghost_move_timer = torch.where(outside, ghost_move_timer + 1, ghost_move_timer)
    can_move = outside & (ghost_move_timer >= ghost_speed)
    ghost_move_timer = torch.where(can_move, torch.zeros_like(ghost_move_timer), ghost_move_timer)

    if not can_move.any():
        return ghost_pos, ghost_dir, ghost_move_timer

    # Compute targets
    # Chase targets depend on ghost personality
    chase_targets = compute_chase_targets(
        pacman_pos, pacman_dir, ghost_pos, scatter_targets, device,
    )

    # Scatter targets: (4, 2) expanded to (N, 4, 2)
    scatter_tgts = scatter_targets.unsqueeze(0).expand(N, 4, 2).to(device)

    # Select target based on state
    is_scatter = (ghost_state == SCATTER).unsqueeze(2)  # (N, 4, 1)
    is_chase = (ghost_state == CHASE).unsqueeze(2)
    is_eaten = (ghost_state == EATEN).unsqueeze(2)

    house_target = torch.tensor(
        [GHOST_HOUSE_TARGET[0], GHOST_HOUSE_TARGET[1]],
        dtype=torch.int32, device=device,
    ).unsqueeze(0).unsqueeze(0).expand(N, 4, 2)

    target = torch.where(is_scatter, scatter_tgts,
             torch.where(is_chase, chase_targets,
             torch.where(is_eaten, house_target,
             ghost_pos)))  # FRIGHTENED: target = self (unused, random movement)

    # Process each ghost (4 is constant, loop is fine)
    for gi in range(4):
        mask = can_move[:, gi]  # (N,)
        if not mask.any():
            continue

        gx = ghost_pos[:, gi, 0]  # (N,)
        gy = ghost_pos[:, gi, 1]  # (N,)
        cur_dir = ghost_dir[:, gi]  # (N, 2)
        state = ghost_state[:, gi]  # (N,)
        tgt = target[:, gi]  # (N, 2)

        # Compute 4 candidate next positions
        cand_x = gx.unsqueeze(1) + dirs[:, 0].unsqueeze(0)  # (N, 4)
        cand_y = gy.unsqueeze(1) + dirs[:, 1].unsqueeze(0)  # (N, 4)

        # Tunnel wrapping
        cand_x = wrap_tunnel(cand_x.reshape(-1), cand_y.reshape(-1), tunnel_y).reshape(N, 4)

        # Wall check — clamp for safe indexing
        cx_safe = cand_x.clamp(0, COLS - 1).long()
        cy_safe = cand_y.clamp(0, ROWS - 1).long()

        # Out of bounds check (except tunnel row)
        oob = (cand_y < 0) | (cand_y >= ROWS) | (cand_x < 0) | (cand_x >= COLS)
        on_tunnel = cand_y == tunnel_y
        oob = oob & ~on_tunnel

        is_wall = walls[cy_safe, cx_safe] | oob  # (N, 4)

        # Door check: only allow if EATEN
        is_door = door_mask[cy_safe, cx_safe]  # (N, 4)
        eaten_ghost = (state == EATEN).unsqueeze(1).expand_as(is_door)
        blocked = is_wall | (is_door & ~eaten_ghost)

        valid = ~blocked  # (N, 4)

        # Reverse direction check: ghosts can't reverse (except when no forward move)
        reverse_dx = -cur_dir[:, 0]  # (N,)
        reverse_dy = -cur_dir[:, 1]  # (N,)

        # Check which candidate directions match the reverse
        is_reverse = (dirs[:, 0].unsqueeze(0) == reverse_dx.unsqueeze(1)) & \
                     (dirs[:, 1].unsqueeze(0) == reverse_dy.unsqueeze(1))  # (N, 4)

        forward = valid & ~is_reverse  # (N, 4)

        # If no forward moves available, allow reverse
        no_forward = ~forward.any(dim=1)  # (N,)
        forward = torch.where(no_forward.unsqueeze(1), valid, forward)

        # Choose direction based on state
        is_fright = (state == FRIGHTENED)  # (N,)

        # --- Frightened: random from valid forward moves ---
        # Assign random priorities, pick argmax among valid
        if rng is not None:
            rand_scores = torch.rand(N, 4, device=device, generator=rng)
        else:
            rand_scores = torch.rand(N, 4, device=device)
        rand_scores = torch.where(forward, rand_scores, torch.tensor(-1.0, device=device))
        fright_choice = rand_scores.argmax(dim=1)  # (N,)

        # --- Chase/Scatter/Eaten: minimize manhattan distance to target ---
        tgt_x = tgt[:, 0].unsqueeze(1)  # (N, 1)
        tgt_y = tgt[:, 1].unsqueeze(1)  # (N, 1)
        manhattan = (cand_x - tgt_x).abs() + (cand_y - tgt_y).abs()  # (N, 4)

        # Set invalid to huge distance
        manhattan = torch.where(forward, manhattan.float(),
                                torch.tensor(float('inf'), device=device))

        # For ties, the original iterates UP(0), DOWN(1), LEFT(2), RIGHT(3)
        # and picks first minimum. argmin does the same (returns first min index).
        chase_choice = manhattan.argmin(dim=1)  # (N,)

        # Select: frightened uses random, others use chase
        chosen_idx = torch.where(is_fright, fright_choice, chase_choice)  # (N,)

        # Get chosen direction and position
        chosen_dir = dirs[chosen_idx.long()]  # (N, 2)
        chosen_x = cand_x[torch.arange(N, device=device), chosen_idx.long()]  # (N,)
        chosen_y = cand_y[torch.arange(N, device=device), chosen_idx.long()]  # (N,)

        # Apply movement only where mask is True
        ghost_dir[:, gi] = torch.where(mask.unsqueeze(1), chosen_dir, ghost_dir[:, gi])
        ghost_pos[:, gi, 0] = torch.where(mask, chosen_x, ghost_pos[:, gi, 0])
        ghost_pos[:, gi, 1] = torch.where(mask, chosen_y, ghost_pos[:, gi, 1])

    return ghost_pos, ghost_dir, ghost_move_timer


def check_ghost_collisions(
    pacman_pos: torch.Tensor,       # (N, 2)
    prev_pac_pos: torch.Tensor,     # (N, 2)
    ghost_pos: torch.Tensor,        # (N, 4, 2)
    prev_ghost_pos: torch.Tensor,   # (N, 4, 2)
    ghost_state: torch.Tensor,      # (N, 4) int
    ghost_in_house: torch.Tensor,   # (N, 4) bool
    active: torch.Tensor,           # (N,) bool
) -> torch.Tensor:
    """Detect collisions between Pacman and ghosts (same-tile + swap).

    Returns: collision mask (N, 4) bool — True where collision occurred.
    """
    # Same-tile: pacman_pos == ghost_pos
    same_tile = (pacman_pos.unsqueeze(1) == ghost_pos).all(dim=2)  # (N, 4)

    # Swap: pacman moved to ghost's old pos AND ghost moved to pacman's old pos
    pac_to_ghost_old = (pacman_pos.unsqueeze(1) == prev_ghost_pos).all(dim=2)  # (N, 4)
    ghost_to_pac_old = (ghost_pos == prev_pac_pos.unsqueeze(1)).all(dim=2)  # (N, 4)
    swapped = pac_to_ghost_old & ghost_to_pac_old

    # Collision: (same_tile OR swap) AND ghost not in house AND env is active
    collision = (same_tile | swapped) & ~ghost_in_house & active.unsqueeze(1)

    return collision


def handle_ghost_collisions(
    collision: torch.Tensor,         # (N, 4) bool
    ghost_state: torch.Tensor,       # (N, 4) int
    ghost_prev_state: torch.Tensor,  # (N, 4) int
    ghost_in_house: torch.Tensor,    # (N, 4) bool
    ghost_exit_timer: torch.Tensor,  # (N, 4) int
    ghost_pos: torch.Tensor,         # (N, 4, 2)
    ghost_eat_combo: torch.Tensor,   # (N,) int
    score: torch.Tensor,             # (N,) int
    reward_pacman: torch.Tensor,     # (N,) float
    reward_ghost: torch.Tensor,      # (N,) float
    game_over: torch.Tensor,         # (N,) bool
) -> tuple:
    """Handle ghost-Pacman collisions: eat frightened ghosts or kill Pacman.

    Returns updated state tensors.
    """
    N = collision.shape[0]
    device = collision.device

    for gi in range(4):
        col = collision[:, gi]  # (N,)
        if not col.any():
            continue

        state = ghost_state[:, gi]  # (N,)

        # Frightened ghost: Pacman eats it
        fright_col = col & (state == FRIGHTENED)
        if fright_col.any():
            ghost_state[:, gi] = torch.where(fright_col,
                                              torch.full_like(state, EATEN), state)
            ghost_eat_combo = torch.where(fright_col, ghost_eat_combo + 1, ghost_eat_combo)
            points = GHOST_SCORE * ghost_eat_combo
            score = torch.where(fright_col, score + points, score)
            reward_pacman = torch.where(fright_col,
                                         reward_pacman + 0.5 * ghost_eat_combo.float(),
                                         reward_pacman)
            reward_ghost = torch.where(fright_col, reward_ghost - 0.5, reward_ghost)

        # Normal ghost: Pacman dies
        normal_col = col & (state != FRIGHTENED) & (state != EATEN)
        if normal_col.any():
            reward_pacman = torch.where(normal_col, reward_pacman - 5.0, reward_pacman)
            reward_ghost = torch.where(normal_col, reward_ghost + 1.0, reward_ghost)
            game_over = game_over | normal_col  # 1 life in headless = game over

    return (ghost_state, ghost_eat_combo, score,
            reward_pacman, reward_ghost, game_over)


def update_ghost_mode(
    mode_timer: torch.Tensor,     # (N,) int
    current_mode: torch.Tensor,   # (N,) int
    ghost_state: torch.Tensor,    # (N, 4) int
    active: torch.Tensor,         # (N,) bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Timer-based scatter/chase mode switching.

    Returns updated mode_timer, current_mode, ghost_state.
    """
    mode_timer = torch.where(active, mode_timer + 1, mode_timer)

    cycle_length = SCATTER_DURATION + CHASE_DURATION
    phase = mode_timer % cycle_length

    new_mode = torch.where(phase < SCATTER_DURATION,
                           torch.full_like(current_mode, SCATTER),
                           torch.full_like(current_mode, CHASE))

    # Detect mode change
    changed = active & (new_mode != current_mode)
    current_mode = torch.where(active, new_mode, current_mode)

    # Apply mode to ghosts not in FRIGHTENED or EATEN states
    if changed.any():
        for gi in range(4):
            state = ghost_state[:, gi]
            normal = (state != FRIGHTENED) & (state != EATEN)
            apply = changed & normal
            ghost_state[:, gi] = torch.where(apply, new_mode, state)

    return mode_timer, current_mode, ghost_state


def frightened_timer_step(
    ghost_state: torch.Tensor,       # (N, 4) int
    ghost_prev_state: torch.Tensor,  # (N, 4) int
    ghost_fright_timer: torch.Tensor,  # (N, 4) int
    active: torch.Tensor,            # (N,) bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decrement frightened timer and end frightened mode when expired.

    Returns updated ghost_state, ghost_fright_timer.
    """
    frightened = (ghost_state == FRIGHTENED) & active.unsqueeze(1)

    ghost_fright_timer = torch.where(frightened, ghost_fright_timer - 1, ghost_fright_timer)

    # End frightened when timer <= 0
    expired = frightened & (ghost_fright_timer <= 0)
    ghost_state = torch.where(expired, ghost_prev_state, ghost_state)

    return ghost_state, ghost_fright_timer


def eaten_ghost_check(
    ghost_pos: torch.Tensor,         # (N, 4, 2)
    ghost_state: torch.Tensor,       # (N, 4) int
    ghost_prev_state: torch.Tensor,  # (N, 4) int
    ghost_in_house: torch.Tensor,    # (N, 4) bool
    ghost_exit_timer: torch.Tensor,  # (N, 4) int
    active: torch.Tensor,            # (N,) bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check if eaten ghosts have reached the ghost house target and re-enter.

    Returns updated ghost_state, ghost_prev_state, ghost_in_house, ghost_exit_timer.
    """
    target_x, target_y = GHOST_HOUSE_TARGET

    for gi in range(4):
        is_eaten = (ghost_state[:, gi] == EATEN) & active
        if not is_eaten.any():
            continue

        gx = ghost_pos[:, gi, 0]
        gy = ghost_pos[:, gi, 1]
        at_target = is_eaten & (gx == target_x) & (gy == target_y)

        if at_target.any():
            ghost_state[:, gi] = torch.where(
                at_target, ghost_prev_state[:, gi], ghost_state[:, gi]
            )
            ghost_in_house[:, gi] = torch.where(
                at_target, torch.ones_like(ghost_in_house[:, gi]),
                ghost_in_house[:, gi]
            )
            ghost_exit_timer[:, gi] = torch.where(
                at_target,
                torch.full_like(ghost_exit_timer[:, gi], 60),
                ghost_exit_timer[:, gi],
            )

    return ghost_state, ghost_prev_state, ghost_in_house, ghost_exit_timer


def activate_frightened(
    ate_power: torch.Tensor,          # (N,) bool — envs where power pellet was eaten
    ghost_state: torch.Tensor,        # (N, 4) int
    ghost_prev_state: torch.Tensor,   # (N, 4) int
    ghost_dir: torch.Tensor,          # (N, 4, 2) int
    ghost_fright_timer: torch.Tensor, # (N, 4) int
    ghost_eat_combo: torch.Tensor,    # (N,) int
    ghost_in_house: torch.Tensor,     # (N, 4) bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Activate frightened mode for all non-eaten ghosts when power pellet is eaten.

    Matches original: set_frightened() called for all ghosts with state != EATEN.
    - If not already FRIGHTENED: save prev state, reverse direction
    - Set state = FRIGHTENED, reset timer

    Returns updated ghost_state, ghost_prev_state, ghost_dir, ghost_fright_timer, ghost_eat_combo.
    """
    if not ate_power.any():
        return ghost_state, ghost_prev_state, ghost_dir, ghost_fright_timer, ghost_eat_combo

    # Reset eat combo for envs that ate a power pellet
    ghost_eat_combo = torch.where(ate_power, torch.zeros_like(ghost_eat_combo), ghost_eat_combo)

    for gi in range(4):
        state = ghost_state[:, gi]

        # Apply to: envs where power pellet eaten AND ghost is not EATEN
        apply = ate_power & (state != EATEN)
        if not apply.any():
            continue

        # Save previous state (only if not already FRIGHTENED)
        not_fright = apply & (state != FRIGHTENED)
        ghost_prev_state[:, gi] = torch.where(not_fright, state, ghost_prev_state[:, gi])

        # Reverse direction (only if not already FRIGHTENED and not in house)
        reverse = not_fright & ~ghost_in_house[:, gi]
        ghost_dir[:, gi] = torch.where(
            reverse.unsqueeze(1),
            -ghost_dir[:, gi],
            ghost_dir[:, gi],
        )

        # Set state to FRIGHTENED
        ghost_state[:, gi] = torch.where(apply, torch.full_like(state, FRIGHTENED), state)

        # Reset frightened timer
        ghost_fright_timer[:, gi] = torch.where(
            apply,
            torch.full_like(ghost_fright_timer[:, gi], FRIGHTENED_DURATION),
            ghost_fright_timer[:, gi],
        )

    return ghost_state, ghost_prev_state, ghost_dir, ghost_fright_timer, ghost_eat_combo
