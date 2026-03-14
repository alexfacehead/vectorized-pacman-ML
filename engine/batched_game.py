"""BatchedGame — vectorized Pac-Man engine running N games simultaneously.

Built incrementally. Current: Phase 4 (movement + pellets + rewards + ghosts).
"""

import torch

from .constants import (
    ROWS, COLS, ACTION_DIRS, READY_TIMER, MAX_STEPS,
    PELLET_SCORE, POWER_PELLET_SCORE,
    NUM_STATE_CHANNELS, CHANNEL_WALLS, CHANNEL_PELLETS,
    CHANNEL_POWER_PELLETS, CHANNEL_PACMAN, CHANNEL_GHOSTS,
    CHANNEL_FRIGHTENED, CHANNEL_VISITED,
    SCATTER, CHASE, FRIGHTENED, EATEN,
    GHOST_HOUSE_DOOR, GHOST_SCATTER_TARGETS,
)
from .maze import Maze
from .distances import precompute_distances
from .pacman_movement import update_pacman
from .pellets import collect_pellets, check_level_complete
from .rewards import update_visit_map_and_penalty, proximity_reward, ghost_proximity_penalty
from .ghosts import (
    ghost_house_step, ghost_ai_move, check_ghost_collisions,
    handle_ghost_collisions, update_ghost_mode, frightened_timer_step,
    eaten_ghost_check, activate_frightened,
)
from .action_mask import get_action_mask


class BatchedGame:
    """Manages N simultaneous Pac-Man games using batched tensor operations."""

    def __init__(self, n_envs: int, maze_file: str, device: torch.device = torch.device("cpu")):
        self.n = n_envs
        self.device = device

        # Load static maze data (shared across all envs)
        self.maze = Maze(maze_file)
        self.maze.to(device)

        # Precompute all-pairs shortest paths (one-time cost)
        self.dist_matrix = precompute_distances(self.maze).to(device)

        # Ensure ACTION_DIRS is on the right device
        self._action_dirs = ACTION_DIRS.to(device)

        # Ghost house door position
        self._door_pos = torch.tensor(
            [GHOST_HOUSE_DOOR[0], GHOST_HOUSE_DOOR[1]],
            dtype=torch.int32, device=device,
        )

        # Scatter targets on device
        self._scatter_targets = GHOST_SCATTER_TARGETS.to(device)

        # Pre-allocate double observation buffer (ping-pong for get_state)
        self._state_bufs = [
            torch.zeros(n_envs, NUM_STATE_CHANNELS, ROWS, COLS,
                        dtype=torch.float32, device=self.device)
            for _ in range(2)
        ]
        self._state_buf_idx = 0
        # Cache walls as float (static, never changes)
        self._walls_float = self.maze.walls.float()

        # Initialize all game state
        self._init_state()

    def _init_state(self) -> None:
        """Initialize (or reset) all state tensors for N environments."""
        N = self.n
        dev = self.device

        # Pacman state
        sx, sy = self.maze.pacman_start
        self.pacman_pos = torch.tensor([[sx, sy]], dtype=torch.int32, device=dev).expand(N, 2).contiguous()
        self.pacman_dir = torch.zeros(N, 2, dtype=torch.int32, device=dev)
        self.pacman_next_dir = torch.zeros(N, 2, dtype=torch.int32, device=dev)

        # Pellet state
        self.pellets = self.maze.initial_pellets.unsqueeze(0).expand(N, -1, -1).clone()
        self.power_pellets = self.maze.initial_power_pellets.unsqueeze(0).expand(N, -1, -1).clone()
        self.score = torch.zeros(N, dtype=torch.int32, device=dev)

        # Visit heatmap
        self.visit_map = torch.zeros(N, ROWS, COLS, dtype=torch.float32, device=dev)

        # Proximity tracking
        self.prev_nearest_dist = torch.zeros(N, dtype=torch.float32, device=dev)

        # Step counters and timers
        self.step_count = torch.zeros(N, dtype=torch.int32, device=dev)
        self.ready_timer = torch.full((N,), READY_TIMER, dtype=torch.int32, device=dev)

        # Done flags
        self.game_over = torch.zeros(N, dtype=torch.bool, device=dev)
        self.level_complete = torch.zeros(N, dtype=torch.bool, device=dev)

        # Rewards (reset each step)
        self.reward_pacman = torch.zeros(N, dtype=torch.float32, device=dev)
        self.reward_ghost = torch.zeros(N, dtype=torch.float32, device=dev)

        # Ghost state
        self.ghost_pos = self.maze.ghost_starts.unsqueeze(0).expand(N, -1, -1).contiguous().clone()
        self.ghost_dir = torch.zeros(N, 4, 2, dtype=torch.int32, device=dev)
        self.ghost_in_house = torch.ones(N, 4, dtype=torch.bool, device=dev)
        self.ghost_state = torch.full((N, 4), SCATTER, dtype=torch.int32, device=dev)
        self.ghost_prev_state = torch.full((N, 4), SCATTER, dtype=torch.int32, device=dev)
        # Curriculum defaults (matching original headless mode Stage 1):
        # Stage 1: all ghosts frozen (exit_timer=999999, speed=3)
        exit_timers = torch.tensor([999999, 999999, 999999, 999999], dtype=torch.int32, device=dev)
        self.ghost_exit_timer = exit_timers.unsqueeze(0).expand(N, 4).contiguous().clone()
        self.ghost_fright_timer = torch.zeros(N, 4, dtype=torch.int32, device=dev)
        self.ghost_move_timer = torch.zeros(N, 4, dtype=torch.int32, device=dev)
        speeds = torch.tensor([3, 3, 3, 3], dtype=torch.int32, device=dev)
        self.ghost_speed = speeds.unsqueeze(0).expand(N, 4).contiguous().clone()
        self.ghost_eat_combo = torch.zeros(N, dtype=torch.int32, device=dev)
        self.mode_timer = torch.zeros(N, dtype=torch.int32, device=dev)
        self.current_mode = torch.full((N,), SCATTER, dtype=torch.int32, device=dev)

        # Previous action tracking (for direction-change penalty)
        # -1 means no previous action yet
        self.prev_action = torch.full((N,), -1, dtype=torch.int32, device=dev)

        # Curriculum stage (default Stage 1: no ghosts)
        self._stage = 1
        self._stage_exit_timers = torch.tensor([999999, 999999, 999999, 999999],
                                                dtype=torch.int32, device=dev)
        self._stage_speeds = torch.tensor([3, 3, 3, 3], dtype=torch.int32, device=dev)

    def configure_stage(self, stage: int) -> None:
        """Set curriculum stage, controlling which ghosts are active.

        Stage 1: No ghosts (all frozen in house).
        Stage 2: Blinky slow (exit_timer=0, speed=2). Others frozen.
        Stage 3: Blinky full speed (exit_timer=0, speed=1). Others frozen.
        Stage 4: Blinky + Pinky slow (both exit_timer=0, speed=2). Others frozen.
        Stage 5: Blinky + Pinky + Inky slow (all speed=2). Clyde frozen.
        Stage 6: Blinky slow + Pinky fast (speed=2, speed=1). Others frozen.
        Stage 7: 3 ghosts mixed (Blinky speed=2, Pinky speed=1, Inky speed=1). Clyde frozen.
        """
        dev = self.device
        self._stage = stage
        if stage == 1:
            self._stage_exit_timers = torch.tensor([999999, 999999, 999999, 999999],
                                                    dtype=torch.int32, device=dev)
            self._stage_speeds = torch.tensor([3, 3, 3, 3], dtype=torch.int32, device=dev)
        elif stage == 2:
            self._stage_exit_timers = torch.tensor([0, 999999, 999999, 999999],
                                                    dtype=torch.int32, device=dev)
            self._stage_speeds = torch.tensor([2, 3, 3, 3], dtype=torch.int32, device=dev)
        elif stage == 3:
            self._stage_exit_timers = torch.tensor([0, 999999, 999999, 999999],
                                                    dtype=torch.int32, device=dev)
            self._stage_speeds = torch.tensor([1, 3, 3, 3], dtype=torch.int32, device=dev)
        elif stage == 4:
            self._stage_exit_timers = torch.tensor([0, 0, 999999, 999999],
                                                    dtype=torch.int32, device=dev)
            self._stage_speeds = torch.tensor([2, 2, 3, 3], dtype=torch.int32, device=dev)
        elif stage == 5:
            self._stage_exit_timers = torch.tensor([0, 0, 0, 999999],
                                                    dtype=torch.int32, device=dev)
            self._stage_speeds = torch.tensor([2, 2, 2, 3], dtype=torch.int32, device=dev)
        elif stage == 6:
            self._stage_exit_timers = torch.tensor([0, 0, 999999, 999999],
                                                    dtype=torch.int32, device=dev)
            self._stage_speeds = torch.tensor([2, 1, 3, 3], dtype=torch.int32, device=dev)
        elif stage == 7:
            self._stage_exit_timers = torch.tensor([0, 0, 0, 999999],
                                                    dtype=torch.int32, device=dev)
            self._stage_speeds = torch.tensor([2, 1, 1, 3], dtype=torch.int32, device=dev)
        else:
            raise ValueError(f"Unknown stage {stage}")

        # Apply to all envs immediately
        N = self.n
        self.ghost_exit_timer = self._stage_exit_timers.unsqueeze(0).expand(N, 4).contiguous().clone()
        self.ghost_speed = self._stage_speeds.unsqueeze(0).expand(N, 4).contiguous().clone()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Advance all N games by one step.

        Order of operations matches the original PacmanML engine:
        1. Time penalty + visit map (before movement)
        2. Move Pacman
        3. Ghost updates (house exit, fright timer, eaten check, AI move)
        4. Pellet collection + power pellet frightened activation
        5. Ghost collision detection + handling
        6. Proximity reward + ghost proximity penalty
        7. Mode timer update + level completion check
        """
        self.reward_pacman.zero_()
        self.reward_ghost.zero_()
        self.step_count += 1

        # Skip already-done environments
        done = self.game_over | self.level_complete

        # Ready countdown — no movement during ready
        ready = self.ready_timer > 0
        self.ready_timer = torch.where(ready & ~done, self.ready_timer - 1, self.ready_timer)

        # Active environments: not done and not in ready countdown
        active = ~done & ~ready

        if not active.any():
            return self.reward_pacman, done, {}

        # --- Pre-movement ---

        # Per-step time penalty
        self.reward_pacman = torch.where(active, self.reward_pacman - 0.05, self.reward_pacman)

        # Visit map decay + penalty (before movement, matching original order)
        ghosts_active = (~self.ghost_in_house).any(dim=1)  # (N,) — any ghost out of house
        visit_penalty, self.visit_map = update_visit_map_and_penalty(
            self.visit_map, self.pacman_pos, self.pellets, self.power_pellets,
            self.maze.total_pellets, ghosts_active, active,
        )
        self.reward_pacman = self.reward_pacman + visit_penalty

        # Direction-change penalty: soft anti-oscillation (-0.03 for changing action)
        has_prev = self.prev_action >= 0
        changed = actions.int() != self.prev_action
        dir_change_penalty = active & has_prev & changed
        self.reward_pacman = torch.where(dir_change_penalty, self.reward_pacman - 0.03, self.reward_pacman)
        self.prev_action = torch.where(active, actions.int(), self.prev_action)

        # Convert actions to direction vectors: (N, 2)
        self.pacman_next_dir = self._action_dirs[actions.long()]
        self.pacman_next_dir = torch.where(active.unsqueeze(1), self.pacman_next_dir,
                                           torch.zeros_like(self.pacman_next_dir))

        # --- Save previous positions (for swap collision detection) ---
        self.prev_pac_pos = self.pacman_pos.clone()
        prev_ghost_pos = self.ghost_pos.clone()

        # --- Move Pacman ---
        new_pos, new_dir = update_pacman(
            self.pacman_pos.clone(),
            self.pacman_dir.clone(),
            self.pacman_next_dir,
            self.maze.pacman_blocked,
            self.maze.tunnel_y,
        )
        self.pacman_pos = torch.where(active.unsqueeze(1), new_pos, self.pacman_pos)
        self.pacman_dir = torch.where(active.unsqueeze(1), new_dir, self.pacman_dir)

        # --- Ghost updates ---
        # Save in-house state BEFORE house step — ghosts that exit this step
        # should NOT have their AI move timer incremented (matches original
        # ghost.update() which returns after exiting the house).
        was_in_house = self.ghost_in_house.clone()

        # 1. Ghost house exit (for in-house ghosts)
        self.ghost_pos, self.ghost_in_house, self.ghost_exit_timer = ghost_house_step(
            self.ghost_pos, self.ghost_in_house, self.ghost_exit_timer,
            self._door_pos, active, self.maze.walls,
        )

        # 2. Frightened timer (for non-in-house ghosts, excluding just-exited)
        self.ghost_state, self.ghost_fright_timer = frightened_timer_step(
            self.ghost_state, self.ghost_prev_state, self.ghost_fright_timer, active,
        )

        # 3. Eaten ghost check (reached ghost house target → re-enter)
        self.ghost_state, self.ghost_prev_state, self.ghost_in_house, self.ghost_exit_timer = \
            eaten_ghost_check(
                self.ghost_pos, self.ghost_state, self.ghost_prev_state,
                self.ghost_in_house, self.ghost_exit_timer, active,
            )

        # 4. Ghost AI movement (with speed timer)
        # Use was_in_house to exclude ghosts that just exited the house this step
        self.ghost_pos, self.ghost_dir, self.ghost_move_timer = ghost_ai_move(
            self.ghost_pos, self.ghost_dir, self.ghost_state, was_in_house,
            self.ghost_move_timer, self.ghost_speed,
            self.pacman_pos, self.pacman_dir,
            self.maze.walls, self.maze.ghost_house_door_mask,
            self.maze.tunnel_y, self._scatter_targets, active,
        )

        # --- Pellet collection ---
        pellet_reward, ate_pellet, ate_power = collect_pellets(
            self.pacman_pos, self.pellets, self.power_pellets,
            self.maze.total_pellets, active,
        )
        self.reward_pacman = self.reward_pacman + pellet_reward

        # Ghost reward for pellets
        self.reward_ghost = torch.where(ate_pellet, self.reward_ghost - 0.1, self.reward_ghost)
        self.reward_ghost = torch.where(ate_power, self.reward_ghost - 0.5, self.reward_ghost)

        # Update score
        self.score = self.score + (PELLET_SCORE * ate_pellet.int()) + (POWER_PELLET_SCORE * ate_power.int())

        # --- Power pellet → activate frightened ---
        (self.ghost_state, self.ghost_prev_state, self.ghost_dir,
         self.ghost_fright_timer, self.ghost_eat_combo) = activate_frightened(
            ate_power, self.ghost_state, self.ghost_prev_state,
            self.ghost_dir, self.ghost_fright_timer, self.ghost_eat_combo,
            self.ghost_in_house,
        )

        # --- Ghost collision detection ---
        collision = check_ghost_collisions(
            self.pacman_pos, self.prev_pac_pos,
            self.ghost_pos, prev_ghost_pos,
            self.ghost_state, self.ghost_in_house, active,
        )

        # Handle ghost collisions
        (self.ghost_state, self.ghost_eat_combo, self.score,
         self.reward_pacman, self.reward_ghost, self.game_over) = handle_ghost_collisions(
            collision, self.ghost_state, self.ghost_prev_state,
            self.ghost_in_house, self.ghost_exit_timer, self.ghost_pos,
            self.ghost_eat_combo, self.score,
            self.reward_pacman, self.reward_ghost, self.game_over,
        )

        # --- Proximity reward (uses precomputed distances) ---
        prox_reward, self.prev_nearest_dist = proximity_reward(
            self.pacman_pos, self.pellets, self.power_pellets,
            self.maze.total_pellets, self.prev_nearest_dist,
            self.dist_matrix, self.maze.tile_to_idx, active,
            self.game_over, self.level_complete,
        )
        self.reward_pacman = self.reward_pacman + prox_reward

        # --- Ghost proximity penalty ---
        ghost_is_frightened = self.ghost_state == FRIGHTENED
        ghost_is_eaten = self.ghost_state == EATEN
        ghost_prox = ghost_proximity_penalty(
            self.pacman_pos, self.ghost_pos,
            self.ghost_in_house, ghost_is_frightened, ghost_is_eaten,
            active, self.game_over, self.level_complete,
        )
        self.reward_pacman = self.reward_pacman + ghost_prox

        # --- Update ghost mode timer ---
        self.mode_timer, self.current_mode, self.ghost_state = update_ghost_mode(
            self.mode_timer, self.current_mode, self.ghost_state, active,
        )

        # --- Level completion check ---
        completed, completion_reward = check_level_complete(
            self.pellets, self.power_pellets, self.step_count, MAX_STEPS,
        )
        newly_complete = completed & ~self.level_complete & active
        self.level_complete = self.level_complete | newly_complete
        self.reward_pacman = self.reward_pacman + completion_reward * newly_complete.float()
        self.reward_ghost = torch.where(newly_complete, self.reward_ghost - 5.0, self.reward_ghost)

        dones = self.game_over | self.level_complete
        return self.reward_pacman, dones, {"ate_pellet": ate_pellet, "ate_power": ate_power}

    def reset(self, mask: torch.Tensor | None = None) -> None:
        """Reset specified environments (preserves stage config)."""
        if mask is None:
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)

        if not mask.any():
            return

        dev = self.device
        sx, sy = self.maze.pacman_start
        start = torch.tensor([sx, sy], dtype=torch.int32, device=dev)
        zero2 = torch.zeros(2, dtype=torch.int32, device=dev)

        self.pacman_pos = torch.where(mask.unsqueeze(1), start.unsqueeze(0), self.pacman_pos)
        self.pacman_dir = torch.where(mask.unsqueeze(1), zero2.unsqueeze(0), self.pacman_dir)
        self.pacman_next_dir = torch.where(mask.unsqueeze(1), zero2.unsqueeze(0), self.pacman_next_dir)
        self.step_count = torch.where(mask, torch.zeros_like(self.step_count), self.step_count)
        self.ready_timer = torch.where(mask, torch.full_like(self.ready_timer, READY_TIMER), self.ready_timer)
        self.game_over = torch.where(mask, torch.zeros_like(self.game_over), self.game_over)
        self.level_complete = torch.where(mask, torch.zeros_like(self.level_complete), self.level_complete)
        self.prev_nearest_dist = torch.where(mask, torch.zeros_like(self.prev_nearest_dist), self.prev_nearest_dist)
        self.prev_action = torch.where(mask, torch.full_like(self.prev_action, -1), self.prev_action)

        # Reset pellets, visit map, ghost state for masked envs
        for i in range(self.n):
            if mask[i]:
                self.pellets[i] = self.maze.initial_pellets.clone()
                self.power_pellets[i] = self.maze.initial_power_pellets.clone()
                self.score[i] = 0
                self.visit_map[i].zero_()
                # Ghost state
                self.ghost_pos[i] = self.maze.ghost_starts.clone()
                self.ghost_dir[i].zero_()
                self.ghost_in_house[i] = True
                self.ghost_state[i] = SCATTER
                self.ghost_prev_state[i] = SCATTER
                self.ghost_exit_timer[i] = self._stage_exit_timers.clone()
                self.ghost_fright_timer[i] = 0
                self.ghost_move_timer[i] = 0
                self.ghost_speed[i] = self._stage_speeds.clone()
                self.ghost_eat_combo[i] = 0
                self.mode_timer[i] = 0
                self.current_mode[i] = SCATTER

    def get_state(self) -> torch.Tensor:
        """Return observation tensors for all N environments.

        Shape: (N, 7, ROWS, COLS) float32.
        Channels: walls, pellets, power_pellets, pacman, ghosts, frightened, visited.

        Uses a pre-allocated buffer to avoid allocation overhead.
        """
        N = self.n
        dev = self.device
        state = self._state_bufs[self._state_buf_idx]
        self._state_buf_idx = 1 - self._state_buf_idx

        # 0: Walls (static, pre-converted to float)
        state[:, CHANNEL_WALLS] = self._walls_float

        # 1-2: Pellets and power pellets (bool -> float via view)
        state[:, CHANNEL_PELLETS] = self.pellets
        state[:, CHANNEL_POWER_PELLETS] = self.power_pellets

        # 3: Pacman position (clear then scatter)
        state[:, CHANNEL_PACMAN] = 0
        env_idx = torch.arange(N, device=dev)
        px = self.pacman_pos[:, 0].long()
        py = self.pacman_pos[:, 1].long()
        state[env_idx, CHANNEL_PACMAN, py, px] = 1.0

        # 4-5: Ghost channels (clear then scatter)
        state[:, CHANNEL_GHOSTS] = 0
        state[:, CHANNEL_FRIGHTENED] = 0
        for gi in range(4):
            gx = self.ghost_pos[:, gi, 0].long()
            gy = self.ghost_pos[:, gi, 1].long()
            gs = self.ghost_state[:, gi]
            in_house = self.ghost_in_house[:, gi]

            frightened = (gs == FRIGHTENED) & ~in_house
            normal = (gs != FRIGHTENED) & (gs != EATEN) & ~in_house

            if frightened.any():
                state[env_idx[frightened], CHANNEL_FRIGHTENED, gy[frightened], gx[frightened]] = 1.0
            if normal.any():
                state[env_idx[normal], CHANNEL_GHOSTS, gy[normal], gx[normal]] = 1.0

        # 6: Visit heatmap
        state[:, CHANNEL_VISITED] = self.visit_map

        return state

    def get_action_mask(self, no_reverse: bool = False) -> torch.Tensor:
        """Return valid action mask for all N environments. (N, 4) bool.

        Args:
            no_reverse: If True, mask the reverse of Pacman's current direction
                (proximity-based anti-oscillation). Reverse is allowed when any
                non-house ghost is within Manhattan distance 4 of that env's
                Pacman, so the agent can flee.
        """
        return get_action_mask(
            self.pacman_pos, self.pacman_dir,
            self.maze.pacman_blocked, self.maze.tunnel_y,
            no_reverse=no_reverse,
            ghost_pos=self.ghost_pos,
            ghost_in_house=self.ghost_in_house,
        )

    def get_pacman_positions(self) -> torch.Tensor:
        return self.pacman_pos

    def get_reward_pacman(self) -> torch.Tensor:
        return self.reward_pacman

    def get_reward_ghost(self) -> torch.Tensor:
        return self.reward_ghost
