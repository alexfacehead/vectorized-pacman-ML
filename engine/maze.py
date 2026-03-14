"""Static maze loading — parses level_1.txt into tensors.

The Maze object is created once and shared across all N environments.
It holds only static, immutable data about the maze layout.
"""

import os
from typing import List, Tuple, Dict

import torch

from .constants import ROWS, COLS


class Maze:
    """Immutable maze data parsed from a level text file."""

    def __init__(self, level_file: str):
        grid = self._load_grid(level_file)

        # --- Static boolean grids ---
        self.walls = torch.zeros(ROWS, COLS, dtype=torch.bool)
        self.ghost_house_door_mask = torch.zeros(ROWS, COLS, dtype=torch.bool)
        self.initial_pellets = torch.zeros(ROWS, COLS, dtype=torch.bool)
        self.initial_power_pellets = torch.zeros(ROWS, COLS, dtype=torch.bool)

        # --- Positions ---
        self.pacman_start: Tuple[int, int] = (0, 0)
        ghost_starts: List[Tuple[int, int]] = []

        # --- Parse grid ---
        for y in range(ROWS):
            for x in range(COLS):
                ch = grid[y][x]
                if ch == "#":
                    self.walls[y, x] = True
                elif ch == "-":
                    self.ghost_house_door_mask[y, x] = True
                elif ch == ".":
                    self.initial_pellets[y, x] = True
                elif ch == "o":
                    self.initial_power_pellets[y, x] = True
                elif ch == "P":
                    self.pacman_start = (x, y)
                elif ch == "G":
                    ghost_starts.append((x, y))

        self.ghost_starts = torch.tensor(ghost_starts[:4], dtype=torch.int32)  # (4, 2) x,y

        # --- Tunnel row ---
        self.tunnel_y: int = -1
        for y in range(ROWS):
            if grid[y][0] == " " and grid[y][-1] == " ":
                self.tunnel_y = y
                break

        # Ghost house door position (single cell)
        self.ghost_house_door_pos = torch.tensor(
            [self.ghost_house_door_mask.nonzero()[0, 1].item(),
             self.ghost_house_door_mask.nonzero()[0, 0].item()],
            dtype=torch.int32,
        )  # (2,) as (x, y)

        # --- Blocked mask for Pacman: walls OR ghost house door ---
        self.pacman_blocked = self.walls | self.ghost_house_door_mask

        # --- Walkable tiles (for distance matrix) ---
        # Everything that isn't a wall is walkable (including door, spaces, pellet positions)
        walkable_yx = (~self.walls).nonzero()  # (num_walkable, 2) as (y, x)
        self.num_walkable = walkable_yx.shape[0]

        # tile_to_idx: (ROWS, COLS) int — maps (y, x) to tile index, -1 for walls
        self.tile_to_idx = torch.full((ROWS, COLS), -1, dtype=torch.int32)
        for i in range(self.num_walkable):
            y, x = walkable_yx[i, 0].item(), walkable_yx[i, 1].item()
            self.tile_to_idx[y, x] = i

        # idx_to_tile: (num_walkable, 2) as (x, y)
        self.idx_to_tile = torch.stack(
            [walkable_yx[:, 1], walkable_yx[:, 0]], dim=1
        ).to(torch.int32)  # (num_walkable, 2) as (x, y)

        # Total pellet count at start
        self.total_pellets = int(self.initial_pellets.sum().item() + self.initial_power_pellets.sum().item())

    def _load_grid(self, level_file: str) -> List[List[str]]:
        """Load grid from text file, pad rows to COLS width."""
        if not os.path.isabs(level_file):
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            level_file = os.path.join(pkg_dir, level_file)
        with open(level_file) as f:
            grid = [list(line.rstrip("\n")) for line in f.readlines()]
        # Pad rows
        for row in grid:
            while len(row) < COLS:
                row.append(" ")
        return grid

    def to(self, device: torch.device) -> "Maze":
        """Move all tensors to the given device. Returns self for chaining."""
        self.walls = self.walls.to(device)
        self.ghost_house_door_mask = self.ghost_house_door_mask.to(device)
        self.initial_pellets = self.initial_pellets.to(device)
        self.initial_power_pellets = self.initial_power_pellets.to(device)
        self.ghost_starts = self.ghost_starts.to(device)
        self.ghost_house_door_pos = self.ghost_house_door_pos.to(device)
        self.pacman_blocked = self.pacman_blocked.to(device)
        self.tile_to_idx = self.tile_to_idx.to(device)
        self.idx_to_tile = self.idx_to_tile.to(device)
        return self
