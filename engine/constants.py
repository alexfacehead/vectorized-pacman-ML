"""Constants for the vectorized Pac-Man engine.

All timer values are pre-scaled for headless mode (divided by PACMAN_SPEED=8).
In headless, every step() is one tile move — no frame-counting needed.
"""

import torch

# Grid dimensions
COLS = 28
ROWS = 31

# Original frame-based speeds (used only for timer scaling reference)
PACMAN_SPEED = 8  # original frames per move
GHOST_SPEED = 12  # original frames per move

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Direction vectors: (dx, dy) for each action index
# Shape: (4, 2) — index with action integer
ACTION_DIRS = torch.tensor([
    [0, -1],   # UP
    [0, 1],    # DOWN
    [-1, 0],   # LEFT
    [1, 0],    # RIGHT
], dtype=torch.int32)

# Ghost states
SCATTER = 0
CHASE = 1
FRIGHTENED = 2
EATEN = 3

# Timer durations — headless scaled (original_frames // PACMAN_SPEED)
SCATTER_DURATION = 420 // PACMAN_SPEED     # 52
CHASE_DURATION = 1200 // PACMAN_SPEED      # 150
FRIGHTENED_DURATION = 360 // PACMAN_SPEED  # 45
POWERUP_DURATION = 360 // PACMAN_SPEED     # 45
FRIGHTENED_FLASH_TIME = 90 // PACMAN_SPEED # 11
READY_TIMER = 120 // PACMAN_SPEED          # 15

# Scoring
PELLET_SCORE = 10
POWER_PELLET_SCORE = 50
GHOST_SCORE = 200

# Ghost house positions
GHOST_HOUSE_TARGET = (13, 14)  # center of ghost house
GHOST_HOUSE_DOOR = (13, 12)    # door position

# Ghost names and scatter targets (corners)
GHOST_NAMES = ["blinky", "pinky", "inky", "clyde"]
GHOST_SCATTER_TARGETS = torch.tensor([
    [25, 0],   # blinky
    [2, 0],    # pinky
    [27, 30],  # inky
    [0, 30],   # clyde
], dtype=torch.int32)

# Episode limits
MAX_STEPS = 3500

# State channels
CHANNEL_WALLS = 0
CHANNEL_PELLETS = 1
CHANNEL_POWER_PELLETS = 2
CHANNEL_PACMAN = 3
CHANNEL_GHOSTS = 4
CHANNEL_FRIGHTENED = 5
CHANNEL_VISITED = 6
NUM_STATE_CHANNELS = 7
