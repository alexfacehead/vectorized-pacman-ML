"""Visual test: render the vectorized engine through the original pygame renderer.

Usage:
    python watch.py              # Watch with random actions
    python watch.py --side       # Side-by-side comparison with original engine
    python watch.py --keyboard   # Control Pacman with arrow keys

Controls:
    Arrow keys: move Pacman (in --keyboard mode)
    ESC: quit
    R: reset
    SPACE: pause/unpause
    +/-: speed up/slow down (1-60 FPS)
"""

import sys
import os
import argparse
import numpy as np
import torch

# Add original PacmanML to path (append, not insert — this repo's
# packages like training/ and engine/ must take priority)
PACMANML_DIR = os.path.expanduser("~/dev/PacmanML")
sys.path.append(PACMANML_DIR)

from pacman.core.renderer import Renderer
from pacman.core.level import Level
from pacman.utils.constants import (
    GRID_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, FPS, SCATTER,
    FRIGHTENED, EATEN, BLUE, WHITE,
)

from engine.constants import GHOST_NAMES

# Ghost base colors matching original
GHOST_BASE_COLORS = {
    "blinky": (255, 0, 0),
    "pinky": (255, 184, 255),
    "inky": (0, 255, 255),
    "clyde": (255, 184, 82),
}

MAZE_FILE = os.path.join(os.path.dirname(__file__), "levels", "level_1.txt")
ORIGINAL_MAZE_FILE = os.path.join(PACMANML_DIR, "pacman", "levels", "level_1.txt")


# ---------------------------------------------------------------------------
# Adapter objects — make our tensor state look like original engine objects
# ---------------------------------------------------------------------------

class PacmanAdapter:
    """Wraps vectorized Pacman state for the renderer."""

    def __init__(self):
        self.position = (0, 0)
        self.direction = (0, 0)
        self.mouth_angle = 30
        self.mouth_opening = True
        self.frame_counter = 0
        self.lives = 1
        self.powered_up = False

    def update_from_batch(self, batched_game, env_idx=0):
        pos = batched_game.pacman_pos[env_idx].tolist()
        self.position = tuple(pos)
        d = batched_game.pacman_dir[env_idx].tolist()
        self.direction = tuple(d)
        # Animate mouth
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            if self.mouth_opening:
                self.mouth_angle += 5
                if self.mouth_angle >= 45:
                    self.mouth_opening = False
            else:
                self.mouth_angle -= 5
                if self.mouth_angle <= 5:
                    self.mouth_opening = True


class GhostAdapter:
    """Wraps vectorized ghost state for the renderer."""

    def __init__(self, ghost_index: int):
        self.index = ghost_index
        self.name = GHOST_NAMES[ghost_index]
        self.base_color = GHOST_BASE_COLORS[self.name]
        self.position = (0, 0)
        self.direction = (0, 0)
        self.state = SCATTER
        self.in_ghost_house = True

    @property
    def is_eaten(self) -> bool:
        return self.state == EATEN

    @property
    def is_frightened(self) -> bool:
        return self.state == FRIGHTENED

    @property
    def color(self):
        if self.state == FRIGHTENED:
            return BLUE
        if self.state == EATEN:
            return WHITE
        return self.base_color

    def update_from_batch(self, batched_game, env_idx=0):
        pos = batched_game.ghost_pos[env_idx, self.index].tolist()
        self.position = tuple(pos)
        d = batched_game.ghost_dir[env_idx, self.index].tolist()
        self.direction = tuple(d)
        self.state = batched_game.ghost_state[env_idx, self.index].item()
        self.in_ghost_house = batched_game.ghost_in_house[env_idx, self.index].item()


class LevelAdapter:
    """Wraps vectorized maze + pellet state for the renderer."""

    def __init__(self, original_level, batched_game, env_idx=0):
        self._original = original_level
        self._batched = batched_game
        self._env_idx = env_idx

    @property
    def height(self):
        return self._original.height

    @property
    def width(self):
        return self._original.width

    @property
    def grid(self):
        return self._original.grid

    @property
    def pellets(self):
        """Return pellet-like objects matching uneaten state from tensors."""
        return self._build_pellet_list(
            self._batched.pellets[self._env_idx],
            self._original.pellets,
        )

    @property
    def power_pellets(self):
        return self._build_pellet_list(
            self._batched.power_pellets[self._env_idx],
            self._original.power_pellets,
        )

    def _build_pellet_list(self, tensor_state, original_pellets):
        """Sync original pellet objects' eaten state with tensor state."""
        for p in original_pellets:
            x, y = p.position
            p.eaten = not tensor_state[y, x].item()
        return original_pellets


def run_visual(args):
    import pygame

    pygame.init()

    if args.side:
        # Side-by-side: double width
        screen = pygame.display.set_mode((SCREEN_WIDTH * 2 + 4, SCREEN_HEIGHT))
        pygame.display.set_caption("Vectorized vs Original — Side-by-Side")
    else:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Vectorized Pac-Man Engine")

    clock = pygame.time.Clock()

    # ---- Vectorized engine ----
    from engine.batched_game import BatchedGame
    batched = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
    batched.configure_stage(args.stage)
    if args.ghost_speed is not None:
        batched.ghost_speed[:] = args.ghost_speed

    # ---- Load trained model (optional) ----
    agent = None
    if args.model:
        from training.train import BatchedPacmanAgent
        from engine.constants import NUM_STATE_CHANNELS
        agent = BatchedPacmanAgent(in_channels=NUM_STATE_CHANNELS, device="cpu")
        agent.load(args.model)
        agent.epsilon = 0.01  # tiny noise to avoid wall-staring
        print(f"Loaded model from {args.model} (epsilon={agent.epsilon:.2f})")

    # LSTM hidden state for DRQN agent
    hidden = None
    if agent is not None:
        hidden = agent.init_hidden(1)

    if args.ghosts:
        for i in range(4):
            batched.ghost_exit_timer[0, i] = i * 300  # Staggered exits
    pacman_adapter = PacmanAdapter()
    ghost_adapters = [GhostAdapter(i) for i in range(4)]

    # We need an original Level for the renderer (grid data, pellet objects)
    orig_level = Level(ORIGINAL_MAZE_FILE)
    level_adapter = LevelAdapter(orig_level, batched, env_idx=0)

    # ---- Original engine (for side-by-side) ----
    orig_game = None
    if args.side:
        from pacman.core.game import Game as OriginalGame
        orig_game = OriginalGame(headless=True)
        orig_game.load_level(ORIGINAL_MAZE_FILE)
        orig_game.reset()

    # ---- Renderers ----
    renderer_vec = Renderer()
    surf_vec = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    renderer_vec.init_display(surf_vec)

    renderer_orig = None
    surf_orig = None
    if args.side:
        renderer_orig = Renderer()
        surf_orig = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        renderer_orig.init_display(surf_orig)

    # ---- State ----
    np.random.seed(42)
    running = True
    paused = False
    pending_action = None
    step_count = 0
    playback_fps = args.fps

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    batched.reset()
                    batched.configure_stage(args.stage)
                    if args.ghost_speed is not None:
                        batched.ghost_speed[:] = args.ghost_speed
                    orig_level = Level(ORIGINAL_MAZE_FILE)
                    level_adapter = LevelAdapter(orig_level, batched, 0)
                    if orig_game:
                        orig_game.reset()
                    if agent is not None:
                        hidden = agent.init_hidden(1)
                    step_count = 0
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    playback_fps = min(60, playback_fps + 5)
                    print(f"Speed: {playback_fps} FPS")
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    playback_fps = max(1, playback_fps - 5)
                    print(f"Speed: {playback_fps} FPS")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif args.keyboard:
                    key_map = {
                        pygame.K_UP: 0,
                        pygame.K_DOWN: 1,
                        pygame.K_LEFT: 2,
                        pygame.K_RIGHT: 3,
                    }
                    if event.key in key_map:
                        pending_action = key_map[event.key]

        if paused:
            clock.tick(FPS)
            continue

        # Pick action
        if args.keyboard:
            action = pending_action if pending_action is not None else 0
        elif agent is not None:
            state = batched.get_state()
            action_mask = batched.get_action_mask(no_reverse=True)
            actions, hidden = agent.select_actions_batched(state, action_mask, hidden)
            action = actions[0].item()
        else:
            action = np.random.randint(0, 4)

        # Step vectorized engine
        done_before = (batched.game_over | batched.level_complete)[0].item()
        if not done_before:
            rewards, dones, infos = batched.step(torch.tensor([action]))
            step_count += 1

        # Step original engine (side-by-side)
        if orig_game and not (orig_game.game_over or orig_game.level_complete):
            orig_game.step(pacman_action=action)

        # Update adapters
        pacman_adapter.update_from_batch(batched)
        for ga in ghost_adapters:
            ga.update_from_batch(batched)

        # Check parity (side-by-side)
        if orig_game and not done_before:
            vec_pos = batched.pacman_pos[0].tolist()
            orig_pos = list(orig_game.pacman.position)
            if vec_pos != orig_pos:
                print(f"DIVERGENCE at step {step_count}: vec={vec_pos} orig={orig_pos}")

        # ---- Render vectorized ----
        score = batched.score[0].item()
        # Filter: only show ghosts that are out of the house
        visible_ghosts = [ga for ga in ghost_adapters if not ga.in_ghost_house]
        renderer_vec.draw(
            level_adapter, pacman_adapter, visible_ghosts,
            score, 1,
            game_over=batched.game_over[0].item(),
            level_complete=batched.level_complete[0].item(),
            ready=(batched.ready_timer[0].item() > 0),
            mode=batched.current_mode[0].item(),
        )

        if args.side:
            # ---- Render original ----
            renderer_orig.draw(
                orig_game.level, orig_game.pacman, orig_game.ghosts,
                orig_game.score, orig_game.pacman.lives,
                game_over=orig_game.game_over,
                level_complete=orig_game.level_complete,
                ready=(orig_game.ready_timer > 0),
                mode=orig_game.current_mode,
            )

            # Composite
            screen.fill((40, 40, 40))
            screen.blit(surf_vec, (0, 0))
            screen.blit(surf_orig, (SCREEN_WIDTH + 4, 0))

            # Labels
            font = pygame.font.Font(None, 24)
            lbl_vec = font.render("VECTORIZED", True, (0, 255, 0))
            lbl_orig = font.render("ORIGINAL", True, (255, 255, 0))
            screen.blit(lbl_vec, (SCREEN_WIDTH // 2 - lbl_vec.get_width() // 2, 2))
            screen.blit(lbl_orig, (SCREEN_WIDTH + 4 + SCREEN_WIDTH // 2 - lbl_orig.get_width() // 2, 2))
        else:
            screen.blit(surf_vec, (0, 0))

            # Step counter
            font = pygame.font.Font(None, 20)
            info = font.render(
                f"Step: {step_count}  Score: {score}  "
                f"Pellets: {int(batched.pellets[0].sum().item() + batched.power_pellets[0].sum().item())}",
                True, (200, 200, 200),
            )
            screen.blit(info, (8, 2))

        pygame.display.flip()
        clock.tick(playback_fps)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual test for vectorized Pac-Man engine")
    parser.add_argument("--side", action="store_true", help="Side-by-side comparison with original")
    parser.add_argument("--keyboard", action="store_true", help="Control Pacman with arrow keys")
    parser.add_argument("--ghosts", action="store_true", help="Release all ghosts with staggered exits")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Curriculum stage (1=no ghosts, 2=Blinky slow, 3=Blinky fast, 4=B+P slow, 5=3 ghosts slow, 6=B slow+P fast, 7=3 ghosts mixed)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Playback speed in FPS (default: 15, use +/- keys to adjust live)")
    parser.add_argument("--ghost-speed", type=int, default=None,
                        help="Override ghost speed (1=full, 2=half, 3=third). Overrides stage default.")
    args = parser.parse_args()
    run_visual(args)
