import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 2

PLAYER_SIZE = (16, 11)
PLAYER_START_X = 129
PLAYER_START_Y = 61

class PooyanState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    arrow_x: chex.Array
    arrow_y: chex.Array
    step_counter: chex.Array

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class PooyanObservation(NamedTuple):
    player: EntityPosition

class PooyanInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

def get_human_action() -> chex.Array:
    """
    Records if UP or DOWN is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        return jnp.array(Action.UP)
    elif keys[pygame.K_s]:
        return jnp.array(Action.DOWN)
    elif keys[pygame.K_SPACE]:
        return jnp.array(Action.FIRE)
    else:
        return jnp.array(Action.NOOP)

def load_sprites():
    """Load all sprites required for Pong rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load sprites
    bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pooyan/background.npy"))
    # print(numpy.shape(bg))
    # background = numpy.array([numpy.array([numpy.array([0, 28, 136, 255]) for j in range(160)]) for i in range(202)])
    # numpy.save('background.npy', background)
    walls = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pooyan/walls.npy"))
    floor = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pooyan/floor.npy"))
    roof = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pooyan/roof.npy"))
    player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pooyan/player.npy"))
    arrow = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/pooyan/player.npy"))

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(bg, axis=0)
    SPRITE_WALLS = jnp.expand_dims(walls, axis=0)
    SPRITE_FLOOR = jnp.expand_dims(floor, axis=0)
    SPRITE_ROOF = jnp.expand_dims(roof, axis=0)
    SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
    SPRITE_ARROW = jnp.expand_dims(arrow, axis=0)

    # Load digits for scores
    return (
        SPRITE_BG,
        SPRITE_WALLS,
        SPRITE_FLOOR,
        SPRITE_ROOF,
        SPRITE_PLAYER,
        SPRITE_ARROW
    )

class PooyanRenderer(JAXGameRenderer):
    """JAX-based Pong game renderer, optimized with JIT compilation."""

    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_WALLS,
            self.SPRITE_FLOOR,
            self.SPRITE_ROOF,
            self.SPRITE_PLAYER,
            self.SPRITE_ARROW
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A PooyanState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """

        raster = jnp.zeros((HEIGHT, WIDTH, 3))
        
        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)
        frame_walls = jr.get_sprite_frame(self.SPRITE_WALLS, 0)
        raster = jr.render_at(raster, 0, 57, frame_walls)
        frame_roof = jr.get_sprite_frame(self.SPRITE_ROOF, 0)
        raster = jr.render_at(raster, 0, 37, frame_roof)
        frame_floor = jr.get_sprite_frame(self.SPRITE_FLOOR, 0)
        raster = jr.render_at(raster, 0, 202, frame_floor)

        frame_player = jr.get_sprite_frame(self.SPRITE_PLAYER, 0)
        raster = jr.render_at(raster, state.player_x, state.player_y, frame_player)

        should_render = state.arrow_x > 0
        raster = jax.lax.cond(
            should_render,
            lambda r: jr.render_at(
                r,
                state.arrow_x,
                state.arrow_y,
                frame_player,
            ),
            lambda r: r,
            raster,
        )

        return raster

@jax.jit
def player_step(state, action: chex.Array):
    should_move = state.step_counter % 16 == 0
    print(state.step_counter, should_move)
    up = jnp.any(
            jnp.array(
                [
                    action == Action.UP,
                ]
            )
        )
    down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                ]
            )
        )
    player_y = jax.lax.cond(
            should_move,
            lambda _: jnp.where(
                down, state.player_y + 21, jnp.where(up, state.player_y - 21, state.player_y)
            ),
            lambda _: state.player_y,
            operand=None
        )
    # player_y = jnp.where(
    #         should_move,
    #         jnp.where(
    #             down, state.player_y + 21, jnp.where(up, state.player_y - 21, state.player_y)
    #         ),
    #         state.player_y
        # )
    # if should_move:
    #     player_y = 
    # else:
    #     player_y = state.player_y
    # player_y = jnp.where(
            #     down, state.player_y + 1, jnp.where(up, state.player_y - 1, state.player_y)
            # )
    
    return PooyanState(
        player_x=jnp.array(state.player_x).astype(jnp.int32),
        player_y=jnp.array(player_y).astype(jnp.int32),
        step_counter=state.step_counter,
        arrow_x=state.arrow_x,
        arrow_y=state.arrow_y
    )

@jax.jit
def arrow_step(state, action: chex.Array):
    could_fire = state.arrow_x <= 10
    fire = jnp.any(
            jnp.array(
                [
                    action == Action.FIRE,
                ]
            )
        )
    
    arrow_x = jax.lax.cond(
            could_fire,
            lambda _: jnp.where(
                fire, state.player_x, 0
            ),
            lambda _: state.arrow_x - 4,
            operand=None
        )
    arrow_y = jax.lax.cond(
            could_fire,
            lambda _: jnp.where(
                fire, state.player_y, 0
            ),
            lambda _: state.arrow_y,
            operand=None
        )
    
    return PooyanState(
        player_x=state.player_x,
        player_y=state.player_y,
        step_counter=state.step_counter,
        arrow_x=arrow_x,
        arrow_y=arrow_y
    )

class PooyanConstants(NamedTuple):
    MAX_SPEED: int = 12
    ENEMY_STEP_SIZE: int = 2
    WIDTH: int = 160
    HEIGHT: int = 210
    PLAYER_ACCELERATION: chex.Array = jnp.array([6, 3, 1, -1, 1, -1, 0, 0, 1, 0, -1, 0, 1])
    BACKGROUND_COLOR: Tuple[int, int, int] = (144, 72, 17)
    PLAYER_COLOR: Tuple[int, int, int] = (92, 186, 92)
    ENEMY_COLOR: Tuple[int, int, int] = (213, 130, 74)
    WALL_COLOR: Tuple[int, int, int] = (236, 236, 236)
    SCORE_COLOR: Tuple[int, int, int] = (236, 236, 236)
    PLAYER_X: int = 140
    ENEMY_X: int = 16
    PLAYER_SIZE: Tuple[int, int] = (4, 16)
    ENEMY_SIZE: Tuple[int, int] = (4, 16)
    WALL_TOP_Y: int = 24
    WALL_TOP_HEIGHT: int = 10
    WALL_BOTTOM_Y: int = 194
    WALL_BOTTOM_HEIGHT: int = 16

class JaxPooyan(JaxEnvironment[PooyanState, PooyanObservation, PooyanInfo, PooyanConstants]):
    def __init__(self, reward_funcs: list[callable]=None):
        super().__init__()
        self.action_set = {
            Action.UP,
            Action.DOWN,
            Action.FIRE,
            Action.NOOP,
        }


    def reset(self, key=None) -> Tuple[PooyanObservation, PooyanState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        state = PooyanState(
            player_x=jnp.array(129).astype(jnp.int32),
            player_y=jnp.array(61).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            arrow_x=jnp.array(0).astype(jnp.int32),
            arrow_y=jnp.array(0).astype(jnp.int32)
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PooyanState, action: chex.Array) -> Tuple[PooyanObservation, PooyanState, float, bool, PooyanInfo]:
        # Step 1: Update player position and speed
        # only execute player step on even steps (base implementation only moves the player every second tick)
        new_player = player_step(
            state, action
        )

        new_arrow = arrow_step(
            state, action
        )

        step_counter = jax.lax.cond(
            False,
            lambda s: jnp.array(0),
            lambda s: s + 1,
            operand=state.step_counter,
        )

        new_state = PooyanState(
            player_x=new_player.player_x,
            player_y=new_player.player_y,
            step_counter=step_counter,
            arrow_x=new_arrow.arrow_x,
            arrow_y=new_arrow.arrow_y
        )

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PooyanState):
        # create player
        player = EntityPosition(
            x=jnp.array(state.player_x),
            y=state.player_y,
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
        )

        return PooyanObservation(
            player=player,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: PooyanObservation) -> jnp.ndarray:
           return jnp.concatenate([
               obs.player.x.flatten(),
               obs.player.y.flatten(),
               obs.player.height.flatten(),
               obs.player.width.flatten(),
            ]
           )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=jnp.uint8,
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PooyanState, all_rewards: chex.Array) -> PooyanInfo:
        return PooyanInfo(time=0, all_rewards=all_rewards)
        # return PooyanInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: PooyanState, state: PooyanState):
        return 0
        # return (state.player_score - state.enemy_score) - (
        #     previous_state.player_score - previous_state.enemy_score
        # )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: PooyanState, state: PooyanState):
        # if self.reward_funcs is None:
        #     return jnp.zeros(1)
        # rewards = jnp.array(
        #     [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        # )
        return 0#rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PooyanState) -> bool:
        return False
        # return jnp.logical_or(
        #     jnp.greater_equal(state.player_score, 20),
        #     jnp.greater_equal(state.enemy_score, 20),
        # )

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Pong Game")
    clock = pygame.time.Clock()

    game = JaxPooyan()

    # Create the JAX renderer
    renderer = PooyanRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

    while running:
        raster = renderer.render(curr_state)
        curr_state = jitted_step(curr_state, 0)[1]

        jr.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(60)

    pygame.quit()