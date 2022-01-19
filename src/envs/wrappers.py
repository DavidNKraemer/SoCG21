import gym
import numpy as np
import pyglet
import pyglet.gl as gl
import sys

from itertools import product
from pyglet import shapes, text
from pyglet.window import key, mouse
from random import choice


class Viewer:

    def __init__(self, **params):
        self.width         = params.get('width', 1200)
        self.height        = params.get('height', 600)
        self.n_blocks      = params.get('n_blocks', 40)
        self.scale         = params.get('scale', 1.1)
        self.update_speed  = params.get('update_speed', 1/2)
        self.key_speed     = params.get('key_speed', 1/30)
        self.current_scale = params.get('current_scale', 1.)
        self.max_scale     = params.get('max_scale', 2.)
        self.min_scale     = params.get('min_scale', 0.2)
        self.spec          = params.get('spec', None)

        self.block_size = min(self.width, self.height) // self.n_blocks
        self.translate_speed = 2 * self.update_speed * self.block_size
    
        self.display = Viewer.get_display(self.spec)
        self.window = Viewer.get_window(self.width, self.height, self.display)

        pyglet.clock.schedule_interval(self._process_key, self.key_speed)
    
        self.batch = pyglet.graphics.Batch()

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    
    @staticmethod
    def get_display(spec):
        if spec is None:
            return pyglet.canvas.get_display()
        elif isinstance(spec, str):
            return pyglet.canvas.Display(spec)
        else:
            raise error.Error(
                f"Invalid display specification: {spec}. (Must be a string like :0 or None.)"
            )
    
    @staticmethod
    def get_window(width, height, display, **kwargs):
        screen = display.get_screens()
        config = screen[0].get_best_config()
        context = config.create_context(None)
    
        window = pyglet.window.Window(
            width=width,
            height=height,
            display=display,
            config=config,
            context=context
        )
    
        window.key_handler = pyglet.window.key.KeyStateHandler()
        window.push_handlers(window.key_handler)
    
        gl.glPushMatrix()
    
        return window

    @staticmethod
    def _print_help():
        char_pad = 10
        info_pad = 40
    
        arrows = {
            'right': '\u2192',
            'left': '\u2190',
            'up': '\u2191',
            'down': '\u2193',
        }
    
        instructions = {
            **{
                char: f'Move {direction}' 
                for direction, char in arrows.items()
            },
            'h': 'Print viewer instructions',
            ',': 'Zoom in',
            '.': 'Zoom out',
            '/': 'Reset view',
        }
    
        print('\u250c', end='')
        print('\u2500' * (char_pad + info_pad + 5), end='')
        print('\u2510')
    
        print('\u2502', end='')
        print('Viewer Instructions'.center(char_pad + info_pad + 5), end='')
        print('\u2502')
    
        print('\u251c', end='')
        print('\u2500' * char_pad, end='')
        print('\u252c', end='')
        print('\u2500' * (info_pad + 4), end='')
        print('\u2524')
    
        for char, info in instructions.items():
            print('\u2502', end='')
            print(char.center(char_pad), end='')
            print('\u2502\t', end='')
            print(info.ljust(info_pad), end='')
            print('\u2502')
    
        print('\u2514', end='')
        print('\u2500' * char_pad, end='')
        print('\u2534', end='')
        print('\u2500' * (info_pad + 4), end='')
        print('\u2518')

    def _process_key(self, dt):
        speed = self.translate_speed / self.current_scale
    
        key_handler = self.window.key_handler
        # translation keys
        if key_handler[key.UP]:
            gl.glTranslatef(0, -speed, 0)
        if key_handler[key.DOWN]:
            gl.glTranslatef(0, speed, 0)
        if key_handler[key.LEFT]:
            gl.glTranslatef(speed, 0, 0)
        if key_handler[key.RIGHT]:
            gl.glTranslatef(-speed, 0, 0)
        if key_handler[key.H]:
            Viewer._print_help()
    
        if key_handler[key.SLASH]:
            gl.glPopMatrix()
            gl.glPushMatrix()
            self.current_scale = 1.
    
        # zooming keys
        if key_handler[key.PERIOD]:  # out
            if self.current_scale > self.min_scale:
                gl.glScalef(1/self.scale, 1/self.scale, 1.)
                self.current_scale /= self.scale
    
        if key_handler[key.COMMA]:  # out
            if self.current_scale < self.max_scale:
                gl.glScalef(self.scale, self.scale, 1.)
                self.current_scale *= self.scale

    def close(self):
        self.window.close()
    
    def render(self):
        pyglet.clock.tick()
        gl.glClearColor(1, 1, 1, 1)
    
        self.window.clear()
        self.batch.draw()
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.flip()

        return None
    
    def draw_square(self, x, y, length, color=None):
        if color is None:
            color = (0, 0, 0)

        rectangle = shapes.Rectangle(
            x=x, 
            y=y, 
            width=length, 
            height=length, 
            color=color,
            batch=self.batch
        )

        rectangle.opacity = 127

        return rectangle

    def draw_text(self, x, y, label, color=None):
        if color is None:
            color = (0, 0, 0, 255)

        mid_x = x + self.block_size / 4
        mid_y = y + self.block_size / 4

        return text.Label(
            label,
            x=mid_x, y=mid_y, 
            color=color,
            font_size=10,
            batch=self.batch
        )
    

class gym_board_env(gym.Env):
    def __init__(self, env):
        self.pz_env = env
        self.raw_pz_env = self.pz_env.unwrapped


        self.board = self.raw_pz_env.board

        self.n_agents = len(self.raw_pz_env.possible_agents)

        # self.action_space = gym.spaces.Box(
        #     low=0, high=4, shape=(self.n_agents,), dtype=np.uint8
        # )

        self.int_to_action = {}
        self.action_to_int = {}

        for i, action in enumerate(product(range(5), repeat=self.n_agents)):
            self.int_to_action[i] = action
            self.action_to_int[action] = i 

        self.action_space = gym.spaces.Discrete(5 ** (self.n_agents))

        self.observation_space = self.raw_pz_env.observation_spaces

        # rendering
        self.viewer = None

    def reset(self):
        self.pz_env.reset()

        self.close()
        # maybe return self.state ?? unclear
        return self.state

    def step(self, action):
        """

        Returns
        -------
        next_state: dict
        reward: float
        done: bool
        info: dict
        """
        assert self.action_space.contains(action), "bad action"

        action_tuple = self.int_to_action[action]

        for agent in self.pz_env.agent_iter(max_iter=self.n_agents):
            agent_id = self.raw_pz_env.agent_name_mapping[agent]
            self.pz_env.step(action_tuple[agent_id])

        done = self.board.isdone()
        # centralized training: one agent
        reward = self.raw_pz_env.rewards[self.raw_pz_env.agents[0]]

        return self.state, reward, done, {}

    @property
    def state(self):
        return self.raw_pz_env.observations

    def render(self, mode="human"):
        """
        """

        if self.viewer is None:

            self.viewer = Viewer()

            size = self.viewer.block_size
            self.obstacles = []
            for i, (x, y) in enumerate(self.board.obstacles):
                self.obstacles.append(
                    self.viewer.draw_square(
                        x * size, 
                        y * size, 
                        size,
                        color=(127, 127, 127),
                    )
                )
                # self.viewer.draw_text(*obs, str(i))

            self.targets = []
            for i, (x, y) in enumerate(self.board._targets):
                self.targets.append(
                    self.viewer.draw_square(
                        x * size, 
                        y * size, 
                        size,
                        color=(0, 255, 0)
                    )
                )
                self.viewer.draw_text(x * size, y * size, str(i))

            self.sprites = []
            for i, bot in enumerate(self.board.bots):
                bot = bot.position * size
                square = self.viewer.draw_square(
                    *bot, 
                    size, 
                    color=(0, 0, 255)
                )
                label = self.viewer.draw_text(*bot, str(i))
                self.sprites.append((square, label))

        pyglet.clock.tick()

        for bot, (square, label) in zip(self.board.bots, self.sprites):
            x, y = bot.position * self.viewer.block_size

            square.x = x
            square.y = y
            
            label.x = x + self.viewer.block_size / 4
            label.y = y + self.viewer.block_size / 4

        return self.viewer.render()

    def close(self):
        """
        Close the viewer, if needed
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

