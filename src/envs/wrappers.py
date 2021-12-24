import gym
import numpy as np
from itertools import product


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

    def reset(self):
        self.pz_env.reset()
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
            agent_id = self.pz_env.agent_name_mapping[agent]
            self.pz_env.step(action_tuple[agent_id])

        done = self.board.isdone()
        # centralized training: one agent
        reward = self.raw_pz_env.rewards[self.raw_pz_env.agents[0]]

        return self.state, reward, done, {}

    @property
    def state(self):
        return self.raw_pz_env.observations
