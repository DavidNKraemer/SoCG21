import gym


class gym_board_env(gym.Env):
    def __init__(self, env):
        self.pz_env = env
        self.raw_pz_env = self.pz_env.unwrapped

        self.board = self.raw_pz_env.board

        self.action_space = self.raw_pz_env.action_spaces
        self.observation_space = self.raw_pz_env.observation_spaces

    def reset(self):
        self.pz_env.reset()
        # maybe return self.state ?? unclear
        return self.state

    def step(self, actions):
        """

        Returns
        -------
        next_state: dict
        reward: float
        done: bool
        info: dict
        """
        assert self.action_space.contains(actions), "Try a dict"

        for agent in self.pz_env.agent_iter(max_iter=len(self.pz_env.agents)):
            self.pz_env.step(actions[agent])

        done = self.board.isdone()
        # centralized training: one agent
        reward = self.raw_pz_env.rewards[self.raw_pz_env.agents[0]]

        return self.state, reward, done, {}

    @property
    def state(self):
        return self.raw_pz_env.observations
