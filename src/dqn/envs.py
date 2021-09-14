import numpy as np


class DummyEnv:
    """
    Dummy environment for testing DQN implementations.
    """

    def __init__(self, in_channels=3, side_len=9, num_actions=5):

        self.in_channels = in_channels
        self.side_len = side_len

        self.state = None
        self.seed = 0

    def _random_state(self):
        """
        Generate a random state.
        """

        return self.rng.normal(
            size=(1, self.in_channels, self.side_len, self.side_len)
        )

    def reset(self):
        """
        Initialize the environment state.
        """

        self.state = self._random_state()
        self.rng = np.random.default_rng(self.seed)

        return self.state

    def step(self, action):
        """
        Take an action, return a reward, next state, and whether the
        episode is done.
        """

        reward = self.rng.random()
        self.state = self._random_state()
        done = False

        return reward, self.state, done
