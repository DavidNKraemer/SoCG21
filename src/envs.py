import gym
import numpy as np

from src.board import DistributedBoard, LocalState


def board_reward(board):
    """
    Given a DistributedBoard, return the "instantaneous" reward signal.

    This is to be highly engineered. Currently it's stupid.

    Params
    ------
    board: DistributedBoard
        Game board whose state determines the reward signal.

    Returns
    -------
    reward: float
        Instantaneous reward signal

    Preconditions
    -------------
    board.reset() has already been called
    """
    return 1


class BoardEnv(gym.Env):

    def __init__(self, starts, targets, obstacles):
        """
        Params
        ------
        starts: numpy ndarray
            Array of starting positions for the agents
        targets: numpy ndarray
            Array of target positions for the agents
        obstacles: numpy ndarray
            Array of obstacle locations

        Preconditions
        -------------
        starts.shape[0] == targets.shape[0]
        starts.shape[1] == targets.shape[1] == obstacles.shape[1] == 2
        """
        self.board = DistributedBoard(starts, targets, obstacles)

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=LocalState.shape
        )

    def step(self, action):
        """
        Given an action, update the game board and return the next state.

        Params
        ------
        action: str
            Action for the given agent to move and update the board

        Returns
        -------
        observation: numpy ndarray
            Representation of the next state of the game board (or of the next
            agent in the queue?? TODO: clarify)
        reward: float
            Signal from the given action. This will require engineering, so I'm
            abstracting it by passing in an external function.
        done: bool
            Whether every agent has reached their target
        info: dict
            Currently empty

        Preconditions
        -------------
        action in range(5)
        self.reset() was already called
        """
        actions = ['E', 'W', 'N', 'S', '']

        # pop the next agent, move according to the action, and re-insert
        agent = self.board.pop()
        agent.move(actions[action])
        self.board.insert(agent)

        # save the state by peeking at the following agent in the queue
        self.state = self.board.peek().state

        done = self.board.isdone()

        # TODO: do something better
        reward = board_reward(self.board)

        return agent.state, reward, done, {}

    def reset(self):
        """
        Resets the board environment, along with all of its agents.

        [Called for side-effects]
        """
        self.board.reset()
        self.state = self.board.peek().state

        return self.state

    def render(self):
        """
        If we ever decide to make a visual representation of the game board...

        [Called for side-effects]
        """
        pass

    def close(self):
        """
        If we ever decide to make a visual representation of the game board...

        [Called for side-effects]
        """
        pass
