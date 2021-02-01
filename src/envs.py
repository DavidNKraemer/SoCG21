import gym
import numpy as np

from src.board import DistributedBoard, LocalState


def obstacles_hit(agent):
    """
    Return the number of obstacles with which the specified agent collided
    over the previous time-step.

    The output will either be 0 or 1, for an agent moves only one unit per
    time-step, and no two obstacles overlap (obstacles are merely pixels, not
    collections thereof). We chose to make this function integer-valued -- not
    boolean-valued -- to maintain continuity with agents_hit().
    """
    # obstacles don't move. Collision occurred in previous time-step iff
    # agent's position in present time-step is an obstacle pixel.
    if agent.position in agent.board.obstacles:
        return 1
    else:
        return 0

def agents_hit(agent):
    """
    Return the number of other agents with which the specified agent collided
    over the previous time-step.
    """
    # a collision occurred iff in agent's neighborhood, another agent a' moved
    # cardinal directions (i.e., from N to E, from S to N, from E to N, etc.)
    N = np.array([0, 1])
    S = np.array([0, -1])
    W = np.array([-1, 0])
    E = np.array([1, 0])

    n_collisions = 0
    for direction in [N, S, W, E]:
        # the set of agents in the pixel to the e.g., North of the agent;
        # store set of agents in this pixel at previous time-step
        prev_agents = agent.board.prev_active_pixels[
            tuple(agent.position + direction)]
        for other_direction in [N, S, W, E]:
            # check all other cardinal directions at the current time-step
            if other_direction is not direction:
                # store set of agents in this pixel at current time-step
                curr_agents = agent.board.active_pixels[
                    tuple(agent.position + other_direction)]
                # take intersection of two sets, and increment by cardinality
                n_collisions += len(curr_agents & prev_agents)

    return n_collisions

def agent_reward(agent, alpha, beta, gamma):
    """
    Return the "instantaneous" reward signal for the specified agent.

    alpha, beta, and gamma require tuning.

    Params
    ------
    agent: Agent
        Agent under consideration.
    alpha: float
        Penalty multiplier for distance-to-go.
    beta: float
        Pentalty multiplier for hitting an obstacle.
    gamma: float
        Penalty multiplier for hitting another agent.

    Returns
    -------
    reward: float
        Instantaneous reward signal.

    Preconditions
    -------------
    board.reset() has already been called.
    """
    # dist_to_go() returns the l_1 distance between an agent's position
    # and its target, ignoring intermediate obstacles and other agents that
    # might be in the way. See board.py's Agent class.
    return (alpha*agent.dist_to_go() + beta*obstacles_hit(agent)
            + gamma*agents_hit(agent))

def board_reward(board, alpha, beta, gamma):
    """
    Given a DistributedBoard, return the "instantaneous" reward signal.

    This is to be highly engineered. Currently it's stupid.

    Params
    ------
    board: DistributedBoard
        Game board whose state determines the reward signal.
    alpha: float
        Penalty multiplier for distance-to-go.
    beta: float
        Pentalty multiplier for hitting an obstacle.
    gamma: float
        Penalty multiplier for hitting another agent.

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
