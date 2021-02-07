import gym
import numpy as np
from operator import attrgetter
from src.sim_stats import SimStats
from src.board import DistributedBoard, LocalState


def fitness(board_env, dist_trav_pen, time_pen, obs_hit_pen,
            agent_collisions_pen, error_pen, finish_bonus):
    """
    Return the value of the fitness function associated with a simulation
    trajectory.

    IMPORTANT
    ---------
    Higher *_pen values are associated with more penalty (i.e., more negative
    reward), and higher finish_bonus values are associated with more reward.
    Penalties are negated to convert them into rewards.

    Params
    ------
    board_env: BoardEnv
        Board environment AFTER a simulation of the genetic algorithm has
        terminated. From this, we extract board_env.sim_stats.
    dist_trav_pen: float
        Penalty multiplier for total distance travelled.
    time_pen: float
        Penalty multiplier for total time elapsed, computed using board clock.
    obs_hit_pen: float
        Pentalty multiplier for the number of obstacles hit.
    agent_collisions_pen: float
        Penalty multiplier for the number of agent collisions.
    error_pen: float
        Penalty multiplier for the error. Error is defined as the distance
        between agent's final position at simulation end and target, aggregated
        over all agents.
    finish_bonus: float
        Reward multiplier for finishing before the simulation circuit breaker
        has been tripped.

    Returns
    -------
    reward: float
        Reward signal.
    """
    sim_stats = board_env.sim_stats  # alias
    # destructure sim_stats so below linear combination is more concise
    dist_trav, time = sim_stats.dist_trav, sim_stats.time
    obs_hit, agents_hit = sim_stats.obs_hit, sim_stats, agents_hit
    error, finish = sim_stats.error, sim_stats.finish
    # return a linear combination of rewards
    return (finish_bonus*finish -(dist_trav_pen*dist_trav + time_pen*time
                                  + obs_hit_pen*obs_hit
                                  + agent_collisons_pen*agent_collisions
                                  + error_pen*error))

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
    return int(agent.position in agent.board.obstacles)

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

    agent_id = attrgetter("agent_id")
    n_collisions = len(agent.board.active_pixels[tuple(agent.position)]) - 1
    for direction in [N, S, W, E]:
        # the set of agents in the pixel to the e.g., North of the agent;
        # store set of agents in this pixel at previous time-step
        prev_agents = agent.board.prev_active_pixels[
            tuple(agent.prev_position + direction)]
        prev_ids = set(map(agent_id, prev_agents))
        for other_direction in [N, S, W, E]:
            # check all other cardinal directions at the current time-step
            if other_direction is not direction:
                # store set of agents in this pixel at current time-step
                curr_agents = agent.board.active_pixels[
                    tuple(agent.position + other_direction)]
                curr_ids = set(map(agent_id, curr_agents))
                # take intersection of two sets, and increment by cardinality
                n_collisions += len(curr_ids & prev_ids)

    return n_collisions

def agent_reward(agent, dist_pen, obs_hit_pen, agents_hit_pen):
    """
    Return the "instantaneous" reward signal for the specified agent.

    IMPORTANT
    ---------
    Higher *_pen values are associated with more penalty (i.e., more negative
    reward). Penalties are negated to convert them into rewards.

    Params
    ------
    agent: Agent
        Agent under consideration.
    dist_pen: float
        Penalty multiplier for distance-to-go.
    obs_hit_pen: float
        Pentalty multiplier for hitting an obstacle.
    agents_hit_pen: float
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
    return -(dist_pen*agent.dist_to_go() + obs_hit_pen*obstacles_hit(agent)
            + agents_hit_pen*agents_hit(agent))

# TODO: decide if we need this function
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

    def __init__(self, starts, targets, obstacles, reward_fn):
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
        self.reward_fn = reward_fn
        self.sim_stats = SimStats()

    def step(self, action):
        """
        Given an action, update the game board and return the next state.

        Params
        ------
        action: str
            Action for the given agent to move and update the board.

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
        # please don't re-order
        actions = ['E', 'W', 'N', 'S', '']

        # pop the next agent, move according to the action, and re-insert
        agent = self.board.pop()
        agent.move(actions[action])
        self.board.insert(agent)

        # update sim_stats
        self.sim_stats.agent_collisions += agents_hit(agent)
        self.sim_stats.obs_hit += obstacles_hit(agent)
        # if action is not the empty string
        if action != 4:
            self.sim_stats.dist_trav += 1
        self.sim_stats.time = self.board.clock
        self.sim_stats.error(self.board)

        # save the state by peeking at the following agent in the queue
        self.state = self.board.peek().state

        done = self.board.isdone()

        # TODO: do something better
        reward = self.reward_fn(agent)

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
