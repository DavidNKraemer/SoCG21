import gym
import numpy as np
from operator import attrgetter, mul
from itertools import product
from src.sim_stats import SimStats
from src.board import DistributedBoard, LocalState
from cgshop2021_pyutils import Direction, Solution, SolutionStep


from pettingzoo import AECEnv


def fitness(
    board_env,
    dist_trav_pen,
    time_pen,
    obs_hit_pen,
    bot_collisions_pen,
    error_pen,
    finish_bonus,
):
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
    bot_collisions_pen: float
        Penalty multiplier for the number of bot collisions.
    error_pen: float
        Penalty multiplier for the error. Error is defined as the distance
        between bot's final position at simulation end and target, aggregated
        over all bots.
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
    obs_hit, bot_collisions = sim_stats.obs_hit, sim_stats.bot_collisions
    error, finished = sim_stats.error, sim_stats.finished
    # return a linear combination of rewards
    penalties = [
        dist_trav_pen,
        time_pen,
        obs_hit_pen,
        bot_collisions_pen,
        error_pen,
    ]
    stats = [dist_trav, time, obs_hit, bot_collisions, error]
    costs = sum(map(mul, penalties, stats))

    return finish_bonus * finished - costs


def obstacles_hit(bot):
    """
    Return the number of obstacles with which the specified bot collided
    over the previous time-step.

    The output will either be 0 or 1, for an bot moves only one unit per
    time-step, and no two obstacles overlap (obstacles are merely pixels, not
    collections thereof). We chose to make this function integer-valued -- not
    boolean-valued -- to maintain continuity with bots_hit().
    """
    # obstacles don't move. Collision occurred in previous time-step iff
    # bot's position in present time-step is an obstacle pixel.
    return int(bot.position in bot.board.obstacles)


def bots_hit(bot):
    """
    Return the number of other bots with which the specified bot collided
    over the previous time-step.
    """
    # a collision occurred iff in bot's neighborhood, another bot a' moved
    # cardinal directions (i.e., from N to E, from S to N, from E to N, etc.)
    # TODO: import the dictionary from board.py
    N = np.array([0, 1])
    S = np.array([0, -1])
    W = np.array([-1, 0])
    E = np.array([1, 0])

    # number of other bots sharing the same pixel as bot
    n_collisions = len(bot.board.occupied_pixels[tuple(bot.position)]) - 1
    # DEBUG
    # print(f"Position: {bot.board.occupied_pixels[tuple(bot.position)]}")

    for direction, other_direction in product([N, S, W, E], [N, S, W, E]):
        if direction is other_direction:
            continue

        # DEBUG
        # print("direction: ", bot.board.prev_active_pixels[
        #    tuple(bot.prev_position + direction)])

        # store set of bot ids occupying this pixel at previous time-step
        prev_ids = bot.board.prev_occupied_pixels[
            tuple(bot.prev_position + direction)
        ]

        # DEBUG
        # print("other_direction: ", bot.board.prev_active_pixels[
        #    tuple(bot.position + other_direction)])

        # store set of bot ids occupying this pixel at current time-step
        curr_ids = bot.board.occupied_pixels[
            tuple(bot.position + other_direction)
        ]
        # take intersection of two sets, and increment by cardinality
        # print(f"Intersection: {curr_ids & prev_ids}")
        n_collisions += len(curr_ids & prev_ids)

    return n_collisions


def bot_reward(bot, dist_pen, obs_hit_pen, bots_hit_pen, finish_bonus):
    """
    Return the "instantaneous" reward signal for the specified bot.

    IMPORTANT
    ---------
    Higher *_pen values are associated with more penalty (i.e., more negative
    reward). Penalties are negated to convert them into rewards.

    Params
    ------
    bot: Bot
        Bot under consideration.
    dist_pen: float
        Penalty multiplier for distance-to-go.
    obs_hit_pen: float
        Penalty multiplier for hitting an obstacle.
    bots_hit_pen: float
        Penalty multiplier for hitting another bot.
    finish_bonus: float
        Reward for reaching one's target.

    Returns
    -------
    reward: float
        Instantaneous reward signal.

    Preconditions
    -------------
    board.reset() has already been called.
    """
    # dist_to_go() returns the l_1 distance between an bot's position
    # and its target, ignoring intermediate obstacles and other bots that
    # might be in the way. See board.py's Bot class.
    return finish_bonus * bot.attarget() - (
        dist_pen * bot.dist_to_go
        + obs_hit_pen * obstacles_hit(bot)
        + bots_hit_pen * bots_hit(bot)
    )


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
        Penalty multiplier for hitting another bot.

    Returns
    -------
    reward: float
        Instantaneous reward signal

    Preconditions
    -------------
    board.reset() has already been called
    """
    return 1  # TODO: unneeded?


class BoardEnv(gym.Env):
    def __init__(
        self, starts, targets, obstacles, instance, reward_fn, **board_kwargs
    ):
        """
        Params
        ------
        starts: numpy ndarray
            Array of starting positions for the bots
        targets: numpy ndarray
            Array of target positions for the bots
        obstacles: numpy ndarray
            Array of obstacle locations

        Preconditions
        -------------
        starts.shape[0] == targets.shape[0]
        starts.shape[1] == targets.shape[1] == obstacles.shape[1] == 2
        """
        self.board = DistributedBoard(
            starts, targets, obstacles, instance, **board_kwargs
        )

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=LocalState.shape
        )
        self.instance = instance  # SoCG artifact
        self.reward_fn = reward_fn

    def step(self, action):
        """
        Given an action, update the game board and return the next state.

        Params
        ------
        action: str
            Action for the given bot to move and update the board.

        Returns
        -------
        observation: numpy ndarray
            Representation of the next state of the game board (or of the next
            bot in the queue?? TODO: clarify)
        reward: float
            Signal from the given action. This will require engineering, so I'm
            abstracting it by passing in an external function.
        done: bool
            Whether every bot has reached their target
        info: dict
            Currently empty

        Preconditions
        -------------
        action in range(5)
        self.reset() was already called
        """
        # please don't re-order
        actions = ["E", "W", "N", "S", ""]

        # SoCG artifacts
        directions = [
            Direction.EAST,
            Direction.WEST,
            Direction.NORTH,
            Direction.SOUTH,
        ]

        # pop the next bot, move according to the action, and re-insert
        bot = self.board.pop()
        bot.move(actions[action])
        self.board.insert(bot)
        done = self.board.isdone()

        # SoCG artifacts TODO determine if needed at all
        if actions[action] != "":
            self.board.step[bot.bot_id] = directions[action]

        # update sim_stats
        self.sim_stats.bot_collisions += bots_hit(bot)
        self.sim_stats.obs_hit += obstacles_hit(bot)
        self.sim_stats.finished = done

        # if action is not the empty string
        if actions[action] != "":
            self.sim_stats.dist_trav += 1

        if done:
            self.sim_stats.time = self.board.clock
            self.sim_stats.compute_l1error(self.board)

        # save the state by peeking at the following bot in the queue
        self.state = self.board.peek().state

        reward = self.reward_fn(bot)

        return bot.state, reward, done, {}

    def reset(self):
        """
        Resets the board environment, along with all of its bots.

        [Called for side-effects]
        """
        self.sim_stats = SimStats()
        self.board.reset()
        self.state = self.board.peek().state

        # SoCG artifact
        self.solution = Solution(self.instance)

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


class PZBoardEnv(AECEnv):
    def __init__(
        self, starts, targets, obstacles, instance, reward_fn, **board_kwargs
    ):
        """ 
        Params
        ------
        starts: numpy ndarray
            Array of starting positions for the bots
        targets: numpy ndarray
            Array of target positions for the bots
        obstacles: numpy ndarray
            Array of obstacle locations
        instance: SoCG artifact
            (soon to be deprecated)
        reward_fn: board.Bot -> float
            instantaneous reward function that takes a Bot object and returns
            the reward associated with its local state

        Preconditions
        -------------
        starts.shape[0] == targets.shape[0]
        starts.shape[1] == targets.shape[1] == obstacles.shape[1] == 2


        """
        self.board = DistributedBoard(
            starts, targets, obstacles, instance, **board_kwargs
        )

        self.reward_fn = reward_fn

        n_bots = len(starts)


        # list of agents. these are *labels* associated with each agent. once
        # this list is filled, it is not modified.
        self.agents = []  

        # dictionary mapping agent (labels) to ids.
        self.agent_name_mapping = {}

        # dictionary mapping agent (labels) to action spaces.
        self.action_spaces = {}

        # dictionary mapping aggent (labels) to observation spaces.
        self.observation_spaces = {}
        for agent_id in range(n_bots):
            agent = f'bot_{agent_id}'
            self.agents.append(agent)
            self.agent_name_mapping[agent] = agent_id 
            self.action_spaces[agent] = gym.spaces.Discrete(5)
            self.observation_spaces[agent] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=LocalState.shape
            )

        # function to select the next agent. this is the mechanism through
        # which self.agent_iter() moves through the agents. currently, grabs
        # the agent label associated with a newly-popped bot from the
        # DistributedBoard priority queue. 
        # caution: calling this function modifies the DistributedBoard!
        self._agent_selector = lambda: self.agents[self.board.pop().bot_id]

    def reset(self):
        """
        Resets the underlying DistributedBoard and all of its bots, resets the
        internal SimStats tracker, and resets all of the reward, done, info,
        and observation data.

        Call this method before simulating the environment!

        [Called for side-effects]
        """
        self.sim_stats = SimStats()
        self.board.reset()

        self.rewards = {}
        self._cumulative_rewards = {}
        self.dones = {}
        self.infos = {}
        self.observations = {}

        for agent in self.agents:

            # grab the associated bot
            bot = self.board.bots[self.agent_name_mapping[agent]]

            self.rewards[agent] = 0.0
            self._cumulative_rewards[agent] = 0.0

            # TODO: maybe just bot.attarget()?
            self.dones[agent] = self.board.isdone()  

            self.infos[agent] = {}
            self.observations[agent] = bot.state

        # grab the first agent from the priority queue
        self.agent_selection = self._agent_selector()

    def step(self, action):
        """ 
        Given an action, update the game board and return the next state.

        Params
        ------
        action: int
            Action for the given bot to move and update the board.
            
        Caution: The step method does not ask for the agent to update. The
        agent up to bat is the agent stored in the 'agent_selection' field. The
        action supplied as a parameter to this method will be applied to that
        agent. The step method will process the agent, its internal bot
        representation, update the DistributedBoard accordingly, and set the
        next agent. Don't do this yourself!

        Caution: Unlike the gym API, the step method does not return anything!
        To get the relevant (observation, reward, done, info) tuple, call the
        last() method.
    
        [Called for side-effects]

        """
        if self.dones[self.agent_selection]:
            action = None
            self._was_done_step(action)
            return None

        agent = self.agent_selection

        assert action in self.action_spaces[agent], "Invalid action!"

        actions = ["E", "W", "N", "S", ""]

        # process the underlying DistributedBoard and Bot object. 
        bot = self.board.bots[self.agent_name_mapping[agent]]
        bot.move(actions[action])
        self.board.insert(bot)  # pushed back onto the priority queue

        # maybe just bot.attarget()?
        self.dones[agent] = self.board.isdone()  

        self.observations[agent] = bot.state

        # update sim_stats
        self.sim_stats.bot_collisions += bots_hit(bot)
        self.sim_stats.obs_hit += obstacles_hit(bot)
        self.sim_stats.finished = self.dones[agent]

        # if action is not the empty string
        if actions[action] != "":
            self.sim_stats.dist_trav += 1

        if self.dones[agent]:
            self.sim_stats.time = self.board.clock
            self.sim_stats.compute_l1error(self.board)

        # reward does not accumulate: cumulative == instantaneous
        self.rewards[agent] = self.reward_fn(bot)
        self._cumulative_rewards[agent] = 0.0
        self._accumulate_rewards()

        # pops the agent corresponding to the next bot off the queue and stores
        # it in the field agent_selection
        self.agent_selection = self._agent_selector()

    def seed(self, seed=None):
        """
        TODO: Implement
        """
        pass

    def observe(self, agent):
        assert agent in self.agents, "Invalid agent!"

        return self.observations[agent]

    def render(self, mode="human"):
        """
        TODO: Implement
        """
        pass

    def state(self):
        """
        TODO: Implement
        """
        pass

    def close(self):
        pass

    def observation_space(self, agent):
        assert agent in self.agents, "Invalid agent!"

        return self.observation_spaces[agent]

    def action_space(self, agent):
        assert agent in self.agents, "Invalid agent!"

        return self.action_spaces[agent]
