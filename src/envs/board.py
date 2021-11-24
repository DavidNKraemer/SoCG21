import gym
import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from src.sim_stats import SimStats
from src.board import DistributedBoard, LocalState
from src.utils import obstacles_hit, bots_hit, bot_reward


def env(*args, **kwargs):
    env = raw_env(*args, **kwargs)

    wrapper_classes = [
        # crashes for discrete actions out of bounds
        wrappers.AssertOutOfBoundsWrapper,

        # crashes when reset() hasn't been called before environment processing
        # warns on close() before render() or reset(), on step() after env is
        # done
        wrappers.OrderEnforcingWrapper
    ]

    for wrapper in wrapper_classes:
        env = wrapper(env)

    return env


class raw_env(AECEnv):
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
        self.metadata = {}

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
