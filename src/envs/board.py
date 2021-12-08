import gym
import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector

from src.sim_stats import SimStats
from src.board import DistributedBoard, LocalState
from src.utils import obstacles_hit, bots_hit, board_reward


def env(*args, **kwargs):
    env = raw_env(*args, **kwargs)

    wrapper_classes = [
        # crashes for discrete actions out of bounds
        wrappers.AssertOutOfBoundsWrapper,
        # crashes when reset() hasn't been called before environment processing
        # warns on close() before render() or reset(), on step() after env is
        # done
        wrappers.OrderEnforcingWrapper,
    ]

    for wrapper in wrapper_classes:
        env = wrapper(env)

    return env


class raw_env(AECEnv):
    def __init__(self, starts, targets, obstacles, **board_kwargs):
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

        Preconditions
        -------------
        starts.shape[0] == targets.shape[0]
        starts.shape[1] == targets.shape[1] == obstacles.shape[1] == 2
        """
        self.metadata = {}

        self.board = DistributedBoard(
            starts, targets, obstacles, **board_kwargs
        )

        alpha, beta, gamma, finish_bonus = 0, 0, 1.0, 0  # 1.0, 1.5, 2.0, 10.0
        self.reward_fn = lambda board: board_reward(
            board, alpha, beta, gamma, finish_bonus
        )

        n_bots = len(starts)

        # list of agents. these are *labels* associated with each agent. once
        # this list is filled, it is not modified.
        self.possible_agents = [f"bot_{agent_id}" for agent_id in range(n_bots)]
        agents = self.possible_agents

        # dictionary mapping agent (labels) to ids.
        self.agent_name_mapping = {
            agent: agent_id for agent_id, agent in enumerate(agents)
        }

        # dictionary mapping agent (labels) to action spaces.
        self.action_spaces = gym.spaces.Dict(
            {agent: gym.spaces.Discrete(5) for agent in agents}
        )

        # dictionary mapping aggent (labels) to observation spaces.
        self.observation_spaces = gym.spaces.Dict(
            {
                agent: gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=self.board._obs_shape
                )
                for agent in agents
            }
        )

        self._agent_selector = agent_selector(agents)

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

        self.agents = self.possible_agents.copy()

        for agent in self.agents:
            # grab the associated bot
            bot = self.board.bots[self.agent_name_mapping[agent]]

            self.rewards[agent] = 0.0
            self._cumulative_rewards[agent] = 0.0

            self.dones[agent] = self.board.isdone()
            self.infos[agent] = {}
            self.observations[agent] = bot.state

        # grab the first agent from the priority queue
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        Given an action, update the game board and return the next state.

        Params
        ------
        action: int
            Action for the given bot to move and update the board.

        Caution
        -------
        1. The step method does not ask for the agent to update. The agent up
           to bat is the agent stored in the 'agent_selection' field. The
           action supplied as a parameter to this method will be applied to
           that agent. The step method will process the agent, its internal bot
           representation, update the DistributedBoard accordingly, and set the
           next agent. Don't do this yourself!

        2. Unlike the gym API, the step method does not return anything!  To
           get the relevant (observation, reward, done, info) tuple, call the
           last() method.

        [Called for side-effects]
        """
        if self.dones[self.agent_selection]:
            action = None
            self._was_done_step(action)
            return None

        agent = self.agent_selection

        actions = ["E", "W", "N", "S", ""]

        # process the underlying DistributedBoard and Bot object.
        bot_id = self.agent_name_mapping[agent]
        self.board.bot_actions[bot_id] = actions[action]

        # only at the last bot do we actually move, and thereby update
        # observations
        if self._agent_selector.is_last():
            self.board.update_bots()
            # update all observations, dones, etc.

            for agent, bot in zip(self.agents, self.board.bots):
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
                self.rewards[agent] = self.reward_fn(self.board)

        self.agent_selection = self._agent_selector.next()
        self.board.bot_selection = self.board._bot_selector.next()

        self._cumulative_rewards[agent] = 0.0
        self._accumulate_rewards()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        """
        Get the current observation associated with a particular agent.

        Params
        ------
        agent: str

        Returns
        -------
        observation
        """
        return self.observations[agent]

    def seed(self, seed=None):
        """ """
        raise NotImplementedError

    def render(self, mode="human"):
        """
        TODO: Implement
        """
        pass

    def state(self):
        """
        TODO: Implement
        """
        return self.observations

    def close(self):
        """
        TODO: Implement
        """
        pass
