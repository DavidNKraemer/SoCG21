import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from src.dqn import agents, envs, models
from src.envs import BoardEnv, agent_reward
from src.board import AgentForDQN
from plot.plot_schedule import plot


class DQNTrainer:
    """
    Trains a single DQN agent across multiple different training sessions.

    The Trainer can do the following:
        - read in a YAML configuration file and construct the agent and
          associated networks
        - load a pre-existing checkpoint of the agent, if requested
        - take a tuple (starts, targets, obstacles) as input and construct
          the associated environment
        - take a specification for a training session and carry it out
        - save a checkpoint of the agent
    """

    def __init__(self, config_filename):
        """
        Read a YAML configuration file and construct the agent.
        """

        with open(config_filename, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda') if \
                self.config['agent_config']['enable_cuda'] \
                else torch.device('cpu')
        self._reward_fn = None
        self.env = None

        self.reset()

    def _tensor(self, x):
        """Convert numpy array to torch tensor on desired device."""
        x = x.copy() if isinstance(x, np.ndarray) else x
        return torch.tensor(
            x, dtype=torch.float32).to(self.device).unsqueeze(dim=0)

    def _create_env(self, env_tuple):
        """
        Take env_tuple = (starts, targets, obstacles), then create and
        internally store the corresponding BoardEnv.
        """
        
        self.env = BoardEnv(
            *env_tuple, self.reward_fn, agent_type=AgentForDQN,
            neighborhood_radius=self.config['env_config']['neighborhood_radius']
        )

    def reset(self):
        """
        (Re-)create the agent from self.config.
        """

        dummy_env = BoardEnv(
            np.array([[0, 0]]), np.array([[0, 0]]), np.array([[]]),
            lambda x: 1, agent_type=AgentForDQN,
            neighborhood_radius=self.config['env_config']['neighborhood_radius']
        )
        example_state_tensor = self._tensor(dummy_env.reset())

        q_net = models.ConvolutionalDQN(
            in_channels=self.config['env_config']['num_images'],
            num_actions=self.config['env_config']['num_actions'],
            example_state_tensor=example_state_tensor,
            **self.config['network_config']
        )

        self.agent = agents.DoubleDQNAgent(
            num_actions=self.config['env_config']['num_actions'],
            q_net=q_net,
            **self.config['agent_config']
        )

        def reward_fn(agent):
            config = self.config['env_config']
            return agent_reward(
                agent,
                dist_pen=config['dist_penalty'],
                obs_hit_pen=config['obs_hit_penalty'],
                agents_hit_pen=config['agents_hit_penalty'],
                finish_bonus=config['finish_bonus']
            )

        self.reward_fn = reward_fn

    def load_checkpoint(self, checkpoint_filename, continue_training=True):
        """
        Load previous checkpoint.

        If the agent is only needed for evaluation, set
        continue_training=False.
        """

        self.agent.load_checkpoint(checkpoint_filename,
                                   continue_training=continue_training)

    def save_checkpoint(self, checkpoint_filename):
        """
        Save models and optimizers. Allows resumption of training at a
        later time.
        """

        self.agent.save_checkpoint(checkpoint_filename)

    def train(self, num_episodes, episode_length,
              env_tuple=None):
        """
        Carry out a training session.

        If env_tuple is None, use the current environment.
        """

        if env_tuple is not None:
            self._create_env(env_tuple)
        assert self.env is not None, 'Specify an env_tuple.'

        for ep in range(num_episodes):
            self.env.reset()
            rewards = []
            for step in range(episode_length):
                state = self._tensor(self.env.state)
                action = self.agent.sample_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.update(self._tensor(reward).view(1,1),
                                  self._tensor(next_state),
                                  self._tensor(int(done)).view(1,1))
                rewards.append(reward)

            print(f'Episode {ep}: average reward {np.mean(rewards)}')

    def plot(self, episode_length, plot_filename, env_tuple=None):
        """
        Carry out an episode without training and generate a plot.

        If env_tuple is None, use the current environment.
        plot_filename should end with '.pdf'.
        """

        if env_tuple is not None:
            self._create_env(env_tuple)
        assert self.env is not None, 'Specify an env_tuple.'

        pp = PdfPages(plot_filename)

        self.env.reset()
        rewards = []
        clock = self.env.board.clock
        for step in range(episode_length):
            state = self._tensor(self.env.state)
            action = self.agent.sample_action(state)
            _, reward, done, _ = self.env.step(action)
            rewards.append(reward)

            # only plot the last episode, after which we hope to be not dumb
            if self.env.board.clock > clock:
                # advance plotting clock
                clock = self.env.board.clock
                # put plot callback here
                plot(self.env, pad=5)
                pp.savefig()

        pp.close()
