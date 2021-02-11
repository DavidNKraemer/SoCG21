import numpy as np
import torch
import copy
import warnings


from src.dqn.utils import TorchBuffer


class DoubleDQNAgent:
    """
    Agent carrying out the double DQN algorithm for maximization problems.

    Action selection is performed epsilon-greedily for now. Actions are
    assumed to be integers.
    """

    def __init__(self, batch_size, num_actions, buff_maxlen,
                 q_net, q_lr,
                 discount_gamma=0.99, polyak_tau=0.005,
                 greedy_eps=0.01, enable_cuda=False,
                 optimizer=torch.optim.Adam,
                 q_loss=torch.nn.MSELoss(),
                 grad_clip_radius=None):

        self.batch_size = batch_size
        self.num_actions = num_actions
        self.buffer = TorchBuffer(maxlen=buff_maxlen)

        self.q = q_net
        self.target_q = copy.deepcopy(self.q)

        self.__cuda_enabled = enable_cuda
        self.enable_cuda(self.__cuda_enabled, warn=False)

        self.q_optim = optimizer(self.q.parameters(), lr=q_lr)
        self.q_loss = q_loss

        self.gamma = discount_gamma
        self.tau = polyak_tau
        self.eps = greedy_eps

        self.state = None
        self.action = None

        self.grad_clip_radius = grad_clip_radius

    @property
    def cuda_enabled(self):
        return self.__cuda_enabled

    def enable_cuda(self, enable_cuda=True, warn=True):
        """
        Enable or disable CUDA and update models and actions.
        """

        if warn:
            warnings.warn("Converting between 'cpu' and 'cuda' after "
                          "initializing the optimizer can give errors when "
                          "using optimizers other than SGD or Adam!")

        self.__cuda_enabled = enable_cuda
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.__cuda_enabled \
            else 'cpu')
        self.q.to(self.device)
        self.target_q.to(self.device)

    def _polyak_average_params(self, params1, params2):
        """
        Compute Polyak average of params1 and params2 using tau. Copy result
        to params1.
        """

        for param1, param2 in zip(params1, params2):
            param1.data.copy_(self.tau * param2.data \
                              + (1.0 - self.tau) * param1.data)

    def sample_action(self, state):
        """
        Sample and action epsilon-greedily.
        """

        self.state = state.to(self.device)

        self.action = np.random.randint(0, self.num_actions) \
                if np.random.uniform() < self.eps \
                else self.q
        if np.random.uniform() < self.eps:
            self.action = torch.randint(high=self.num_actions, size=(1,1))
        else:
            with torch.no_grad():
                self.action = self.q(self.state).max(1)[1].view(1,1)

        return self.action

    def update(self, reward, next_state, done):
        """
        Add a new sample to the buffer and perform an update.
        """

        assert not None in {self.state, self.action}, 'sample_action must ' \
            + 'be called before update'

        self.buffer.append(self.state, self.action, reward, next_state, done)

        if len(self.buffer) >= self.batch_size:

            states, actions, rewards, next_states, dones = \
                    self.buffer.sample(self.batch_size, self.device)

            with torch.no_grad():
                target_vals = rewards + \
                        (1 - dones) * \
                        self.gamma * \
                        self.target_q(next_states).max(1)[0].unsqueeze(dim=1)

            loss = self.q_loss(self.q(states).gather(1, actions), target_vals)
            self.q_optim.zero_grad()
            loss.backward()
            if self.grad_clip_radius is not None:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(),
                                               self.grad_clip_radius)
            self.q_optim.step()

            self._polyak_average_params(self.target_q.parameters(),
                                        self.q.parameters())

    def save_checkpoint(self, filename):
        """
        Save state_dicts of models and optimizers. Allows resumption of
        training at a later time.
        """

        torch.save({'using_cuda': self.__cuda_enabled,
                    'q_state_dict': self.q.state_dict(),
                    'target_q_state_dict': self.target_q.state_dict(),
                    'q_optimizer_state_dict': self.q_optim.state_dict(),
                   }, filename)

    def load_checkpoint(self, filename, continue_training=True):
        """
        Load state_dicts for models and optimizers and continue training,
        if specified.
        """

        checkpoint = torch.load(filename)

        self.q.load_state_dict(checkpoint['q_state_dict'])
        self.target_q.load_state_dict(checkpoint['target_q_state_dict'])
        self.q_optim.load_state_dict(checkpoint['q_optimizer_state_dict'])

        if continue_training:
            self.q.train()
            self.target_q.train()

        else:
            self.q.eval()
            self.target_q.eval()

        self.enable_cuda(checkpoint['using_cuda'], warn=False)
