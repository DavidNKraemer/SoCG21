import numpy as np
import torch
from collections import namedtuple, deque


class TorchBuffer:
    """
    Buffer for easily storing and sampling torch tensors of experiences
    for batch training of RL algorithms.
    """

    Experience = namedtuple('Experience', ('state', 'action', 'reward',
                                           'next_state', 'done'))

    def __init__(self, maxlen=100000):
        self.__buffer = deque(maxlen=maxlen)

    @property
    def buffer(self):
        return self.__buffer

    def __len__(self):
        return len(self.__buffer)

    def append(self, state, action, reward, next_state, done):
        """
        Append to the buffer.

        NOTE: state, action, reward, next_state, and done should all be
        torch tensors of an appropriate kind. The burden is on the user
        to ensure this, as no checks are performed.
        """

        self.__buffer.append(
            self.Experience(state, action, reward, next_state, done)
        )

    def sample(self, batch_size, device=torch.device('cpu')):
        """
        Sample a batch of experiences:
            states, actions, rewards, next_states, dones
        and return them in nice torch.tensor form.

        TODO: be more concise.
        """

        batch_size = min(batch_size, len(self))
        indices = np.random.randint(0, len(self), size=(batch_size,))
        sample = [self.__buffer[i] for i in indices]
        states = torch.cat([elem.state for elem in sample]).to(device)
        actions = torch.cat([elem.action for elem in sample]).to(device)
        rewards = torch.cat([elem.reward for elem in sample]).to(device)
        next_states = torch.cat([elem.next_state for elem in sample]).to(device)
        dones = torch.cat([elem.done for elem in sample]).to(device)

        return states, actions, rewards, next_states, dones
