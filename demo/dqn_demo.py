import torch
import numpy as np


from src.dqn import agents, envs, models
from src.envs import BoardEnv, agent_reward
from src.board import AgentForDQN


### hyperparameters

# env
num_images = 3  # number of images in a state
starts = np.array([[0, 0]])
targets = np.array([[5, 5]])
obstacles = np.array([])
dist_penalty = 1
obs_hit_penalty = 0
agents_hit_penalty = 0
num_actions = 5

def reward_fn(agent):
    return agent_reward(
        agent, dist_penalty, obs_hit_penalty, agents_hit_penalty
    )

# DQN
in_channels = num_images
out_channels1 = 16
kernel_size1 = 2
stride1 = 1
in_channels2 = 16
out_channels2 = 32
kernel_size2 = 2
stride2 = 2
out_features3 = 256

# agent
batch_size = 256
buff_maxlen = 100_000
q_lr = 0.01
discount_gamma = 0.99
polyak_tau = 0.005
greedy_eps = 0.1
enable_cuda = False  # TODO: get working on CUDA
grad_clip_radius = None

# training
num_episodes = 100
episode_length = 100


def tensor(x, cuda=enable_cuda):
    """Convert numpy array to torch tensor on desired device."""
    device = torch.device('cuda') if cuda else torch.device('cpu')
    x = x.copy() if isinstance(x, np.ndarray) else x
    return torch.tensor(
        x, dtype=torch.float32).to(device).unsqueeze(dim=0)


if __name__ == "__main__":

    env = BoardEnv(starts, targets, obstacles, reward_fn,
                   agent_type=AgentForDQN)
    example_state_tensor = tensor(env.reset())

    q_net = models.ConvolutionalDQN(
        in_channels, num_actions,
        example_state_tensor,
        out_channels1=out_channels1,
        kernel_size1=kernel_size1,
        stride1=stride1,
        in_channels2=in_channels2,
        out_channels2=out_channels2,
        kernel_size2=kernel_size2,
        stride2=stride2,
        out_features3=out_features3
    )

    learner = agents.DoubleDQNAgent(
        batch_size, num_actions, buff_maxlen,
        q_net, q_lr,
        discount_gamma=discount_gamma,
        polyak_tau=polyak_tau,
        greedy_eps=greedy_eps,
        enable_cuda=enable_cuda,
        grad_clip_radius=grad_clip_radius
    )

    for ep in range(num_episodes):
        rewards = []
        for step in range(episode_length):
            state = tensor(env.state)
            action = learner.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            learner.update(tensor(reward).view(1,1),
                           tensor(next_state),
                           tensor(int(done)).view(1,1))
            rewards.append(reward)

        print(f'Episode {ep}: average reward {np.mean(rewards)}')