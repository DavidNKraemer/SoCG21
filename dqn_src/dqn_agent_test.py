import torch


import agents, envs, models, util


### hyperparameters

# env
in_channels = 3
side_len = 9
num_actions = 5

# DQN
out_channels1 = 16
kernel_size1 = 3
stride1 = 2
in_channels2 = 16
out_channels2 = 32
kernel_size2 = 2
stride2 = 1
out_features3 = 256

# agent
batch_size = 256
buff_maxlen = 100_000
q_lr = 0.01
discount_gamma = 0.99
polyak_tau = 0.005
greedy_eps = 0.01
enable_cuda = False  # TODO: get working on CUDA
grad_clip_radius = None

# training
num_episodes = 10
episode_length = 100


def tensor(x, cuda=enable_cuda):
    """Convert numpy array to torch tensor on desired device."""
    device = torch.device('cuda') if cuda else torch.device('cpu')
    return torch.tensor(x, dtype=torch.float32).to(device)


if __name__ == "__main__":

    env = envs.DummyEnv(in_channels, side_len, num_actions)
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

    agent = agents.DoubleDQNAgent(
        batch_size, num_actions, buff_maxlen,
        q_net, q_lr,
        discount_gamma=discount_gamma,
        polyak_tau=polyak_tau,
        greedy_eps=greedy_eps,
        enable_cuda=enable_cuda,
        grad_clip_radius=grad_clip_radius
    )

    for ep in range(num_episodes):
        for step in range(episode_length):

            action = agent.sample_action(tensor(env.state))
            reward, next_state, done = env.step(action)
            agent.update(tensor(reward).view(1,1),
                         tensor(next_state),
                         tensor(int(done)).view(1,1))

            print(f'Episode {ep}: step {step}')
