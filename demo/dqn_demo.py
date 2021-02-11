import torch
import numpy as np
from src.dqn import agents, envs, models
from src.envs import BoardEnv, agent_reward, agents_hit
from src.board import AgentForDQN
from plot.plot_schedule import plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


### hyperparameters

# env
num_images = 3  # number of images in a state
starts = np.array([[0,0], [5,0]])
targets = np.array([[5,5], [0,5]])
obstacles = np.array([[3,0], [3,1], [3,2], [3,3], [3,4], [3,5]])
dist_penalty = 100
obs_hit_penalty = 50
agents_hit_penalty = 50
finish_bonus = 1000

num_actions = 5
neighborhood_radius = 10

def reward_fn(agent):
    return agent_reward(
        agent, dist_penalty, obs_hit_penalty, agents_hit_penalty, finish_bonus
    ), agents_hit(agent), int(np.all(agent.position == agent.target))

# DQN
in_channels = num_images
out_channels1 = 64
kernel_size1 = 3
stride1 = 2
in_channels2 = 64
out_channels2 = 128
kernel_size2 = 4
stride2 = 2
out_features3 = 256

# agent
batch_size = 256
buff_maxlen = 100_000
q_lr = 0.1
discount_gamma = 0.99
polyak_tau = 0.005
greedy_eps = 0.01
enable_cuda = False  # TODO: get working on CUDA
grad_clip_radius = None

# training
num_episodes = 50
episode_length = 20
make_plot = True


def tensor(x, cuda=enable_cuda):
    """Convert numpy array to torch tensor on desired device."""
    device = torch.device('cuda') if cuda else torch.device('cpu')
    x = x.copy() if isinstance(x, np.ndarray) else x
    return torch.tensor(
        x, dtype=torch.float32).to(device).unsqueeze(dim=0)


if __name__ == "__main__":

    env = BoardEnv(starts, targets, obstacles, reward_fn,
                   agent_type=AgentForDQN,
                   neighborhood_radius=neighborhood_radius)
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

    if make_plot:
        pp = PdfPages('schedule.pdf')

    for ep in range(num_episodes):
        env.reset()
        rewards = []
        total_agent_hits = 0
        successes = 0
        for step in range(episode_length):
            state = tensor(env.state)
            action = learner.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            reward, hits, success = reward
            learner.update(tensor(reward).view(1,1),
                           tensor(next_state),
                           tensor(int(done)).view(1,1))
            rewards.append(reward)
            total_agent_hits += hits
            successes += success

            # only plot the last episode, after which we hope to be not dumb
            if (ep == num_episodes - 1) and make_plot:
                # put plot callback here
                plot(env, pad=5)
                pp.savefig()

        print(f'Episode {ep}: average reward {np.mean(rewards)}, ',
              f'total hits {total_agent_hits}, successes {successes}')

    pp.close()
