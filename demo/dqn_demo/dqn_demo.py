import numpy as np

import src.dqn.trainer as trainer


# trainer specification
dqn_trainer_config = 'dqn_config.yml'

# training specification
num_episodes = 10
episode_length = 50
make_plot = True
plot_filename = 'plot.pdf'

# checkpointing info
checkpoint_filename = 'checkpoint.pt'
continue_training = True

# environment specification
starts = np.array([[0, 0]])
targets = np.array([[5, 5]])
obstacles = np.array([[]])
env_tuple = (starts, targets, obstacles)



if __name__ == '__main__':

    dqn_trainer1 = trainer.DQNTrainer(dqn_trainer_config)
    dqn_trainer1.train(num_episodes, episode_length, env_tuple)
    print('Training completed, saving checkpoint...')
    dqn_trainer1.save_checkpoint(checkpoint_filename)

    del dqn_trainer1

    dqn_trainer2 = trainer.DQNTrainer(dqn_trainer_config)
    print('Loading checkpoing...')
    dqn_trainer2.load_checkpoint(checkpoint_filename,
                                 continue_training)
    print('Resuming training...')
    dqn_trainer2.train(num_episodes, episode_length, env_tuple)
    print('Generating plotting data...')
    dqn_trainer2.plot(episode_length, plot_filename)
