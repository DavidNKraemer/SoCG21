import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalDQN(nn.Module):
    """
    Convolutional neural network for use as a deep Q network in a deep
    Q-learning algorithm.

    Network contains two convolutional layers followed by two fully
    connected layers with ReLU activation functions.
    """

    def __init__(self, in_channels, num_actions,
                 example_state_tensor,  # example state converted to torch tensor
                 out_channels1=16, kernel_size1=3, stride1=2,
                 in_channels2=16, out_channels2=32, kernel_size2=2, stride2=1,
                 out_features3=256):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size1, stride1)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size2, stride2)

        in_features3 = self.conv2(
            self.conv1(example_state_tensor.to(torch.device('cpu')))).view(-1).shape[0]

        self.fc3 = nn.Linear(in_features3, out_features3, bias=True)
        self.head = nn.Linear(out_features3, num_actions, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))  # flattens Conv2d output
        return self.head(x)
