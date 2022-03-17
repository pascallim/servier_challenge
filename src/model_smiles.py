from torch import nn, sigmoid
import torch.nn.functional as F

# To modify when there is a change in input data shape
INPUT_SIZE = 2048

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_size = INPUT_SIZE
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 124)
        self.fc3 = nn.Linear(124, 32)
        self.fc_output = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = sigmoid(self.fc_output(x))
        return x
