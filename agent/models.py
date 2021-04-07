import torch
import torch.autograd as autograd
import torch.nn as nn

NUM_LOOKBACK_TIMESTEPS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Base(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.construct()

    def construct(self):
        raise NotImplementedError

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        return x.view(1, -1).size(1)

class DQN(Base):
    def construct(self):
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

class AtariDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

class ConvDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
        )
        super().construct()

class BetterDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

class RecurrentAtariDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )

        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        self.feature_size =  x.view(1, -1).size(1)

        self.recurrent = nn.LSTM(input_size=self.feature_size,hidden_size=512,num_layers=1,batch_first=True)
        
        self.layers = nn.Sequential(
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x, hidden_state, cell_state):
        if len(x.size()) == 4:
            n_batch = x.size(0)
            n_timestep = 1
        else:
            n_batch = x.size(0)
            n_timestep = x.size(1)
            x = x.view(n_batch*n_timestep, x.size(2), x.size(3), x.size(4))

        if hasattr(self, 'features'):
            x = self.features(x)
        
        x = x.view(n_batch, n_timestep, -1)
        x, (h_n, c_n) = self.recurrent(x, (hidden_state, cell_state))

        x = x[:,n_timestep-1,:]
        x = self.layers(x)
        return x, h_n, c_n

    def reset_hidden_states(self, n_batch):
        h = torch.zeros(1, n_batch, 512).float().to(device)
        c = torch.zeros(1, n_batch, 512).float().to(device)
        return h, c