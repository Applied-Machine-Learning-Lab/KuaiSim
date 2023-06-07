from utils.utils import *


class RewardModel(nn.Module):
    def __init__(self, n_input=118 + 23+1+1, n_hidden=256, n_output=1, activation=nn.LeakyReLU):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_output),
            nn.Sigmoid()
        )
        self.model.apply(init_weight)

    def forward(self, x):
        return self.model(x)
