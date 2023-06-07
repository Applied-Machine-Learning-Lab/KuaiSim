from utils.utils import *


class ValueModel(nn.Module):
    def __init__(self, dim_user_state_action=118 + 23+1, dim_hidden=256, dim_out=1, activation=nn.LeakyReLU):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_user_state_action, dim_hidden),
            activation(),
            nn.Linear(dim_hidden, dim_out)
        )

        self.model.apply(init_weight)

    def forward(self, x):
        value = self.model(x)
        return value
