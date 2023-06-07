from utils.OU_noise import OUNoise
from utils.utils import *


class DiscriminatorModel(nn.Module):
    def __init__(self, n_input=118 + 23+1+1, n_hidden=256, n_output=1, activation=nn.LeakyReLU):
        super(DiscriminatorModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_output),
            nn.Dropout(p=0.6),
            nn.Sigmoid()
        )

        self.Noise = OUNoise(n_input)
        self.model.apply(init_weight)

    def forward(self, x):
        noise = torch.zeros_like(x)
        for i in range(noise.size(0)):
            noise[i] += FLOAT(self.Noise.sample()).to(device)
        x += noise
        return self.model(x)
