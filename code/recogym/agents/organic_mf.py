import numpy as np
import torch

from ..envs.configuration import Configuration
from .abstract import Agent

# Default Arguments ----------------------------------------------------------
organic_mf_square_args = {
    'num_products': 10,
    'embed_dim': 5,
    'mini_batch_size': 32,
    'loss_function': torch.nn.CrossEntropyLoss(),
    'optim_function': torch.optim.RMSprop,
    'learning_rate': 0.01,
    'with_ps_all': False,
}


# Model ----------------------------------------------------------------------
class OrganicMFSquare(torch.nn.Module, Agent):
    """
    Organic Matrix Factorisation (Square)

    The Agent that selects an Action from the model that performs
     Organic Events matrix factorisation.
    """

    def __init__(self, config = Configuration(organic_mf_square_args)):
        torch.nn.Module.__init__(self)
        Agent.__init__(self, config)

        self.product_embedding = torch.nn.Embedding(
            self.config.num_products, self.config.embed_dim
        )
        self.output_layer = torch.nn.Linear(
            self.config.embed_dim, self.config.num_products
        )

        # Initializing optimizer type.
        self.optimizer = self.config.optim_function(
            self.parameters(), lr = self.config.learning_rate
        )

        self.last_product_viewed = None
        self.curr_step = 0
        self.train_data = []
        self.action = None

    def forward(self, product):

        product = torch.Tensor([product])

        a = self.product_embedding(product.long())
        b = self.output_layer(a)

        return b

    def act(self, observation, reward, done):
        with torch.no_grad():
            if observation is not None and len(observation.current_sessions) > 0:
                logits = self.forward(observation.current_sessions[-1]['v'])

                # No exploration strategy, choose maximum logit.
                self.action = logits.argmax().item()

            if self.config.with_ps_all:
                all_ps = np.zeros(self.config.num_products)
                all_ps[self.action] = 1.0
            else:
                all_ps = ()
            return {
                **super().act(observation, reward, done),
                **{
                    'a': self.action,
                    'ps': 1.0,
                    'ps-a': all_ps,
                }
            }

    def update_weights(self):
        """Update weights of embedding matrices using mini batch of data"""
        # Eliminate previous gradient.
        self.optimizer.zero_grad()

        for prods in self.train_data:
            # Calculating logit of action and last product viewed.

            # Loop over the number of products.
            for i in range(len(prods) - 1):

                logit = self.forward(prods[i]['v'])

                # Converting label into Tensor.
                label = torch.LongTensor([prods[i + 1]['v']])

                # Calculating supervised loss.
                loss = self.config.loss_function(logit, label)
                loss.backward()

        # Update weight parameters.
        self.optimizer.step()

    def train(self, observation, action, reward, done = False):
        """Method to deal with the """

        # Increment step.
        self.curr_step += 1

        # Update weights of model once mini batch of data accumulated.
        if self.curr_step % self.config.mini_batch_size == 0:
            self.update_weights()
            self.train_data = []
        else:
            if observation is not None:
                data = observation.current_sessions
                self.train_data.append(data)
