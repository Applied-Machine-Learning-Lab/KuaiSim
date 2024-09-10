import random
from collections import namedtuple

# Taken from

Trajectory = namedtuple('Trajectory', ('state', 'action', 'mask'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a (state, action) pair."""
        self.memory.append(Trajectory(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Trajectory(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Trajectory(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
