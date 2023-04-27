import numpy as np
from spaces.space import Space
import random


class Continous(Space):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        assert len(low) == len(high), "Low and high need to be of same dimensions"
        self.n = len(low)

    def shape(self):
        return self.low.shape

    def sample(self):
        return np.array(
            [self.low[i] + random.random() * self.high[i] for i in range(len(self.n))]
        )

    def contains(self, x) -> bool:
        return all([self.low[i] <= x[i] <= self.high[i] for i in range(len(x))])

    def __repr__(self) -> str:
        if self.start != 0:
            return f"Continous({self.n}, start={self.start})"
        return f"Discrete({self.n})"
