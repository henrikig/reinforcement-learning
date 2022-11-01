import numpy as np
from spaces.space import Space


class Discrete(Space):
    def __init__(self, n: int, start: int = 0):
        assert isinstance(n, (int, np.integer))
        assert n > 0, "n (counts) have to be positive"
        assert isinstance(start, (int, np.integer))

        self.n = int(n)
        self.start = int(start)

    def sample(self):
        return int(self.start + np.random.randint(self.n))

    def contains(self, x) -> bool:
        return self.start <= x < self.start + self.n

    def __repr__(self) -> str:
        if self.start != 0:
            return f"Discrete({self.n}, start={self.start})"
        return f"Discrete({self.n})"
