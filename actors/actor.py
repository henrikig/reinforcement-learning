from abc import ABC, abstractmethod


class Actor(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @property
    @abstractmethod
    # The Q-table is problem-dependent, and it it the task of the
    # implemetors of the Actor class to define this
    def Q_Table(self):
        raise NotImplementedError
