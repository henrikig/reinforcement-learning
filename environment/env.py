class Env:
    """Base class defining interface for the environments implemented"""

    def __init__(self, max_steps) -> None:
        self.max_steps = max_steps
        self.current_step = 0

    def increment_step(self):
        self.current_step += 1

    def step():
        raise NotImplementedError

    def reset(self):
        self.current_step = 0

    def get_possible_actions(self, state):
        raise NotImplementedError
