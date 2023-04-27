import numpy as np


class QTable:
    """Simple Q-table"""

    def __init__(self, observation_space_n, action_space_n, max_val=0.0) -> None:
        """Initialize Q-table"""
        self.observation_space_n = observation_space_n
        self.action_space_n = action_space_n

        if max_val == 0.0:
            self.q_table = np.zeros(observation_space_n + (action_space_n,))
        else:
            self.q_table = np.random.uniform(
                0.0, max_val, observation_space_n + (action_space_n,)
            )
