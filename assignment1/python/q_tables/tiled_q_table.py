import numpy as np
from spaces.discrete import Discrete

from q_tables.q_table import QTable


class TiledQTable:
    """Composite Q-table with an internal tile coding scheme"""

    def __init__(self, low, high, tiling_specs, action_size, max_val=0.0):
        self.tilings = TiledQTable.create_tilings(low, high, tiling_specs)
        # The number of tiles in each dimension for each tiling
        self.state_sizes = [
            tuple(len(tiles) + 1 for tiles in tiling_grid)
            for tiling_grid in self.tilings
        ]
        self.action_size = action_size
        self.q_tables = [
            QTable(state_size, self.action_size, max_val=max_val)
            for state_size in self.state_sizes
        ]

    def get(self, state, action):
        """Get Q-value for given <state, action> pair, averaged over all tiles"""
        q_vals = 0.0
        encoded_state = TiledQTable.tile_encode(state, self.tilings)
        for state, table in zip(encoded_state, self.q_tables):
            q_vals += table.q_table[state + (action,)]
        return q_vals / len(self.q_tables)

    def get_max_action(self, state, possible_actions):
        """Get maximizing action for given state, averaged over all tiles"""
        q_vals = np.zeros((len(possible_actions)))
        encoded_state = TiledQTable.tile_encode(state, self.tilings)
        for state, table in zip(encoded_state, self.q_tables):
            for i, action in enumerate(possible_actions):
                q_vals[i] += table.q_table[tuple(state + (action,))]
        # Choose random action among the best actions
        max_action = np.random.choice(np.flatnonzero(q_vals == q_vals.max()))
        return max_action, q_vals[max_action]

    def update(self, state, action, value, alpha=0.1):
        """Soft-update Q-value for given <state, action> pair to value."""
        state = TiledQTable.tile_encode(state, self.tilings)
        for state, table in zip(state, self.q_tables):
            table.q_table[tuple(state + (action,))] = (
                alpha * value + (1.0 - alpha) * table.q_table[state][action]
            )

    def update_all_with_eligibilities(self, td_error, eligibilities, alpha=0.1):
        for table in self.q_tables:
            table.q_table = table.q_table + alpha * td_error * eligibilities

    @staticmethod
    def create_tiling_grid(low, high, bins, offsets):
        """Define a uniformly-spaced grid that can be used for tile-coding a space."""
        grids = []
        for i in range(len(low)):
            grid = (
                np.linspace(low[i], high[i], num=bins[i] + 1, endpoint=True)[1:-1]
                + offsets[i]
            )
            grids.append(grid)
        return grids

    @staticmethod
    def create_tilings(low, high, tiling_specs):
        """Define multiple tilings using the provided specifications."""
        return [
            TiledQTable.create_tiling_grid(low, high, *specs) for specs in tiling_specs
        ]

    @staticmethod
    def discretize(sample, grid):
        """Discretize a sample as per given grid."""
        return tuple(np.searchsorted(g, s, side="right") for s, g in zip(sample, grid))

    @staticmethod
    def tile_encode(sample, tilings, flatten=False):
        """Encode given sample using tile-coding."""
        encoded_sample = [TiledQTable.discretize(sample, grid) for grid in tilings]
        return np.concatenate(encoded_sample) if flatten else encoded_sample

    @staticmethod
    def cart_pole_q_table(
        bounds, action_space: Discrete, n_bins=6, n_grids=3, max_val=0.0
    ):
        bins = tuple([n_bins] * bounds[0].shape[0])
        # Space grids evenly
        offset_pos = (bounds[1] - bounds[0]) / (n_grids * n_bins)

        tiling_specs = [
            (bins, -offset_pos),
            (bins, tuple([0.0] * bounds[0].shape[0])),
            (bins, offset_pos),
        ]

        return TiledQTable(bounds[0], bounds[1], tiling_specs, action_space.n, max_val)
