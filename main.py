import numpy as np
from environment import cart_pole_env
from actors.sarsa import SARSA
from q_tables.tiled_q_table import TiledQTable
import time


if __name__ == "__main__":
    alpha = 0.05
    gamma = 0.999
    lambda_e = 0.92
    high = np.array((2.8, 3.0, 0.25, 4.0))
    n_bins = 8
    n_grids = 3
    num_episodes = 2000
    positive_reward = 0.1
    negative_reward = -1
    epsilon = 1
    min_epsilon = 0.01
    epsilon_decay_rate = 0.997
    max_initial_q_val = 0.0

    env = cart_pole_env.CartPoleEnv()
    initial_state = env.reset()

    bounds = [-high, high]
    tq = TiledQTable.cart_pole_q_table(
        bounds, env.action_space, n_bins, n_grids, max_val=max_initial_q_val
    )
    agent = SARSA(
        env,
        tq,
        alpha=alpha,
        gamma=gamma,
        lambda_e=lambda_e,
        num_episodes=num_episodes,
        epsilon=epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay_rate=epsilon_decay_rate,
        positive_reward=positive_reward,
        negative_reward=negative_reward,
    )
    t1 = time.time()
    agent.train()
    t2 = time.time()
    print(f"\nIt took {t2-t1} s")
    agent.plot_scores()
