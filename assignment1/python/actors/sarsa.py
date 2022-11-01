import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import PyQt5

from environment.env import Env
from q_tables.tiled_q_table import TiledQTable


class SARSA:
    """SARSA Agent working on a continous space by dicretizing it."""

    def __init__(
        self,
        env: Env,
        tq: TiledQTable,
        alpha=0.05,
        gamma=0.95,
        lambda_e=0.92,
        epsilon=0.05,
        min_epsilon=0.05,
        epsilon_decay_rate=0.995,
        num_episodes=5000,
        positive_reward=0.1,
        negative_reward=-1,
    ) -> None:
        self.env = env
        self.tq = tq
        self.state_sizes = self.tq.state_sizes
        self.action_size = self.env.action_space.n
        self.scores = []

        self.alpha = alpha
        self.gamma = gamma
        self.lambda_e = lambda_e
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.num_episodes = num_episodes
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

    def train(self):
        """Train agent in environment by looping through different episodes"""
        self.scores = []
        max_avg_score = -np.inf

        for i_episode in range(1, self.num_episodes + 1):
            state, action, q_val = self.reset_episode()

            done = False
            while not done:
                new_state, reward, done = self.env.step(action)
                new_action, new_q_val = self.epsilon_greedy_policy(new_state)

                # Add ability for custom rewards
                if reward > 0:
                    reward = self.positive_reward
                else:
                    reward = self.negative_reward

                error = reward + self.gamma * new_q_val - q_val
                self.replace_eligibility(state, action)

                # Update Q-values and eligibilities for all <state, action>-pairs
                for q_table in self.tq.q_tables:
                    q_table.q_table = (
                        q_table.q_table + self.alpha * error * self.eligibilities
                    )
                self.eligibilities = self.gamma * self.lambda_e * self.eligibilities
                # self.eligibilities[self.eligibilities < 0.1] = 0

                state, action, q_val = new_state, new_action, new_q_val

            self.scores.append(self.env.current_step)

            if len(self.scores) > 100:
                avg_score = np.mean(self.scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                self.print_score(max_avg_score, i_episode)

    def reset_episode(self):
        # Decay epsilon
        if self.epsilon != self.min_epsilon:
            self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)
        # Reset eligibilities
        self.eligibilities = np.zeros_like(self.tq.q_tables[0].q_table)
        # Initialize state and action
        state = self.env.reset()
        action, q_val = self.epsilon_greedy_policy(state)
        return state, action, q_val

    def epsilon_greedy_policy(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.env.get_possible_actions(state))
            value = self.tq.get(state, action)
        else:
            action, value = self.get_max_action_and_value(state)
        return action, value

    def get_max_action_and_value(self, state):
        possible_actions = self.env.get_possible_actions(state)
        max_action, max_value = self.tq.get_max_action(state, possible_actions)
        return max_action, max_value

    def replace_eligibility(self, state, action):
        encoded_state = TiledQTable.tile_encode(state, self.tq.tilings)
        for state in encoded_state:
            self.eligibilities[tuple(state + (action,))] = 1

    def plot_scores(self, rolling_window=100):
        """Plot scores and optional rolling mean using specified window."""
        plt.plot(self.scores)
        plt.title("Scores")
        rolling_mean = pd.Series(self.scores).rolling(rolling_window).mean()
        for i in range(1, rolling_window + 1):
            rolling_mean[i] = sum(self.scores[:i]) / i
        plt.plot(rolling_mean)
        plt.show()

    def print_score(self, max_avg_score, i_episode):
        print(
            "\rEpisode {}/{} | Max Average Score: {}".format(
                i_episode, self.num_episodes, max_avg_score
            ),
            end="",
        )
        sys.stdout.flush()
