import numpy as np
from actors.actor import Actor
from sklearn.preprocessing import KBinsDiscretizer

from environment.cart_pole_env import CartPoleEnv


class CartPoleActor(Actor):
    def __init__(
        self, env: CartPoleEnv, bins=(20, 20, 20, 20), bounds=(3.0, 3.0, 0.4, 4.0)
    ):
        self.bins = bins
        self.high = np.array(bounds, dtype=np.float32)
        self.low = -np.array(bounds, dtype=np.float32)

        self.discretizer = KBinsDiscretizer(
            n_bins=self.bins, encode="ordinal", strategy="uniform"
        )
        self.discretizer.fit([self.low, self.high])

        self._Q_Table = np.random.uniform(0.0, 0.05, bins + (env.action_space.n,))
        self.env = env

    def discretize(self, state):
        return tuple(map(int, self.discretizer.transform([state])[0]))

    def train(self):
        epsilon = 0.05
        episodes = 3000
        learning_rate = 0.9
        discount = 0.95

        # Repeat for each episode
        counter_list = []

        for episode in range(1, episodes + 1):
            # Reset eligibilities e(s, a) <- 0 Vs, a
            # initialize s <- s_init, a <- policy(s_init)
            state = self.env.reset()
            action, q_val = self.get_policy_action(state)
            # Repeat for each step
            done = False
            counter = 0
            while not done:
                # 1. do action a from state s to obtain s' and reward
                new_state, reward, done = self.env.step(action)
                # 2. a' <- policy(s')
                new_action, new_q_val = self.get_policy_action(
                    new_state, epsilon=epsilon, episode=episode
                )
                # 3. error <- r + discount * Q(s', a'), - Q(s, a)
                error = reward + discount * new_q_val - q_val
                # 4. e(s, a) <- 1
                # 5. for V(s, a) in E
                # a. Q(s, a) <- Q(s, a) + alpha * error * e(s, a)
                # Update Q-table
                discrete_state = self.discretize(state)
                self.Q_Table[discrete_state][action] = q_val + learning_rate * error
                # b. e(s, a) <- discount * lambda * e(s, a)
                # 6. s <- s', a <- a'
                state = new_state
                action = new_action
                q_val = new_q_val
                counter += 1
            counter_list.append(counter)
            if len(counter_list) > 100:
                counter_list.pop(0)
                if episode % 100 == 0:
                    print(
                        f"Episode {episode} avg: {sum(counter_list)/len(counter_list)}"
                    )
        # until end state

    def test(self):
        state = self.env.reset()
        action, _ = self.get_policy_action(state, epsilon=0.0)
        # Repeat for each step
        done = False
        counter = 0
        while not done:
            new_state, _, done = self.env.step(action)
            new_action, _ = self.get_policy_action(new_state, epsilon=0.0)
            state = new_state
            action = new_action
            counter += 1

        print(f"Test made {counter} iterations")

    def get_policy_action(self, state, epsilon=0.05, episode=None):
        discrete_state = self.discretize(state)
        possible_actions = self.Q_Table[discrete_state]
        epsilon = epsilon / episode if episode is not None else epsilon
        if self.env.default_range.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(possible_actions)

        return action, possible_actions[action]

    @property
    def Q_Table(self):
        return self._Q_Table
