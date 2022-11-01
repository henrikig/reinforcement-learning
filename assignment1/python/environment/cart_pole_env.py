import math
import numpy as np
from environment.env import Env
from spaces.continous import Continous
from spaces.discrete import Discrete


class CartPoleEnv(Env):
    def __init__(self, max_steps=300):
        super().__init__(max_steps)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # half of pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = Discrete(2)
        # TODO: how to represent this?
        self.observation_space = Continous(-high, high)

        self.default_range = np.random.default_rng()

    def step(self, action):
        assert self.action_space.contains(
            action
        ), "Action not defined. Please choose from {0, 1}."

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Update state variables
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        lost = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not lost:
            reward = 1.0
        else:
            reward = -1.0
        self.increment_step()
        done = lost or self.current_step >= self.max_steps

        self.state = np.array([x, x_dot, theta, theta_dot])

        return np.array(self.state, dtype=np.float32), reward, done

    def reset(self):
        super().reset()
        initial_state = [
            0.0,
            0.0,
            self.default_range.uniform(low=-0.21, high=0.21),
            0.0,
        ]
        self.state = np.array(initial_state)
        return self.state

    def get_possible_actions(self, _):
        return [action for action in range(self.action_space.n)]
