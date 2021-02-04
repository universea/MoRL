import numpy as np
import paddle
import morl
from paddle.distribution import Categorical

__all__ = ['PolicyGradient']


class PolicyGradient(morl.Algorithm):
    def __init__(self, model, lr):
        """Policy gradient algorithm

        Args:
            model (morl.Model): model defining forward network of policy.
            lr (float): learning rate.

        """
        assert isinstance(lr, float)
        self.model = model
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.parameters())

    def predict(self, obs):
        """Predict the probability of actions

        Args:
            obs (paddle tensor): shape of (obs_dim,)

        Returns:
            prob (paddle tensor): shape of (action_dim,)
        """
        prob = self.model(obs)
        return prob

    def learn(self, obs, action, reward):
        """Update model with policy gradient algorithm

        Args:
            obs (paddle tensor): shape of (batch_size, obs_dim)
            action (paddle tensor): shape of (batch_size, 1)
            reward (paddle tensor): shape of (batch_size, 1)

        Returns:
            loss (paddle tensor): shape of (1)

        """
        prob = self.model(obs)
        log_prob = Categorical(prob).log_prob(action)
        loss = paddle.mean(-1 * log_prob * reward)

        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()
        return loss
