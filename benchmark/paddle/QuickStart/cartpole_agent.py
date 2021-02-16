import morl
import paddle
import numpy as np


class CartpoleAgent(morl.Agent):
    """Agent of Cartpole env.

    Args:
        algorithm(morl.Algorithm): algorithm used to solve the problem.

    """

    def __init__(self, algorithm):
        super(CartpoleAgent, self).__init__(algorithm)

    def sample(self, obs):
        """Sample an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)
        
        Returns:
            act(int): action
        """
        obs = paddle.to_tensor(obs.astype(np.float32))
        prob = self.alg.predict(obs)
        prob = prob.numpy()
        act = np.random.choice(len(prob), 1, p=prob)[0]
        return act

    def predict(self, obs):
        """Predict an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)
        
        Returns:
            act(int): action
        """
        obs = paddle.to_tensor(obs.astype(np.float32))
        prob = self.alg.predict(obs)
        act = prob.argmax().numpy()[0]
        return act

    def learn(self, obs, act, reward):
        """Update model with an episode data

        Args:
            obs(np.float32): shape of (batch_size, obs_dim)
            act(np.int64): shape of (batch_size)
            reward(np.float32): shape of (batch_size)
        
        Returns:
            loss(float)

        """
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)

        obs = paddle.to_tensor(obs.astype(np.float32))
        act = paddle.to_tensor(act.astype(np.int32))
        reward = paddle.to_tensor(reward.astype(np.float32))

        loss = self.alg.learn(obs, act, reward)
        return loss.numpy()[0]
