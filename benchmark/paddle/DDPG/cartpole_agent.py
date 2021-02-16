import morl
import paddle
import numpy as np


class CartpoleAgent(morl.Agent):
    def __init__(self, algorithm, act_dim, expl_noise=0.1):
        assert isinstance(act_dim, int)
        super(CartpoleAgent, self).__init__(algorithm)

        self.act_dim = act_dim
        self.expl_noise = expl_noise
        self.alg.sync_target(decay=0)

    def predict(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1).astype(np.float32))
        action = self.alg.predict(obs)
        action_numpy = action.cpu().numpy()[0]
        action_numpy = (action_numpy).clip(-1, 1)
        return action_numpy

    def sample(self, obs):
        action_numpy = self.predict(obs)
        action_noise = np.random.normal(0, self.expl_noise, size=self.act_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs)
        action = paddle.to_tensor(action)
        reward = paddle.to_tensor(reward)
        next_obs = paddle.to_tensor(next_obs)
        terminal = paddle.to_tensor(terminal)
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss