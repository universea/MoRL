import os
import gym
import numpy as np
import morl
from morl.utils import logger
from cartpole_model import CartpoleModel
from cartpole_agent import CartpoleAgent

LEARNING_RATE = 1e-3


# train an episode
def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# evaluate 5 episodes
def evaluate(env, agent, episode_num=5, render=False):
    eval_reward = []
    for i in range(episode_num):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


def main():
    env = gym.make('CartPole-v0')
    # env = env.unwrapped # Cancel the minimum score limit
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # build an agent
    model = CartpoleModel(obs_dim=obs_dim, act_dim=act_dim)
    alg = morl.algorithms.PolicyGradient(model, lr=LEARNING_RATE)
    agent = CartpoleAgent(alg)

    # load model and evaluate
    if os.path.exists('simple_model.ckpt'):
        logger.info("restore model succeed and test !")
        agent.restore('simple_model.ckpt')
        evaluate(env, agent, render=True)
        exit()

    for i in range(1000):
        obs_list, action_list, reward_list = run_episode(env, agent)
        if i % 10 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward = evaluate(env, agent, render=True)
            logger.info('Test reward: {}'.format(total_reward))

    # save the parameters to model.ckpt
    agent.save('simple_model.ckpt')


if __name__ == '__main__':
    main()
