import gym
import argparse
import numpy as np
from morl.utils import logger, tensorboard, ReplayMemory
from cartpole_model import CartpoleModel
from cartpole_agent import CartpoleAgent
from continuous_cartpole_env import ContinuousCartPoleEnv
from morl.algorithms import SAC
import os

WARMUP_STEPS = 1e4
EVAL_EPISODES = 5
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.001
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3


# Run episode for training
def run_train_episode(agent, env, rpm):
    action_dim = env.action_space.shape[0]
    obs = env.reset()
    done = False
    episode_reward, episode_steps = 0, 0
    while not done:
        episode_steps += 1
        # Select action randomly or according to policy
        if rpm.size() < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            action = agent.sample(obs)
        # Perform action
        next_obs, reward, done, _ = env.step(action[0])
        terminal = float(done) if episode_steps < env._max_episode_steps else 0

        # Store data in replay memory
        rpm.append(obs, action, reward, next_obs, terminal)

        obs = next_obs
        episode_reward += reward

        # Train agent after collecting sufficient data
        if rpm.size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

    return episode_reward, episode_steps


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, eval_episodes, render):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action[0])
            avg_reward += reward
            if render:
                env.render()
    avg_reward /= eval_episodes
    return avg_reward


def main():
    logger.info("------------------- SAC ---------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")
    logger.set_dir('./{}_{}'.format(args.env, args.seed))

    env = ContinuousCartPoleEnv()#gym.make(args.env)
    env.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print('obs_dim, action_dim',(obs_dim,action_dim))

    # Initialize model, algorithm, agent, replay_memory
    model = CartpoleModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=args.alpha,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = CartpoleAgent(algorithm)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

    # load model and evaluate
    if os.path.exists('sac_model.ckpt'):
        logger.info("restore model succeed and test !")
        agent.restore('sac_model.ckpt')
        run_evaluate_episodes(agent, env, EVAL_EPISODES, render=True)
        exit()

    total_steps = 0
    test_flag = 0
    while total_steps < args.train_total_steps:
        # Train episode
        episode_reward, episode_steps = run_train_episode(agent, env, rpm)
        total_steps += episode_steps

        tensorboard.add_scalar('train/episode_reward', episode_reward,
                               total_steps)
        logger.info('Total Steps: {} Reward: {}'.format(
            total_steps, episode_reward))

        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(agent, env, EVAL_EPISODES, render=False)
            tensorboard.add_scalar('eval/episode_reward', avg_reward,
                                   total_steps)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, avg_reward))
    agent.save('sac_model.ckpt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="CartPole-v1", help='Cartpole gym environment name')
    parser.add_argument("--seed", default=0, type=int, help='Sets Gym seed')
    parser.add_argument(
        "--train_total_steps",
        default=3e6,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(5e3),
        help='The step interval between two consecutive evaluations')
    parser.add_argument(
        "--alpha",
        default=0.2,
        type=float,
        help=
        'Determines the relative importance of entropy term against the reward'
    )
    args = parser.parse_args()

    main()
