import argparse

import gym
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import random

from tqdm import tqdm
from approx.gp_agent import GPAgent
from approx.CustomEnvironment import MountainCarEnv


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)


def train(env_name="MountainCar-v0", num_episodes=50, num_grid=10,
          epsilon=0.1, gamma=0.8, magnitude=1.0, length_scale=1.0, alpha=1e-2, num_opt_iter=1,
          print_every=1, plot_every=1, update_rate=0.1, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    # env = gym.make(env_name)
    env = MountainCarEnv()
    global diff
    diff = env.observation_space.high - env.observation_space.low
    agent = GPAgent(num_actions=env.action_space.n, epsilon=epsilon, gamma=gamma, magnitude=magnitude, length_scale=length_scale, alpha=alpha, num_opt_iter=num_opt_iter, update_rate=update_rate, random_seed=random_seed)
    time_list = []
    reward_list = []

    folder = f"gamma-length-rollout-seed/{gamma}-{length_scale}-{num_grid**2}-{random_seed}"
    os.makedirs(folder, exist_ok=True)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(f"{folder}/train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"gamma {gamma} | length_scale {length_scale} | num_opt_iter | {num_opt_iter}")
    logger.info(f"num_episodes {num_episodes} | magnitude {magnitude} | alpha {alpha} | td_update_rate {update_rate}")
    logger.info("-----------------------------------------------------------")
    logger.info("roll-out phase started...")
    roll_out(env, agent, num_grid=num_grid)
    plot_value_fn(env, agent, epoch=-1, step=None, folder=folder)
    logger.info("roll-out finished")

    logger.info("training phase started...")
    for epoch in range(num_episodes):
        logger.info("-----------------------------------------------------------")
        T, G, is_goal = run_episode(env, agent, epoch,
                                    # print_every_step=40 if num_opt_iter==0 else (20/num_opt_iter),
                                    print_every_step=40,
                                    # plot_every_step=100 if num_opt_iter==0 else (20/num_opt_iter),
                                    plot_every_step=100,
                                    folder=folder)
        agent.update_epsilon(epoch, num_episodes)
        if (epoch + 1) % print_every == 0:
            if is_goal:
                logger.info(f"  Return: {G}, Steps: {T}, Goal")  # todo: average G 를 출력하는 것으로
            else:
                logger.info(f"  Return: {G}, Steps: {T}")
        time_list += [T]
        reward_list += [G]
        if np.array(reward_list)[-10:].mean() >= 0.8:
            break
    logger.info("training finished")


def run_episode(env, agent, epoch, print_every_step=1, plot_every_step=50, folder="."):
    T = 0
    G = 0
    done = False
    state = env.reset()
    while not done:
        if T % print_every_step == 0:
            action, value_list = agent.get_action(state / diff, return_value=True)
        else:
            action = agent.get_action(state / diff)
        next_state, reward, done, _ = env.step(action)

        next_action = agent.get_action(next_state / diff)
        observation = (state / diff, action, reward, next_state / diff, next_action, done)
        agent.update(observation, T)

        G += reward
        T += 1
        is_goal = (reward==1.0)

        state = next_state
        if T % print_every_step == 0:
            logger.info(f"epoch {epoch+1:03d} | "
                  f"step {T:03d} | "
                  f"epsilon {agent.epsilon:.2f} | "
                  f"td_update_rate {agent.update_rate / T:.4f} | "
                  f"value {[round(value.item(),2) for value in value_list]} | "
                  f"loglike {[round(value_fn.log_marginal_likelihood_value_ / value_fn.X_train_.shape[0], 2) for value_fn in agent.value_fn_list]}"
                  f"")
        if T % plot_every_step == 0:
            plot_value_fn(env, agent, epoch, step=T, folder=folder)

    return T, G, is_goal


def roll_out(env, agent, num_grid=100):
    x1_low, x2_low = env.observation_space.low
    x1_high, x2_high = env.observation_space.high
    x1_grid = np.linspace(x1_low, x1_high, num_grid)
    x2_grid = np.linspace(x2_low, x2_high, num_grid)
    grid = [[x1_grid[i], x2_grid[j]] for i in range(num_grid) for j in range(num_grid)]
    observation = []
    for state in tqdm(grid):
        _ = env.reset()
        for action in range(3):
            env.state = state
            next_state, reward, done, _ = env.step(action)
            observation += [(np.array(state) / diff, action, reward)]
    agent.rollout_update(observation)


def plot_value_fn(env, agent, epoch, step, num_tiles=100, folder="."):
    plt.clf()
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda state: np.max([agent.value_fn_list[action].predict(state.reshape(1,-1) / diff) for action in range(agent.nA)]), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("\"Mountain Car\" Value Function")
    fig.colorbar(surf)
    plt.show()
    if epoch < 0:
        plt.savefig(f"{folder}/value-surface-roll-out.png")
    else:
        plt.savefig(f"{folder}/value-surface-epoch-{epoch:03d}-step-{step:03d}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", "-g", type=float, default=0.8)
    parser.add_argument("--num_grid", "-r", type=int, default=20)
    parser.add_argument("--length_scale", "-l", type=float, default=1.0)
    parser.add_argument("--num_opt_iter", "-o", type=int, default=0)
    parser.add_argument("--update_rate", "-u", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    train(gamma=args.gamma,
          num_grid=args.num_grid,
          length_scale=args.length_scale,
          num_opt_iter=args.num_opt_iter,
          update_rate=args.update_rate,
          random_seed=args.seed)
