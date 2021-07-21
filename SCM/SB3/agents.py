import numpy as np
import pandas as pd
from stable_baselines3.td3.policies import TD3Policy
import torch as th
import torch.nn as nn
import time

from stable_baselines3 import TD3, DDPG
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch.nn.modules import activation

from environment import SimpleSupplyChain


def make_policy(net_architecture=[512, 512]):
    policy = MlpPolicy(observation_space=SimpleSupplyChain.observation_space,
                       action_space=SimpleSupplyChain.action_space,
                       lr_schedule=0.001,
                       net_arch=net_architecture)

    return policy

# Reward Mean: 7041.430871062279 [std. dev: 465.25285975768423] (episodes: 1000)


def train_td3(timesteps=5e5, net_architecture=None):
    print("Created Environment...")
    env = SimpleSupplyChain()

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(
        n_actions), sigma=0.1 * np.ones(n_actions))

    if net_architecture:
        policy_kwargs = {
            # "activation_fn": th.nn.ReLU,
            "net_arch": net_architecture
        }
    else:
        policy_kwargs = {}

    agent = TD3(policy="MlpPolicy", env=env,
                action_noise=action_noise, policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log="./tensorboard/TD3")

    print("Starting Model Training...")
    agent.learn(total_timesteps=timesteps, log_interval=10)

    file_name = f"td3_{int(time.time())}"
    agent.save(file_name)

    print(f"Training Finished. Model saved as >>> {file_name}")
    return agent


# Reward Mean: 6705.04089457035 [std. dev: 366.5834462025138] (episodes: 1000)
def train_ddpg(timesteps=5e5):
    print("Created Environment...")
    env = SimpleSupplyChain()

    n_actions = env.action_space.shape[-1]

    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(
        n_actions), sigma=0.1 * np.ones(n_actions))

    agent = DDPG("MlpPolicy", env, action_noise=action_noise,
                 verbose=1, tensorboard_log="./tensorboard/DDPG")

    print("Starting Model Training...")
    agent.learn(total_timesteps=timesteps, log_interval=10)

    file_name = f"ddpg_{int(time.time())}"
    agent.save(file_name)

    print(f"Training Finished. Model saved as >>> {file_name}")
    return agent


def load_td3(file_name):
    agent = TD3.load(file_name)
    return agent


def load_ddpg(file_name):
    agent = DDPG.load(file_name)
    return agent


def test_agent(agent, log=False, num_episodes=10):
    env = SimpleSupplyChain()

    total_rewards = []
    transitions = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False

        t = 0

        while not done:
            action, _states = agent.predict(obs)
            obs, reward, done, info = env.step(action)

            total_reward += reward

            if log:
                transitions.append(
                    [episode, t, obs[0], obs[1], obs[2], *action, reward, total_reward])
            t += 1

        total_rewards.append(total_reward)

    if log:
        df = pd.DataFrame(transitions, columns=['episode', 't', 'factory_stock', 'warehouse_stock_0', 'warehouse_stock_1',
                                                'production_level', 'shipping_to_warehouse_0', 'shipping_to_warehouse_1', 'timestep_reward', 'total_reward'])
        df.to_csv(
            f"transitions_{num_episodes}_{int(time.time())}.csv", index=False)

    mean_reward = np.mean(total_rewards)
    return mean_reward, total_rewards, num_episodes

