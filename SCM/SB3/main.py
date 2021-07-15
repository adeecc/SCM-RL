import numpy as np
import pandas as pd

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from environment import SimpleSupplyChain

env = SimpleSupplyChain() 

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


agent = TD3("MlpPolicy", env, action_noise=action_noise, gamma=0.95, verbose=1, tensorboard_log="./tensorboard/td3_test_1")
agent.learn(total_timesteps=5e5, log_interval=10)
agent.save("td3_test_1")

# agent = TD3.load("td3_test_1")

obs = env.reset()
total_reward = 0
done = False

transitions = []

t = 0
while not done:
    action, _states = agent.predict(obs)
    obs, rewards, done, info = env.step(action)
    total_reward += rewards

    print(f"{obs = }")
    print(f"{action = }")
    print(f"{rewards = }")

    transitions.append([t, obs[0], obs[1], obs[2], *action, rewards, total_reward])
    t += 1

df = pd.DataFrame(transitions, columns=['t', 'factory_stock', 'warehouse_stock_0', 'warehouse_stock_1', 'production_level', 'shipping_to_warehouse_0', 'shipping_to_warehouse_1', 'timestep_reward', 'total_reward'])
df.to_csv("transitions.csv", index=False)

print(f"Reward during inference: {total_reward}")
