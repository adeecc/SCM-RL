import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", palette="deep")

from agents import train_ddpg, train_td3, load_ddpg, load_td3, test_agent


def main(algorithm="TD3", train=False, timesteps=5e5, filename=None):
    if train:
        if algorithm == "TD3":
            agent = train_td3(timesteps, net_architecture=[512, 512])
        elif algorithm == "DDPG":
            agent = train_ddpg(timesteps)

    else:
        if algorithm == "TD3":
            agent = load_td3(filename)
        elif algorithm == "DDPG":
            agent = load_ddpg(filename)



    mean_reward, total_rewards, num_episodes = test_agent(agent, log=True, num_episodes=100)
    std_dev = np.std(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {mean_reward} [std. dev: {std_dev}]")

    plt.plot(total_rewards)
    plt.show()

if __name__ == '__main__':
    main(train=True)
