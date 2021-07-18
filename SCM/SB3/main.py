from agents import train_ddpg, train_td3, load_ddpg, load_td3, test_agent
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", palette="deep")


def main(algorithm="TD3", train=False, timesteps=5e5, filename=None, num_episodes=100):
    print("Creating Agent...")

    if train:
        if algorithm == "TD3":
            agent = train_td3(timesteps, net_architecture=[300, 400])
        elif algorithm == "DDPG":
            agent = train_ddpg(timesteps)
        else:
            print("Invalid Algorithm Entered")
            return

    else:
        if algorithm == "TD3":
            agent = load_td3(filename)
        elif algorithm == "DDPG":
            agent = load_ddpg(filename)
        else:
            print("Invalid Algorithm Entered")
            return

    print(f"Starting testing for {num_episodes}...")
    mean_reward, total_rewards, _ = test_agent(
        agent, log=True, num_episodes=100)
    std_dev = np.std(total_rewards)

    print(
        f">> Reward Mean: {mean_reward} [std. dev: {std_dev}] (episodes: {num_episodes}")

    plt.plot(total_rewards)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Python Script to Train, Run and Load RL Algorithms on a Supply Chain Environment")

    parser.add_argument(
        "-a", "--agent", default="TD3", help="Which Agent to use on the model")
    parser.add_argument(
        "-t", "--train", help="Train the model if argument is set", action="store_true")
    parser.add_argument("-s", "--steps", type=int, default=5e5,
                        help="Number of timesteps to train the model for")
    parser.add_argument("-f", "--file", help="Name of File to load")
    parser.add_argument("-e", "--episodes", type=int, default=100,
                        help="Number of episodes to test the model on")

    args = parser.parse_args()

    main(algorithm=args.agent, train=args.train, timesteps=args.steps,
         filename=args.file, num_episodes=args.episodes)
