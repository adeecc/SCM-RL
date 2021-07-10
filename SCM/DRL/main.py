import numpy as np
import pandas as pd
import ray

from environment import SupplyChainEnvironment, Action
from models import train_ddpg, train_td3, load_policy
from utils import visualize_transitions


def test(log_dir, checkpoint_id, log=True):
    policy = load_policy(log_dir=log_dir, checkpoint_id=checkpoint_id)

    env = SupplyChainEnvironment()

    state = env.initial_state()
    transitions_rl = []
    expanded_data = []

    for t in range(env.T):
        action = policy.compute_single_action(state.to_array(), state=[])

        action_obj = Action(env.warehouse_num)
        action_obj.production_level = action[0][0]
        action_obj.shippings_to_warehouses = action[0][1:]
        state, reward, _ = env.step(state, action_obj, log=True)

        transitions_rl.append([state, action_obj, reward])
        expanded_data.append([state.t, state.warehouse_num,
                              state.factory_stock,
                              state.warehouse_stock[0], state.warehouse_stock[1],
                              action_obj.production_level,
                              action_obj.shippings_to_warehouses[0], action_obj.shippings_to_warehouses[1],
                              reward])

    if log:
        df = pd.DataFrame(expanded_data, columns=['t', 'warehouse_num', 'factory_stock',
                                                  'warehouse_stock_0', 'warehouse_stock_1', 'production_level', 'shippings_to_warehouses_0', 'shippings_to_warehouses_1', 'reward'])

        df.to_csv("transitions.csv", index=False)

    visualize_transitions(np.array(transitions_rl), T=env.T,
                          warehouse_num=env.warehouse_num)
    return transitions_rl


def main(train=True, model="td3", normalize_actions=False, num_epochs=500):
    ray.shutdown()
    ray.init()

    if train:
        if model == "ddpg":
            log_dir = train_ddpg(
                normalize_actions=normalize_actions, num_epochs=num_epochs)
        else:
            log_dir = train_td3(
                normalize_actions=normalize_actions, num_epochs=num_epochs)

        log_dir = '/'.join(log_dir.split('/')[:-2])
    else:
        log_dir = input(">>> Parameter Directory: ")

    checkpoint_id = input(">>> Checkpoint ID: ")
    test(log_dir, checkpoint_id)


if __name__ == "__main__":
    main(model="td3")
