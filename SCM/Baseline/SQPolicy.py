import numpy as np
import pandas as pd
import time
from Environment import Action, State, SupplyChainEnvironment

# We use (s, Q)-policy as a baseline
# => Order your Economic Order Quantity Q, every time your inventory
# position drops below s (Reorder Point or Safety Stock).


class SQPolicy(object):
    def __init__(self, factory_safety_stock, factory_reorder_amount, safety_stock, reorder_amount) -> None:
        self.factory_safety_stock = factory_safety_stock
        self.factory_reorder_amount = factory_reorder_amount
        self.safety_stock = safety_stock
        self.reorder_amount = reorder_amount

    def select_action(self, state: State) -> Action:
        action = Action(state.warehouse_num)

        for w in range(state.warehouse_num):
            if state.warehouse_stock[w] < self.safety_stock[w]:
                action.shippings_to_warehouses[w] = self.reorder_amount[w]

        if state.factory_stock - np.sum(action.shippings_to_warehouses) < self.factory_safety_stock:
            action.production_level = self.factory_reorder_amount
        else:
            action.production_level = 0

        return action


def simulate_episode(policy: SQPolicy, log=False) -> list:
    env = SupplyChainEnvironment()
    state = env.initial_state()
    transitions = []
    expanded_data = []
    done = False
    for t in range(env.T):
        # while not done:
        action = policy.select_action(state)
        state, reward, done = env.step(state, action, log)
        transitions.append([state, action, reward])
        expanded_data.append([state.t, state.warehouse_num,
                             state.factory_stock,
                             state.warehouse_stock[0], state.warehouse_stock[1],
                             action.production_level,
                             action.shippings_to_warehouses[0], action.shippings_to_warehouses[1],
                             reward])

    if log:
        df = pd.DataFrame(expanded_data, columns=['t', 'warehouse_num', 'factory_stock',
                      'warehouse_stock_0', 'warehouse_stock_1', 'production_level', 'shippings_to_warehouses_0', 'shippings_to_warehouses_1', 'reward'])

        df.to_csv("transitions.csv", index=False)

    return transitions


def simulate(policy: SQPolicy, num_episodes: int, log=False) -> list:
    returns_trace = []
    for episode in range(num_episodes):
        returns_trace.append(sum(np.array(simulate_episode(policy, log)).T[2]))

    return returns_trace
