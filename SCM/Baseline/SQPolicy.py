import numpy as np
from Environment import Action, State

# We use (s, Q)-policy as a baseline
# => Order your Economic Order Quantity Q, every time your inventory
# position drops below s (Reorder Point or Safety Stock).


class SQPolicy(object):
    def __init__(self, factory_safety_stock, factory_reorder_amount, safety_stock, reorder_amount) -> None:
        self.factory_safety_stock = factory_safety_stock
        self.factory_reorder_amount = factory_reorder_amount
        self.safety_stock = safety_stock
        self.reorder_amount = reorder_amount

    def select_action(self, state: State):
        action = Action(state.warehouse_num)

        for w in range(state.warehouse_num):
            if state.warehouse_stock[w] < self.safety_stock[w]:
                action.shippings_to_warehouses[w] = self.reorder_amount[w]

        if state.factory_stock - np.sum(action.shippings_to_warehouses) < self.factory_safety_stock:
            action.production_level = self.factory_reorder_amount
        else:
            action.production_level = 0

        return action


def simulate_episode(env, policy):
    state = env.initial_state()
    transitions = []
    done = False
    # for t in range(env.T):
    while not done:
        action = policy.select_action(state)
        state, reward, done = env.step(state, action)
        transitions.append([state, action, reward])

    return transitions

def simulate(env, policy, num_episodes):
    returns_trace = []
    for episode in range(num_episodes):
        env.reset()
        returns_trace.append(sum(np.array(simulate_episode(env, policy)).T[2]))

    return returns_trace


