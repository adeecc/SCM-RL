import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt


import gym
from gym.spaces import Box

from utils import rescale


DEMAND_DATA_PATH = "../../data/demand.csv"
NUM_WAREHOUSES = 2
EPISODE_DURATION = 50
MAX_DEMAND = 5
UNIT_PRICE = 100
UNIT_COST = 40


class State(object):
    def __init__(self, warehouse_num, T, demand_history, t=0):
        self.warehouse_num = warehouse_num
        self.factory_stock = 0
        # np.repeat(0, warehouse_num)
        self.warehouse_stock = np.zeros(warehouse_num)
        self.demand_history = demand_history
        self.T = T
        self.t = t

    def to_array(self):
        return np.concatenate(([self.factory_stock], self.warehouse_stock, np.hstack(self.demand_history), [self.t]))

    def stock_levels(self):
        return np.concatenate(([self.factory_stock], self.warehouse_stock))


class Action(object):
    def __init__(self, warehouse_num):
        self.production_level = 0
        self.shippings_to_warehouses = np.zeros(warehouse_num)


class SupplyChainEnvironment(object):
    def __init__(self,
                 T=EPISODE_DURATION,
                 warehouse_num=NUM_WAREHOUSES,
                 d_max=MAX_DEMAND,
                 d_var=2,
                 unit_price=UNIT_PRICE,
                 unit_cost=UNIT_COST):

        self.T = T               # episode duration
        self.warehouse_num = warehouse_num
        self.d_max = d_max           # maximum demand, units
        self.d_var = d_var            # maximum random demand variation, units

        self.unit_price = unit_price     # unit price in dollars
        self.unit_cost = unit_cost       # unit cost in dollars

        demand_data = pd.read_csv(DEMAND_DATA_PATH)
        demand_data = demand_data.drop(columns=["Date"]).to_numpy()
        demand_data = rescale(demand_data, demand_range=(1, self.d_max))
        self.demand_data = demand_data

        self.demand_offset = np.random.randint(
            low=0, high=demand_data.shape[0])  # 420
        # print(f"demand_offset: {self.demand_offset}")

        # Storage Capacity of factory and each warehouse
        self.storage_capacities = np.fromfunction(
            lambda j: 10 * (j + 1), (self.warehouse_num + 1,), dtype=int)

        # storage costs at the factory and each warehouse, dollars per unit
        self.storage_costs = np.fromfunction(
            lambda j: 2 * (j + 1), (self.warehouse_num + 1,), dtype=int)
        # transportation costs for each warehouse, dollars per unit
        self.transporation_costs = np.fromfunction(
            lambda j: 5 * (j + 1), (self.warehouse_num,), dtype=int)
        self.penalty_unit_cost = self.unit_price

        self.reset()

    def reset(self, demand_history_len=4):
        self.demand_history = collections.deque(maxlen=demand_history_len)
        for _ in range(demand_history_len):
            self.demand_history.append(np.zeros(self.warehouse_num))
        self.t = 0

    # demand at time t at all warehouses
    def demand(self, t):
        tmod = (t + self.demand_offset) % self.demand_data.shape[0]
        return np.round(self.demand_data[tmod])

    def initial_state(self):
        return State(self.warehouse_num, self.T, list(self.demand_history))

    def step(self, state: State, action: Action, log=False):
        demands = self.demand(self.t)

        if log:
            df = pd.DataFrame([[self.t, demands[0], demands[1]]],
                              columns=['t', 'demand_0', 'demand_1'])

            df.to_csv(
                "demands.csv", mode='a', header=False, index=False)

        # Calculate returns (reward)
        revenue = self.unit_price * np.sum(demands)

        production_cost = self.unit_cost * action.production_level
        storage_cost = self.storage_costs @ np.maximum(
            state.stock_levels(), np.zeros(self.warehouse_num + 1))

        penalty_cost = -self.penalty_unit_cost * \
            (np.sum(np.minimum(state.warehouse_stock, np.zeros(
                self.warehouse_num))) + min(state.factory_stock, 0))

        transportation_cost = self.transporation_costs @ action.shippings_to_warehouses
        reward = revenue - production_cost - \
            storage_cost - penalty_cost - transportation_cost

        if log:
            df = pd.DataFrame([[self.t, revenue, production_cost, storage_cost, penalty_cost, transportation_cost, reward]],
                              columns=['t', 'revenue', 'production_cost', 'storage_cost', 'penalty_cost', 'transportation_cost', 'reward'])
            df.to_csv("costs.csv", mode='a', header=False, index=False)

        # Calculate the next State
        next_state = State(self.warehouse_num, self.T,
                           list(self.demand_history), self.t)

        next_state.factory_stock = min(state.factory_stock + action.production_level - np.sum(
            action.shippings_to_warehouses), self.storage_capacities[0])

        for w in range(self.warehouse_num):
            next_state.warehouse_stock[w] = min(
                state.warehouse_stock[w] + action.shippings_to_warehouses[w] - demands[w], self.storage_capacities[w + 1])

        self.t += 1
        self.demand_history.append(demands)

        return next_state, reward, self.t == self.T

    def plot_demands(self):
        demands = []
        for t in range(self.T):
            demands.append(self.demand(t))

        demands = np.array(demands)
        print(f"{demands.shape = }")
        plt.figure(figsize=(16, 5))
        plt.xlabel("Time step"), plt.ylabel("Demand")

        plt.plot(demands)
        plt.legend([f'Werehouse {i+1}' for i in range(self.warehouse_num)])

        plt.show()


# gym environment adapter
class SimpleSupplyChain(gym.Env):
    def __init__(self):
        self.reset()
        self.action_space = Box(low=0.0, high=10.0, shape=(
            self.supply_chain.warehouse_num + 1, ), dtype=np.float32)
        self.observation_space = Box(-15000, 15000, shape=(
            len(self.supply_chain.initial_state().to_array()), ), dtype=np.float32)

    def reset(self):
        self.supply_chain = SupplyChainEnvironment()
        self.state = self.supply_chain.initial_state()
        return self.state.to_array()

    def step(self, action):
        action_obj = Action(self.supply_chain.warehouse_num)
        action_obj.production_level = action[0]
        action_obj.shippings_to_warehouses = action[1:]
        self.state, reward, done = self.supply_chain.step(
            self.state, action_obj)
        return self.state.to_array(), reward, done, {}
