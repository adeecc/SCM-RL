import collections
import numpy as np
import pandas as pd


import gym
from gym.spaces import Box

NUM_WAREHOUSES = 2
EPISODE_DURATION = 26
MAX_DEMAND = 5
UNIT_PRICE = 100
UNIT_COST = 40

class State(object):
    def __init__(self, warehouse_num, T, demand_history, t=0):
        self.warehouse_num = warehouse_num
        self.factory_stock = 0
        self.warehouse_stock = np.repeat(0, warehouse_num)
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
    def __init__(self):
        self.T = EPISODE_DURATION               # episode duration
        self.warehouse_num = NUM_WAREHOUSES
        self.d_max = MAX_DEMAND           # maximum demand, units
        self.d_var = 2            # maximum random demand variation, units

        self.unit_price = UNIT_PRICE     # unit price in dollars
        self.unit_cost = UNIT_COST       # unit cost in dollars

        demand_data = pd.read_csv("./data/demand.csv")  # Longest List

        for col in demand_data.columns:
            if col == "Date": 
                continue
            demand_data[col] = (demand_data[col] - demand_data[col].min()) / \
                (demand_data[col].max() - demand_data[col].min())


        self.demand_data = demand_data * self.d_max

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
        for i in range(demand_history_len):
            self.demand_history.append(np.zeros(self.warehouse_num))
        self.t = 0

    # demand at time t at warehouse j
    def demand(self, j, t):
        # return np.round(self.d_max/2 + self.d_max/2 * np.sin(2 * np.pi * (t + 2 * j)/self.T * 2) + np.random.randint(0, self.d_var))
        try:
            return self.demand_data.iloc[420 + t, j].to_numpy(dtype="float64")
        except:
            print(f"SupplyChainEnvironment.demand called with {j = }, {t = }")
            return np.zeros((2,))

    def initial_state(self):
        return State(self.warehouse_num, self.T, list(self.demand_history))

    def step(self, state, action):
        demands = np.fromfunction(lambda j: self.demand(
            j + 1, self.t), (self.warehouse_num,), dtype=int)

        # calculating the reward (profit)
        total_revenue = self.unit_price * np.sum(demands)
        total_production_cost = self.unit_cost * action.production_level
        total_storage_cost = np.dot(self.storage_costs, np.maximum(
            state.stock_levels(), np.zeros(self.warehouse_num + 1)))
        total_penalty_cost = - self.penalty_unit_cost * \
            (np.sum(np.minimum(state.warehouse_stock, np.zeros(
                self.warehouse_num))) + min(state.factory_stock, 0))
        total_transportation_cost = np.dot(
            self.transporation_costs, action.shippings_to_warehouses)
        reward = total_revenue - total_production_cost - \
            total_storage_cost - total_penalty_cost - total_transportation_cost

        # calculating the next state
        next_state = State(self.warehouse_num, self.T, self.t)
        next_state.factory_stock = min(state.factory_stock + action.production_level - np.sum(
            action.shippings_to_warehouses), self.storage_capacities[0])
        for w in range(self.warehouse_num):
            next_state.warehouse_stock[w] = min(
                state.warehouse_stock[w] + action.shippings_to_warehouses[w] - demands[w], self.storage_capacities[w + 1])
        next_state.demand_history = list(self.demand_history)

        self.t += 1
        self.demand_history.append(demands)

        return next_state, reward, self.t == self.T - 1




# gym environment adapter
class SimpleSupplyChain(gym.Env):
    def __init__(self, config):
        self.reset()
        self.action_space = Box(low=0.0, high=20.0, shape=(
            self.supply_chain.warehouse_num + 1, ), dtype=np.float32)
        self.observation_space = Box(-10000, 10000, shape=(
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
