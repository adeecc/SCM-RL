from SQPolicy import SQPolicy, simulate, simulate_episode
from Environment import SupplyChainEnvironment
from utils import visualize_transitions
import numpy as np
from ax import optimize
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


env = SupplyChainEnvironment()
env.plot_demands()


def eval_func(p):
    policy = SQPolicy(
        factory_safety_stock=p['factory_s'],
        factory_reorder_amount=p['factory_Q'],
        safety_stock=(p['w1_s'], p['w2_s']),
        reorder_amount=(p['w1_Q'], p['w2_Q']))

    return np.mean(simulate(env, policy, num_episodes=500))


parameters = (
    {
        "name": "factory_s",
        "type": "range",
        "bounds": [0.0, 50.0],
        "value_type": "float"
    },
    {
        "name": "factory_Q",
        "type": "range",
        "bounds": [0.0, 50.0],
        "value_type": "float"
    },
    {
        "name": "w1_s",
        "type": "range",
        "bounds": [0.0, 50.0],
        "value_type": "float"
    },
    {
        "name": "w1_Q",
        "type": "range",
        "bounds": [0.0, 50.0],
        "value_type": "float"
    },
    {
        "name": "w2_s",
        "type": "range",
        "bounds": [0.0, 50.0],
        "value_type": "float"
    },
    {
        "name": "w2_Q",
        "type": "range",
        "bounds": [0.0, 50.0],
        "value_type": "float"
    },
)

best_parameters, best_values, experiment, model = optimize(
    parameters=parameters,
    evaluation_function=eval_func,
    minimize=False,
    total_trials=250  # Set to 100
)


print(f"Best Parameters: \n{best_parameters}")
print(f"Best Values: \n{best_values}")

env = SupplyChainEnvironment()
sq_policy = SQPolicy(
    factory_safety_stock=best_parameters["factory_s"],
    factory_reorder_amount=best_parameters["factory_Q"],
    safety_stock=(best_parameters["w1_s"], best_parameters["w2_s"]),
    reorder_amount=(best_parameters["w1_Q"], best_parameters["w2_Q"])
)


return_trace = simulate(env, sq_policy, num_episodes=50)

plt.figure(figsize=(16, 4))
plt.plot(range(len(return_trace)), return_trace)
print(
    f"Reward: mean {np.mean(return_trace)}, standard deviation {np.std(return_trace)}")

plt.show()

transitions_sQ = simulate_episode(env, sq_policy)
visualize_transitions(env, np.array(transitions_sQ))
