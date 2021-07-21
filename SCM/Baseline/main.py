from SQPolicy import SQPolicy, simulate, simulate_episode
from Environment import SupplyChainEnvironment
from utils import visualize_transitions
import numpy as np
from ax import optimize
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def eval_func(p: dict, log=False):
    policy = SQPolicy(
        factory_safety_stock=p['factory_s'],
        factory_reorder_amount=p['factory_Q'],
        safety_stock=(p['w1_s'], p['w2_s']),
        reorder_amount=(p['w1_Q'], p['w2_Q']))

    return np.mean(simulate(policy, num_episodes=50))


parameters = (
    {
        "name": "factory_s",
        "type": "range",
        "bounds": [0.0, 10.0],
        "value_type": "float"
    },
    {
        "name": "factory_Q",
        "type": "range",
        "bounds": [5.0, 10.0],
        "value_type": "float"
    },
    {
        "name": "w1_s",
        "type": "range",
        "bounds": [0.0, 10.0],
        "value_type": "float"
    },
    {
        "name": "w1_Q",
        "type": "range",
        "bounds": [5.0, 10.0],
        "value_type": "float"
    },
    {
        "name": "w2_s",
        "type": "range",
        "bounds": [0.0, 10.0],
        "value_type": "float"
    },
    {
        "name": "w2_Q",
        "type": "range",
        "bounds": [5.0, 10.0],
        "value_type": "float"
    },
)

# Reward: mean 6660.313520818671, standard deviation 489.4557712110093
def bayesian_optimization(total_trials, parameters):
    best_parameters, best_values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=eval_func,
        minimize=False,
        total_trials=total_trials
    )

    return best_parameters, best_values

# Reward: mean 5347.2, standard deviation 613.7202620086778
def grid_search(p_grid1=[0, 5, 10],
                p_grid2=[0, 5, 10]):

    param_grid = {
        'factory_s': p_grid1,
        'factory_Q': p_grid1,
        'w1_s': p_grid2,
        'w2_s': p_grid2,
        'w1_Q': p_grid2,
        'w2_Q': p_grid2,
    }

    grid = ParameterGrid(param_grid)

    i = 0
    best_return = float('-inf')
    for p in grid:
        m_return = eval_func(p)
        if m_return > best_return:
            best_return = m_return
            best_params = p

        if i % 100 == 0:
            print(f"Configuration #{i} -- {best_return}")
        i += 1

    return best_params, best_return


def main(total_trials=100, num_episodes=25, optimization_strategy="BayesianOptimization"):
    if optimization_strategy == "BayesianOptimization":
        best_params, best_values = bayesian_optimization(
            total_trials, parameters)
    else:
        best_params, best_values = grid_search()

    print(f"Best Parameters: \n{best_params}")
    print(f"Best Values: \n{best_values}")

    factory_safety_stock = best_params["factory_s"]
    factory_reorder_amount = best_params["factory_Q"]
    safety_stock = (best_params["w1_s"], best_params["w2_s"])
    reorder_amount = (best_params["w1_Q"], best_params["w2_Q"])

    sq_policy = SQPolicy(
        factory_safety_stock,
        factory_reorder_amount,
        safety_stock,
        reorder_amount
    )

    return_trace = simulate(sq_policy, num_episodes=num_episodes, log=True)

    plt.figure(figsize=(16, 4))
    plt.plot(range(len(return_trace)), return_trace)
    print(
        f"Reward: mean {np.mean(return_trace)}, standard deviation {np.std(return_trace)}")

    plt.show()

    # transitions_sQ = simulate_episode(sq_policy, log=True)
    # visualize_transitions(np.array(transitions_sQ), T=50)


if __name__ == '__main__':
    main(total_trials=25, num_episodes=100, optimization_strategy="GridSearch")
