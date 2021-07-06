import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def prepare_metric_plot(plots_n, n, ylabel):
    plt.subplot(plots_n, 1, n)
    plt.ylabel(ylabel)
    plt.tick_params(axis='x', which='both', bottom=True,
                    top=True, labelbottom=False)


def visualize_transitions(env, transitions):
    state_trace, action_trace, reward_trace = (
        transitions.T[0], transitions.T[1], transitions.T[2]  # transitions.T
    )
    plots_n = 10
    mpl.rcParams['lines.linewidth'] = 2
    prepare_metric_plot(plots_n, 1, "Stock,\n Factory")
    plt.plot(range(env.T), list(
        map(lambda s: s.factory_stock, state_trace)), c='purple', alpha=0.5)

    for w in range(env.warehouse_num):
        prepare_metric_plot(plots_n, 2 + w, f"Stock,\n WH {w+1}")
        plt.plot(range(env.T), list(
            map(lambda s: s.warehouse_stock[w], state_trace)), c='purple', alpha=0.5)

    prepare_metric_plot(plots_n, 5, "Production")
    plt.plot(range(env.T), list(
        map(lambda a: a.production_level, action_trace)), c='blue', alpha=0.5)


    for w in range(env.warehouse_num):
        prepare_metric_plot(plots_n, 6 + w, f"Shipment,\n WH {w + 1}")
        plt.plot(range(env.T), list(
            map(lambda a: a.shippings_to_warehouses[w], action_trace)), c='blue', alpha=0.5)


    prepare_metric_plot(plots_n, 9, "Profit")
    plt.plot(range(env.T), reward_trace, c='red', alpha=0.9, linewidth=2)

    plt.subplot(plots_n, 1, 10)
    plt.ylabel("Cumulative\nprofit")
    plt.ylim(0, 10000)
    plt.plot(range(env.T), np.cumsum(reward_trace),
            c='red', alpha=0.9, linewidth=2)
    plt.xlabel("Time step")

    plt.show()
