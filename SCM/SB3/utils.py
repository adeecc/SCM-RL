import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", palette="deep")


def rescale(dd, demand_range=(0, 1)):
    min_val, max_val = demand_range
    dd_min, dd_max = dd.min(axis=0), dd.max(axis=0)
    dd = (dd - dd_min) / (dd_max - dd_min)  # [0, 1]
    dd = min_val + (max_val - min_val) * dd
    return dd


def normalize(dd):
    dd_mean, dd_std = dd.mean(axis=0), dd.std(axis=0)
    return (dd - dd_mean) / dd_std


def prepare_metric_plot(plots_n, n, ylabel):
    plt.subplot(plots_n, 1, n)
    plt.ylabel(ylabel)
    plt.tick_params(axis='x', which='both', bottom=True,
                    top=True, labelbottom=False)


def visualize_transitions(transitions, T=50, warehouse_num=2):
    state_trace, action_trace, reward_trace = (
        transitions.T[0], transitions.T[1], transitions.T[2]  # transitions.T
    )

    plots_n = 8

    prepare_metric_plot(plots_n, 1, "Stock,\n Factory")
    plt.plot(range(T), list(
        map(lambda s: s.factory_stock, state_trace)), c='purple', alpha=0.5)

    for w in range(warehouse_num):
        prepare_metric_plot(plots_n, 2 + w, f"Stock,\n WH {w+1}")
        plt.plot(range(T), list(
            map(lambda s: s.warehouse_stock[w], state_trace)), c='purple', alpha=0.5)

    prepare_metric_plot(plots_n, 4, "Production")
    plt.plot(range(T), list(
        map(lambda a: a.production_level, action_trace)), c='blue', alpha=0.5)

    for w in range(warehouse_num):
        prepare_metric_plot(plots_n, 5 + w, f"Shipment,\n WH {w + 1}")
        plt.plot(range(T), list(
            map(lambda a: a.shippings_to_warehouses[w], action_trace)), c='blue', alpha=0.5)

    prepare_metric_plot(plots_n, 7, "Profit")
    plt.plot(range(T), reward_trace, c='red', alpha=0.9, linewidth=2)

    plt.subplot(plots_n, 1, 8)
    plt.ylabel("Cumulative\nprofit")
    plt.plot(range(T), np.cumsum(reward_trace), c='red', alpha=0.9)
    plt.xlabel("Time step")

    plt.show()
