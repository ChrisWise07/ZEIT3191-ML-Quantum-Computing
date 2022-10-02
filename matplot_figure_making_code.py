import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json

mpl.use("pdf")

# width as measured in inkscape
width = 3.487
height = width / 1.618
title_size = 7
alpha_value = 0.625

plt.rc("font", family="serif", serif="Times")
plt.rc("text", usetex=True)

for parameter_name, x_axis_lim in [
    ["epsilon", [-np.pi / 2, np.pi / 2]],
    ["x", [0, 1.0]],
    ["y", [0, 1.0]],
    ["z", [0, 1.0]],
]:

    fig, ax = plt.subplots(figsize=(width, height))
    fig.suptitle(
        f"PSO Solutions For Various Initial {parameter_name.capitalize()} Values",
        fontsize=title_size,
    )
    ax.set_ylim([-1.0, 1.0])
    ax.set_xlim(x_axis_lim)

    init_data = json.load(open(f"{parameter_name}_init_map.json"))
    x = []
    colour_map = {}

    graphing_data_map = {
        "mse": {"data": [], "colour": "r", "name": "MSE"},
        "epsilon_sol": {"data": [], "colour": "b", "name": "epsilon"},
        "x_sol": {"data": [], "colour": "c", "name": "x"},
        "y_sol": {"data": [], "colour": "m", "name": "y"},
        "z_sol": {"data": [], "colour": "y", "name": "z"},
    }

    for init_value in init_data:
        x.append(round(float(init_value), 2))
        graphing_data_map["mse"]["data"].append(init_data[init_value][0])
        graphing_data_map["epsilon_sol"]["data"].append(
            init_data[init_value][1][0]
        )
        graphing_data_map["x_sol"]["data"].append(init_data[init_value][1][1])
        graphing_data_map["y_sol"]["data"].append(init_data[init_value][1][2])
        graphing_data_map["z_sol"]["data"].append(init_data[init_value][1][3])

    for key, graph_info in graphing_data_map.items():
        ax.plot(
            x,
            graph_info.get("data"),
            marker="o",
            color=graph_info.get("colour"),
            linestyle="solid",
            alpha=alpha_value,
        )
        colour_map.update({graph_info.get("name"): graph_info.get("colour")})

    plt.xticks(
        x,
        fontsize=title_size - 1,
    )

    plt.yticks(
        fontsize=title_size - 1,
    )

    plt.ylabel("Values", fontsize=title_size - 1)
    plt.xlabel(f"Initial {parameter_name} value", fontsize=title_size - 1)

    box = ax.get_position()
    ax.set_position(
        [
            box.x0 - box.width * 0.0,
            box.y0 + box.height * 0.175,
            box.width * 1.075,
            box.height * 0.85,
        ]
    )

    labels = list(colour_map.keys())
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour, alpha=alpha_value)
        for colour in colour_map.values()
    ]
    plt.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=6,
        fontsize=title_size - 1,
        handletextpad=0.2,
        handlelength=1.0,
        columnspacing=1.0,
    )

    fig.savefig(f"{parameter_name}_init_graph.pdf")
