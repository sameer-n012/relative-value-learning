import json
import argparse
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)

    if "log" not in data:
        raise ValueError("JSON file does not contain a 'log' field.")

    return data


def extract_series(logs, x_key="frames_collected"):
    x = [entry[x_key] for entry in logs]

    metrics = {
        "mean_return": [entry.get("mean_return") for entry in logs],
        "mean_eps_length": [entry.get("mean_eps_length") for entry in logs],
        "loss_total": [entry.get("loss_total") for entry in logs],
        "loss_policy": [entry.get("loss_policy") for entry in logs],
        "loss_critic": [entry.get("loss_critic") for entry in logs],
        "loss_entropy": [entry.get("loss_entropy") for entry in logs],
    }

    return x, metrics


def plot_curve(xs, metrics):
    fig = plt.figure()

    window_size = 200
    cutoff = 5000000

    for k, v in x.items():
        x = xs[k][:cutoff]
        y = m[k]["mean_return"][:cutoff]

        y_smooth = np.convolve(
            y,
            np.ones(window_size) / window_size,
            mode="valid"
        )

        x_smooth = x[window_size - 1:]
        plt.plot(x_smooth, y_smooth, linewidth=2, label=k)

    plt.title("Mean Return Moving Average (Asterix)", fontsize=16)
    plt.xlabel("Frames")
    plt.ylabel("Mean Return")
    plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def main():

    env = "asterix"
    data_rv = load_results(f"results/rv_{env}.json")["log"]
    data_rv = load_results(f"results/rvo_{env}.json")["log"]
    data_rv = load_results(f"results/baseline_{env}.json")["log"]

    x_rv, metrics_rv = extract_series(data_rv)
    x_rvo, metrics_rvo = extract_series(data_rvo)
    x_baseline, metrics_baseline = extract_series(data_baseline)

    x = {"Relative Value (No Offset)": x_rv, "Relative Value": x_rvo, "Baseline PPO": x_baseline}
    metrics = {"Relative Value (No Offset)": metrics_rv, "Relative Value": metrics_rvo, "Baseline PPO": metrics_baseline}

    fig = plot_curve(x, metrics)

    fig.savefig("results/a.png", dpi=300, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    main()