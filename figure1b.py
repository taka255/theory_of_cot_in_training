import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED = 20260920
D = 96
L = 192
ETA = 0.15
SIGMA = 0.1
TIMES = (5, 10, 20)
TARGETS = (0.55, 0.7, 0.85, 0.925, 0.975, 1.05, 1.15, 1.3, 1.45)
REPETITIONS = 8
CONTEXTS = 8
OUTPUT = Path(__file__).parent / "output"
THETA = (np.arange(4096) + 0.5) * np.pi / 4096


def mp_nodes(rho):
    c = 1 / rho
    x = 1 + c - 2 * np.sqrt(c) * np.cos(THETA)
    weight = 2 * np.sin(THETA) ** 2 / x
    return x, weight


def filter_at(x, time):
    return -np.expm1(time * np.log1p(-ETA * x)) / x


def gd_risk(time, alpha=2.0):
    x, weight = mp_nodes(alpha)
    q = filter_at(x, time)
    return np.mean(((1 - x * q) ** 2 + SIGMA**2 / alpha * x * q**2) * weight)


def projected_risk(time, phi, alpha=2.0):
    if phi >= 1:
        return gd_risk(time, alpha)
    x, weight = mp_nodes(alpha / phi)
    q = filter_at(x, time)
    average = lambda value: np.mean(value * weight)
    k1 = (alpha - phi) * ETA * time + phi * average(q)
    k2 = (alpha - phi) * (ETA * time) ** 2 + phi * average(q**2)
    return (
        1 - phi
        + phi * average((1 - x * q) ** 2)
        + phi * (2 * (1 - phi) + SIGMA**2) / alpha * average(x * q**2)
        - 2 * (1 - phi) / alpha * k1
        + (1 - phi) * (1 - phi + SIGMA**2) / alpha**2 * k2
        + (1 - phi) / alpha**2 * k1**2
    )


def learned_update(rng, samples):
    if samples >= D:
        return -ETA * np.eye(D)
    frame = np.linalg.qr(rng.standard_normal((D, samples)), mode="reduced")[0]
    return -ETA * frame @ frame.T


def risks(update, design):
    eigenvalues, vectors = np.linalg.eigh(design @ design.T / L)
    transition = np.eye(D) + eigenvalues[:, None] * (vectors.T @ update @ vectors)
    power = np.eye(D)
    geometric_sum = np.zeros((D, D))
    output = {}
    for time in range(1, max(TIMES) + 1):
        geometric_sum += power
        if time in TIMES:
            response = ETA * geometric_sum
            bias = np.linalg.norm(response * eigenvalues - np.eye(D)) ** 2 / D
            noise = SIGMA**2 / L * np.sum(response**2 * eigenvalues)
            output[time] = bias + noise
        power = power @ transition
    return output


def experiment():
    rows = []
    for target in TARGETS:
        samples = round(target * D)
        phi = samples / D
        for repetition in range(1, REPETITIONS + 1):
            rng = np.random.default_rng(SEED + 9176 * samples + 1_000_003 * repetition)
            update = learned_update(rng, samples)
            for context in range(1, CONTEXTS + 1):
                values = risks(update, rng.standard_normal((D, L)))
                rows.extend(
                    {"phi": phi, "M": samples, "repetition": repetition,
                     "context": context, "time": time, "mse": values[time]}
                    for time in TIMES
                )
        print(f"experiment phi={phi:.4f}", flush=True)
    return rows


def summarize(rows):
    summary = []
    for time in TIMES:
        baseline = gd_risk(time)
        for phi in sorted({row["phi"] for row in rows}):
            means = [
                np.mean([row["mse"] for row in rows
                         if row["time"] == time and row["phi"] == phi
                         and row["repetition"] == repetition])
                for repetition in range(1, REPETITIONS + 1)
            ]
            summary.append({
                "phi": phi,
                "time": time,
                "mean_mse": np.mean(means),
                "excess_mse": np.mean(means) - baseline,
                "se": np.std(means, ddof=1) / np.sqrt(REPETITIONS),
            })
    return summary


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(exist_ok=True)
    experiment_rows = experiment()
    summary = summarize(experiment_rows)
    dense_phi = np.r_[np.linspace(0.51, 1, 250), 1.49]
    theory = [
        {"phi": phi, "time": time,
         "excess_mse": projected_risk(time, phi) - gd_risk(time)}
        for time in TIMES for phi in dense_phi
    ]
    write_csv("figure1b_trials.csv", experiment_rows)
    write_csv("figure1b_summary.csv", summary)
    write_csv("figure1b_theory.csv", theory)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": SEED,
        "dimension": D,
        "context_length": L,
        "eta": ETA,
        "sigma": SIGMA,
        "times": TIMES,
        "targets": TARGETS,
        "repetitions": REPETITIONS,
        "contexts": CONTEXTS,
    }
    (OUTPUT / "figure1b_metadata.json").write_text(json.dumps(metadata, indent=2))

    figure, axis = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    for time, color in zip(TIMES, ("C0", "C1", "C2")):
        curve = [row for row in theory if row["time"] == time]
        points = [row for row in summary if row["time"] == time]
        axis.plot([row["phi"] for row in curve], [row["excess_mse"] for row in curve],
                  color=color, label=f"t = {time}")
        axis.errorbar([row["phi"] for row in points], [row["excess_mse"] for row in points],
                      yerr=[row["se"] for row in points], fmt="o", markerfacecolor="white",
                      color=color, capsize=2)
    axis.axvline(1, color="black", linestyle=":")
    axis.set(xlim=(0.51, 1.49), xlabel=r"Training CoT budget $\varphi=\tau(K-1)$",
             ylabel=r"$\mathcal{E}(t)-\mathcal{E}_{\mathrm{GD}}(t)$", title=r"$\alpha=2$")
    axis.legend(frameon=False)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"figure1b.{extension}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
