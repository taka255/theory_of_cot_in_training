import csv
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


D = 32
L = 24
ETA = 0.1
SIGMA = 0.1
TRAINING_STEPS = 2
UPDATES = 5000
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 128
TEST_TASKS = 192
TIMES = (8, 16, 32)
SAMPLES = (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256)
SEEDS = tuple(range(950, 958))
OUTPUT = Path(__file__).parent / "output"

torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.width = 2 * D + 2
        self.heads = 2
        self.head_width = self.width // self.heads
        self.q = torch.nn.Linear(self.width, self.width, bias=False)
        self.k = torch.nn.Linear(self.width, self.width, bias=False)
        self.v = torch.nn.Linear(self.width, self.width, bias=False)
        self.o = torch.nn.Linear(self.width, self.width, bias=False)

    def forward(self, context, state):
        batch = len(state)
        query = torch.cat((torch.zeros_like(state), state.new_zeros(batch, 1),
                           state, state.new_ones(batch, 1)), dim=1)
        q = self.q(query).reshape(batch, self.heads, self.head_width)
        k = self.k(context).reshape(batch, L, self.heads, self.head_width).transpose(1, 2)
        v = self.v(context).reshape(batch, L, self.heads, self.head_width).transpose(1, 2)
        weights = ((k * q[:, :, None]).sum(-1) / math.sqrt(self.head_width)).softmax(-1)
        attended = (weights[..., None] * v).sum(2).flatten(1)
        return self.o(attended)[:, D + 1:2 * D + 1]


def make_tasks(count, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, D, L, generator=generator)
    w = torch.randn(count, D, generator=generator) / math.sqrt(D)
    y = torch.einsum("mdl,md->ml", x, w)
    y += SIGMA * torch.randn(count, L, generator=generator)
    h = x @ x.transpose(1, 2) / L
    b = torch.einsum("mdl,ml->md", x, y) / L
    states = [torch.zeros_like(b)]
    for _ in range(TRAINING_STEPS):
        states.append(states[-1] + ETA * (b - (h @ states[-1][..., None]).squeeze(-1)))
    return {"x": x, "w": w, "y": y, "states": torch.stack(states, dim=1)}


def context_tokens(tasks):
    x, y = tasks["x"], tasks["y"]
    count = len(x)
    return torch.cat((x.transpose(1, 2), y[..., None],
                      x.new_zeros(count, L, D + 1)), dim=2)


def training_pairs(tasks):
    count = len(tasks["x"])
    context = context_tokens(tasks).repeat_interleave(TRAINING_STEPS, 0)
    states = tasks["states"][:, :TRAINING_STEPS].reshape(-1, D)
    targets = (tasks["states"][:, 1:] - tasks["states"][:, :-1]).reshape(-1, D)
    assert len(states) == count * TRAINING_STEPS
    return context, states, targets


def fit(model, tasks, seed):
    context, states, targets = training_pairs(tasks)
    scale = targets.square().mean().clamp_min(1e-12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, UPDATES, eta_min=LEARNING_RATE / 100)
    generator = torch.Generator().manual_seed(30000 + seed)
    for _ in range(UPDATES):
        indices = torch.randint(len(states), (min(BATCH_SIZE, len(states)),),
                                generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(context[indices], states[indices]) - targets[indices]).square().mean() / scale
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        error = (model(context, states) - targets).square().sum()
        return float(error / targets.square().sum().clamp_min(1e-30))


@torch.no_grad()
def evaluate(model, tasks):
    model = model.double().eval()
    tasks = {name: value.double() for name, value in tasks.items()}
    context = context_tokens(tasks)
    state = torch.zeros_like(tasks["w"])
    risks = {}
    for time in range(1, max(TIMES) + 1):
        state += model(context, state)
        if time in TIMES:
            risks[time] = float((state - tasks["w"]).square().sum(1).mean())
    return risks


def experiment():
    rows = []
    for seed in SEEDS:
        pool = make_tasks(1024, 10000 + seed)
        test = make_tasks(TEST_TASKS, 20000 + seed)
        for samples in SAMPLES:
            train = {name: value[:samples] for name, value in pool.items()}
            torch.manual_seed(50000 + seed)
            model = Attention()
            train_error = fit(model, train, seed)
            for time, risk in evaluate(model, test).items():
                rows.append({"M": samples, "phi": samples / D, "seed": seed,
                             "time": time, "risk": risk,
                             "train_relative_mse": train_error})
            print(f"seed={seed} M={samples}", flush=True)
    return rows


def summarize(rows):
    summary = []
    for time in TIMES:
        for samples in SAMPLES:
            values = np.array([row["risk"] for row in rows
                               if row["time"] == time and row["M"] == samples])
            mean = values.mean()
            se = values.std(ddof=1) / np.sqrt(len(values))
            summary.append({
                "M": samples,
                "phi": samples / D,
                "time": time,
                "mean_risk": mean,
                "se": se,
                "log10_mean_risk": np.log10(mean),
                "log10_lower": np.log10(max(mean - se, np.finfo(float).tiny)),
                "log10_upper": np.log10(mean + se),
            })
    return summary


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(exist_ok=True)
    trials = experiment()
    summary = summarize(trials)
    write_csv("figure2_softmax_trials.csv", trials)
    write_csv("figure2_softmax_summary.csv", summary)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "dimension": D,
        "context_length": L,
        "eta": ETA,
        "sigma": SIGMA,
        "training_steps": TRAINING_STEPS,
        "updates": UPDATES,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "test_tasks": TEST_TASKS,
        "times": TIMES,
        "samples": SAMPLES,
        "seeds": SEEDS,
    }
    (OUTPUT / "figure2_softmax_metadata.json").write_text(json.dumps(metadata, indent=2))

    figure, axis = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    for time, color in zip(TIMES, ("C0", "C1", "C2")):
        data = [row for row in summary if row["time"] == time]
        x = [row["phi"] for row in data]
        center = np.array([row["log10_mean_risk"] for row in data])
        lower = np.array([row["log10_lower"] for row in data])
        upper = np.array([row["log10_upper"] for row in data])
        axis.plot(x, center, color=color)
        axis.errorbar(x, center, yerr=(center - lower, upper - center), fmt="o",
                      markerfacecolor="white", color=color, capsize=2, label=f"t = {time}")
    axis.set_xscale("log", base=2)
    axis.set_xticks((0.25, 0.5, 1, 2, 4, 8), ("0.25", "0.5", "1", "2", "4", "8"))
    axis.set(xlim=(0.23, 8.7), ylim=(-0.65, 2.65),
             xlabel=r"Training CoT budget $\varphi=M(K-1)/D$",
             ylabel=r"$\log_{10}\mathcal{E}(t)$", title="Softmax attention")
    axis.legend(frameon=False)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"figure2_softmax.{extension}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
