import argparse
import os
import re
from typing import List, Optional

import matplotlib.pyplot as plt


def parse_log(filepath: str):
    epochs: List[int] = []
    accs: List[float] = []
    stage_pures: List[Optional[float]] = []

    round_pattern = re.compile(r"\[Round\s+(\d+)/(\d+)\].*?Acc=([0-9.]+)")
    stage_pure_pattern = re.compile(r"StagePure=([0-9.]+)")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m = round_pattern.search(line)
            if not m:
                continue

            epoch = int(m.group(1))
            acc = float(m.group(3))

            sp = None
            sp_match = stage_pure_pattern.search(line)
            if sp_match:
                sp = float(sp_match.group(1))

            epochs.append(epoch)
            accs.append(acc)
            stage_pures.append(sp)

    if len(epochs) == 0:
        raise RuntimeError("No round records found in the log.")

    return epochs, accs, stage_pures


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]
    out = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        chunk = values[left:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to a .log file")
    parser.add_argument("--output", type=str, default=None, help="Output png path")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window")
    parser.add_argument(
        "--metric",
        type=str,
        default="both",
        choices=["acc", "stage_pure", "both"],
        help="What to plot",
    )
    args = parser.parse_args()

    epochs, accs, stage_pures = parse_log(args.log)
    accs_plot = moving_average(accs, args.smooth)

    plt.figure(figsize=(9, 5.5))

    if args.metric in ["acc", "both"]:
        plt.plot(epochs, accs_plot, label="Accuracy")

    if args.metric in ["stage_pure", "both"]:
        sp_x = []
        sp_y = []
        for x, y in zip(epochs, stage_pures):
            if y is not None:
                sp_x.append(x)
                sp_y.append(y)
        if len(sp_y) > 0:
            sp_y = moving_average(sp_y, args.smooth)
            plt.plot(sp_x, sp_y, label="StagePure")

    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.title(args.title if args.title else os.path.basename(args.log))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output = args.output
    if output is None:
        output = os.path.splitext(args.log)[0] + "_curve.png"

    plt.savefig(output, dpi=200)
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()