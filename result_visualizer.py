import os
import re
import csv
from pathlib import Path


def extract_from_path(path_str: str):
    info = {
        "model": None,
        "dataset": None,
        "dir_alpha": None,
        "noise_mode": None,
        "noise_rate": None,
        "noise_type": None,
        "strategy": None,
        "drop_rate": None,
    }

    parts = Path(path_str).parts

    if len(parts) >= 4:
        try:
            idx = parts.index("results")
            info["dataset"] = parts[idx + 1] if idx + 1 < len(parts) else None
            info["model"] = parts[idx + 2] if idx + 2 < len(parts) else None
            tag = parts[idx + 3] if idx + 3 < len(parts) else ""
        except ValueError:
            tag = ""
    else:
        tag = ""

    m = re.search(r"_a-([0-9.]+)", tag)
    if m:
        info["dir_alpha"] = m.group(1)

    m = re.search(r"_nm-([a-zA-Z0-9_]+)", tag)
    if m:
        info["noise_mode"] = m.group(1)

    m = re.search(r"_nr-([0-9.]+)", tag)
    if m:
        info["noise_rate"] = m.group(1)

    m = re.search(r"_nt-([a-zA-Z0-9_]+)", tag)
    if m:
        info["noise_type"] = m.group(1)

    m = re.search(r"_st-([a-zA-Z0-9_]+)", tag)
    if m:
        info["strategy"] = m.group(1)

    m = re.search(r"_dr-([0-9.]+)", tag)
    if m:
        info["drop_rate"] = m.group(1)

    return info


def extract_from_log(log_path: str):
    result = {
        "batch_size": None,
        "final_acc": None,
        "best_acc": None,
        "final_pure": None,
        "best_pure": None,
    }

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Local batch size:"):
                m = re.search(r"Local batch size:\s*([0-9]+)", line)
                if m:
                    result["batch_size"] = m.group(1)

            elif line.startswith("Final Acc:"):
                m = re.search(r"Final Acc:\s*([0-9.]+)", line)
                if m:
                    result["final_acc"] = m.group(1)

            elif line.startswith("Best Acc"):
                m = re.search(r"Best Acc\s*:\s*([0-9.]+)", line)
                if m:
                    result["best_acc"] = m.group(1)

            elif line.startswith("Final Pure:"):
                m = re.search(r"Final Pure:\s*([0-9.]+)", line)
                if m:
                    result["final_pure"] = m.group(1)

            elif line.startswith("Best Pure"):
                m = re.search(r"Best Pure\s*:\s*([0-9.]+)", line)
                if m:
                    result["best_pure"] = m.group(1)

    return result


def summarize_results(results_root="results", output_csv="results_summary.csv"):
    rows = []

    for root, _, files in os.walk(results_root):
        for fname in files:
            if not fname.endswith(".log"):
                continue

            log_path = os.path.join(root, fname)

            if "fed" not in log_path:
                continue

            path_info = extract_from_path(log_path)
            log_info = extract_from_log(log_path)

            if log_info["final_acc"] is None and log_info["best_acc"] is None:
                continue

            row = {
                "model": path_info["model"],
                "dataset": path_info["dataset"],
                "dir_alpha": path_info["dir_alpha"],
                "noise_mode": path_info["noise_mode"],
                "noise_rate": path_info["noise_rate"],
                "noise_type": path_info["noise_type"],
                "strategy": path_info["strategy"],
                "drop_rate": path_info["drop_rate"],
                "batch_size": log_info["batch_size"],
                "final_acc": log_info["final_acc"],
                "best_acc": log_info["best_acc"],
                "final_pure": log_info["final_pure"],
                "best_pure": log_info["best_pure"],
            }
            rows.append(row)

    def sort_key(x):
        return (
            x["dataset"] or "",
            x["model"] or "",
            x["strategy"] or "",
            float(x["dir_alpha"]) if x["dir_alpha"] is not None else -1,
            float(x["noise_rate"]) if x["noise_rate"] is not None else -1,
            float(x["drop_rate"]) if x["drop_rate"] is not None else -1,
            int(x["batch_size"]) if x["batch_size"] is not None else -1,
        )

    rows.sort(key=sort_key)

    fieldnames = [
        "model",
        "dataset",
        "dir_alpha",
        "noise_mode",
        "noise_rate",
        "noise_type",
        "strategy",
        "drop_rate",
        "batch_size",
        "final_acc",
        "best_acc",
        "final_pure",
        "best_pure",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} records to {output_csv}")


if __name__ == "__main__":
    summarize_results(results_root="results", output_csv="results_summary.csv")