import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_filename(filename: str):
    stem = Path(filename).stem
    info = {}

    # 识别所有 key=value 的起始位置
    matches = list(re.finditer(r'([A-Za-z0-9]+)=', stem))

    for i, m in enumerate(matches):
        key = m.group(1)
        value_start = m.end()

        if i + 1 < len(matches):
            # 截到下一个 _key= 之前
            next_key_start = matches[i + 1].start()
            value = stem[value_start:next_key_start]
            if value.endswith('_'):
                value = value[:-1]
        else:
            value = stem[value_start:]

        info[key] = value

    # 兼容后续代码里用的小写/固定键名
    normalized = {
        'model': info.get('model'),
        'dataset': info.get('dataset'),
        'structure': info.get('structure'),
        'parti': info.get('parti'),
        'onlineRatio': info.get('onlineRatio'),
        'commE': info.get('commE'),
        'localE': info.get('localE'),
        'bs': info.get('bs'),
        'lr': info.get('lr'),
        'alpha': info.get('alpha'),
        'drop': info.get('drop'),
        'denoise': info.get('denoise'),
        'noiseType': info.get('noiseType'),
        'noiseMax': info.get('noiseMax'),
        'avg': info.get('avg'),
        'seed': info.get('seed'),
    }
    return normalized


def find_result_files(root: Path):
    return sorted(root.rglob('*.txt'))


def load_result_file(path: Path):
    try:
        df = pd.read_csv(path, sep='\t')
    except Exception:
        # fallback for whitespace-separated logs
        df = pd.read_csv(path, sep=r'\s+', engine='python')

    # normalize columns
    rename_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc == 'epoch':
            rename_map[c] = 'epoch'
        elif lc == 'acc':
            rename_map[c] = 'acc'
        elif lc == 'pure':
            rename_map[c] = 'pure'
        elif lc in ('round_time_sec', 'roundtime', 'round_time'):
            rename_map[c] = 'round_time_sec'
        elif lc in ('total_time_sec', 'totaltime', 'total_time'):
            rename_map[c] = 'total_time_sec'
    df = df.rename(columns=rename_map)

    if 'epoch' not in df.columns or 'acc' not in df.columns:
        raise ValueError(f'{path} does not contain required columns epoch and acc')

    if 'pure' in df.columns:
        df['pure'] = pd.to_numeric(df['pure'], errors='coerce')
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df['acc'] = pd.to_numeric(df['acc'], errors='coerce')
    df = df.dropna(subset=['epoch', 'acc']).copy()

    meta = parse_filename(path.name)
    meta['path'] = str(path)
    meta['run_name'] = path.stem
    return df, meta


def build_runs(root: Path):
    runs = []
    for path in find_result_files(root):
        try:
            df, meta = load_result_file(path)
            runs.append((df, meta))
        except Exception as e:
            print(f'[skip] {path}: {e}')
    return runs


def filter_runs(runs, dataset=None, noise_type=None, denoise=None):
    out = []
    for df, meta in runs:
        if dataset and meta.get('dataset') != dataset:
            continue
        if noise_type and meta.get('noiseType') != noise_type:
            continue
        if denoise and meta.get('denoise') != denoise:
            continue
        out.append((df, meta))
    return out


def label_for(meta, fields):
    return ' | '.join(f'{f}={meta.get(f, "NA")}' for f in fields)


def plot_runs(runs, metric='acc', title='Results', out_path=None, label_fields=None):
    if not runs:
        print('No runs matched.')
        return

    label_fields = label_fields or ['dataset', 'noiseType', 'denoise', 'seed']
    plt.figure(figsize=(11, 7))
    for df, meta in runs:
        if metric not in df.columns:
            continue
        plt.plot(df['epoch'], df[metric], label=label_for(meta, label_fields), linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved plot to: {out_path}')
    else:
        plt.show()
    plt.close()


def summarize_runs(runs):
    rows = []
    for df, meta in runs:
        row = dict(meta)
        row['final_acc'] = float(df['acc'].iloc[-1]) if len(df) else None
        row['best_acc'] = float(df['acc'].max()) if len(df) else None
        if 'pure' in df.columns:
            pure_series = pd.to_numeric(df['pure'], errors='coerce').dropna()
            row['final_pure'] = float(pure_series.iloc[-1]) if len(pure_series) else None
            row['best_pure'] = float(pure_series.max()) if len(pure_series) else None
        else:
            row['final_pure'] = None
            row['best_pure'] = None
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        cols = [
            'dataset', 'noiseType', 'denoise', 'seed', 'alpha', 'drop',
            'final_acc', 'best_acc', 'final_pure', 'best_pure', 'path'
        ]
        cols = [c for c in cols if c in summary.columns]
        summary = summary[cols]
    return summary


def compare_mean_curve(runs, group_by=('dataset', 'noiseType', 'denoise'), metric='acc', out_path=None):
    if not runs:
        print('No runs matched.')
        return

    grouped = defaultdict(list)
    for df, meta in runs:
        key = tuple(meta.get(k, 'NA') for k in group_by)
        grouped[key].append(df[['epoch', metric]].copy())

    plt.figure(figsize=(11, 7))
    for key, dfs in grouped.items():
        merged = None
        for i, d in enumerate(dfs):
            d = d.rename(columns={metric: f'{metric}_{i}'})
            merged = d if merged is None else merged.merge(d, on='epoch', how='outer')
        metric_cols = [c for c in merged.columns if c.startswith(metric + '_')]
        merged = merged.sort_values('epoch')
        merged['mean_metric'] = merged[metric_cols].mean(axis=1)
        merged['std_metric'] = merged[metric_cols].std(axis=1)
        label = ' | '.join(f'{k}={v}' for k, v in zip(group_by, key))
        plt.plot(merged['epoch'], merged['mean_metric'], label=label, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel(f'Mean {metric.upper()}')
    plt.title(f'Mean Curve Comparison ({metric})')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved plot to: {out_path}')
    else:
        plt.show()
    plt.close()

def export_split_plots(runs, out_dir='figures', metric='acc'):
    os.makedirs(out_dir, exist_ok=True)

    # 按 (dataset, noiseType) 分组
    grouped = defaultdict(list)
    for df, meta in runs:
        dataset = meta.get('dataset', 'NA')
        noise_type = meta.get('noiseType', 'NA')
        grouped[(dataset, noise_type)].append((df, meta))

    for (dataset, noise_type), group_runs in grouped.items():
        title = f"DenoiseFL Results - {dataset} - {noise_type}"
        out_path = os.path.join(out_dir, f"{dataset}_{noise_type}_{metric}.png")
        plot_runs(
            group_runs,
            metric=metric,
            title=title,
            out_path=out_path,
            label_fields=['denoise', 'seed']
        )


def main():
    parser = argparse.ArgumentParser(description='Visualize DenoiseFL result txt files.')
    parser.add_argument('--root', type=str, default='results', help='Root results directory')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--noise_type', type=str, default=None)
    parser.add_argument('--denoise', type=str, default=None)
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'pure'])
    parser.add_argument('--mode', type=str, default='plot', choices=['plot', 'mean', 'summary', 'split'])
    parser.add_argument('--out', type=str, default=None, help='Output image or csv path')
    args = parser.parse_args()

    runs = build_runs(Path(args.root))
    runs = filter_runs(runs, dataset=args.dataset, noise_type=args.noise_type, denoise=args.denoise)

    if args.mode == 'summary':
        summary = summarize_runs(runs)
        if summary.empty:
            print('No runs matched.')
            return
        print(summary.to_string(index=False))
        if args.out:
            summary.to_csv(args.out, index=False)
            print(f'Saved summary to: {args.out}')

    elif args.mode == 'mean':
        compare_mean_curve(runs, metric=args.metric, out_path=args.out)

    elif args.mode == 'split':
        out_dir = args.out if args.out else 'figures'
        export_split_plots(runs, out_dir=out_dir, metric=args.metric)

    else:
        title_parts = ['DenoiseFL Results']
        if args.dataset:
            title_parts.append(args.dataset)
        if args.noise_type:
            title_parts.append(args.noise_type)
        if args.denoise:
            title_parts.append(args.denoise)
        plot_runs(runs, metric=args.metric, title=' - '.join(title_parts), out_path=args.out)

if __name__ == '__main__':
    main()
