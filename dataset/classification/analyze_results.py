import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_summary(summary_path: str) -> Dict[str, dict]:
    with open(summary_path, 'r') as f:
        return json.load(f)


def discover_experiments(output_dir: str, experiments_filter: List[str]) -> Dict[str, dict]:
    comparison_path = os.path.join(output_dir, 'comparison_summary.json')
    experiments = {}
    if os.path.exists(comparison_path):
        data = load_summary(comparison_path)
        for name, summary in data.items():
            if not experiments_filter or name in experiments_filter:
                experiments[name] = summary

    if experiments_filter:
        candidates = experiments_filter
    else:
        candidates = sorted(
            entry for entry in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, entry))
        )

    for name in candidates:
        if name in experiments:
            continue
        summary_path = os.path.join(output_dir, name, 'summary.json')
        if os.path.exists(summary_path):
            data = load_summary(summary_path)
            experiments[name] = data

    missing = [name for name in experiments_filter if name not in experiments]
    if missing:
        raise FileNotFoundError(f"Missing summaries for experiments: {missing}")
    return experiments


def print_table(experiments: Dict[str, dict]) -> None:
    header = [
        "Experiment", "Best Val Acc", "Overall Acc", "Small Acc", "Noisy Acc", "Epochs", "Batch"
    ]
    print("\n=== Experiment Metrics ===")
    print("{:<24s} {:>12s} {:>12s} {:>12s} {:>12s} {:>6s} {:>6s}".format(*header))
    for name, summary in experiments.items():
        metrics = summary['final_metrics']
        config = summary.get('config', {})
        small = metrics['small_acc']
        noisy = metrics['noisy_acc']
        print("{:<24s} {:>12.4f} {:>12.4f} {:>12s} {:>12s} {:>6} {:>6}".format(
            name,
            summary['best_val_acc'],
            metrics['overall_acc'],
            f"{small:.4f}" if small is not None else "n/a",
            f"{noisy:.4f}" if noisy is not None else "n/a",
            config.get('epochs', '?'),
            config.get('batch_size', '?'),
        ))


def plot_metrics(experiments: Dict[str, dict], output_dir: str, filename: str = 'comparison_plot.png') -> str:
    names = list(experiments.keys())
    overall = [experiments[name]['final_metrics']['overall_acc'] for name in names]
    small = [experiments[name]['final_metrics']['small_acc'] or 0.0 for name in names]
    noisy = [experiments[name]['final_metrics']['noisy_acc'] or 0.0 for name in names]
    indices = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(indices - width, overall, width, label='Overall', color='#4C72B0')
    ax.bar(indices, small, width, label='Small digits', color='#55A868')
    ax.bar(indices + width, noisy, width, label='Noisy digits', color='#C44E52')

    ax.set_xticks(indices)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title('Experiment comparison (overall / small / noisy)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    output_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Analyze experiment summaries and produce comparison plots.'
    )
    parser.add_argument('--output_dir', type=str, default='logs/experiments_fair',
                        help='Directory containing experiment outputs or summaries.')
    parser.add_argument('--experiments', type=str, default=None,
                        help='Comma separated list of experiment names to include.')
    parser.add_argument('--plot_name', type=str, default='comparison_plot.png',
                        help='Filename for the generated plot.')
    args = parser.parse_args()

    experiments_filter = []
    if args.experiments:
        experiments_filter = [name.strip() for name in args.experiments.split(',') if name.strip()]

    experiments = discover_experiments(args.output_dir, experiments_filter)
    if not experiments:
        raise RuntimeError("No experiment summaries found. Run experiments first.")

    print_table(experiments)
    plot_path = plot_metrics(experiments, args.output_dir, args.plot_name)
    print(f"\nComparison plot saved to: {plot_path}")


if __name__ == '__main__':
    main()
