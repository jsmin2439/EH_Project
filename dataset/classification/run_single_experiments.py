import argparse
import os
import json
import torch

from run_experiments import (
    set_seed,
    load_path_list,
    build_experiments,
    run_experiment,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a single digit classification experiment.'
    )
    parser.add_argument('--experiment', type=str, default='baseline_resnet32',
                        help='Experiment name to run (baseline_resnet32, baseline_resnet50, '
                             'augmented_resnet20, superres_tta_resnet20)')
    parser.add_argument('--data_dir', type=str, default='../../digit_data',
                        help='Path to digit_data root directory.')
    parser.add_argument('--output_dir', type=str, default='experiment_runs_single',
                        help='Directory to store experiment outputs.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count for the experiment (default follows shared config).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size; capped at 64 to match shared training budget.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (optional).')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Override weight decay (optional).')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None,
                        help='cuda, cpu or auto detect.')
    parser.add_argument('--small_list', type=str, default='small_valid.txt',
                        help='Path to txt file listing small-digit validation samples.')
    parser.add_argument('--noisy_list', type=str, default='noisy_valid.txt',
                        help='Path to txt file listing noisy validation samples.')
    parser.add_argument('--hard_list', type=str, default='hard_train.txt',
                        help='Path to txt file listing hard training samples.')
    parser.add_argument('--hard_example_weight', type=float, default=3.0)
    parser.add_argument('--small_train_list', type=str, default=None,
                        help='Path to txt file listing small-digit training samples.')
    parser.add_argument('--focus_sampler_weight', type=float, default=None,
                        help='Override the sampler weight applied to focus (small/hard) samples.')
    parser.add_argument('--use_focus_loss', action='store_true',
                        help='Force class-weighted loss using focus sample statistics.')
    parser.add_argument('--focus_loss_weight', type=float, default=None,
                        help='Override the focus weighting factor when building class-weighted loss.')
    parser.add_argument('--max_error_log', type=int, default=50,
                        help='Maximum number of misclassified samples to log.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--focal_alpha', type=float, default=None,
                        help='Alpha value for focal loss (balance hard/easy samples).')
    return parser.parse_args()

def main():
    args = parse_args()
    args.experiments = args.experiment  # for compatibility if needed downstream
    set_seed(args.seed)
    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.batch_size > 64:
        print(f"[Fairness] Requested batch size {args.batch_size} is capped to 64 "
              "to match the shared training budget.")
        args.batch_size = 64

    # Convert relative paths to absolute paths if needed
    def resolve_list_path(path):
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        # Try relative to data_dir first
        data_dir_path = os.path.join(args.data_dir, path)
        if os.path.exists(data_dir_path):
            return data_dir_path
        # Fall back to original path
        return path
    
    small_list_path = resolve_list_path(args.small_list)
    noisy_list_path = resolve_list_path(args.noisy_list) 
    hard_list_path = resolve_list_path(args.hard_list)
    small_train_list_path = resolve_list_path(args.small_train_list)

    small_set = load_path_list(small_list_path)
    noisy_set = load_path_list(noisy_list_path)
    hard_set = load_path_list(hard_list_path)
    small_train_set = load_path_list(small_train_list_path)

    configs = build_experiments(args)
    cfg_map = {cfg.name: cfg for cfg in configs}
    if args.experiment not in cfg_map:
        raise ValueError(f"Unknown experiment '{args.experiment}'. "
                         f"Available options: {list(cfg_map.keys())}")

    summary = run_experiment(cfg_map[args.experiment], args, device,
                             small_set, noisy_set, hard_set, small_train_set)

    summary_path = os.path.join(args.output_dir, f'{args.experiment}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({args.experiment: summary}, f, indent=2)

    print(f"\nExperiment '{args.experiment}' completed.")
    print(f"Best validation accuracy: {summary['best_val_acc']:.4f}")
    metrics = summary['final_metrics']
    print(f"Overall accuracy: {metrics['overall_acc']:.4f}")
    small = metrics['small_acc']
    noisy = metrics['noisy_acc']
    print(f"Small-digit accuracy: {small:.4f}" if small is not None else "Small-digit accuracy: n/a")
    print(f"Noisy-digit accuracy: {noisy:.4f}" if noisy is not None else "Noisy-digit accuracy: n/a")
    print(f"Summary saved to: {summary_path}")

if __name__ == '__main__':
    main()