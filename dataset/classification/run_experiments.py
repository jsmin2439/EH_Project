import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import models
from dataloader import DigitData
from utils import train


DATASET_MEAN = [0.80048384, 0.44734452, 0.50106468]
DATASET_STD = [0.22327253, 0.29523788, 0.24583565]


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_path_list(list_path: Optional[str]) -> Sequence[str]:
    if not list_path or not os.path.exists(list_path):
        return []
    with open(list_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def build_eval_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def build_heavy_aug_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.8, 1.3)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                       saturation=0.3, hue=0.05)], p=0.7),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value='random'),
    ])


def build_superres_train_transform(size: int) -> transforms.Compose:
    upscale = int(size * 1.3)
    return transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.25)),
            transforms.Compose([
                transforms.Resize((upscale, upscale)),
                transforms.RandomCrop(size),
            ]),
        ]),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                       saturation=0.2, hue=0.04)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def build_text_focus_transform(size: int) -> transforms.Compose:
    upscale = int(size * 1.6)
    return transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.7, 1.35)),
            transforms.Compose([
                transforms.Resize((upscale, upscale)),
                transforms.RandomCrop(size),
            ]),
            transforms.Compose([
                transforms.Resize(int(size * 1.8)),
                transforms.CenterCrop(size),
            ]),
        ]),
        transforms.RandomApply([transforms.RandomAffine(degrees=6, translate=(0.05, 0.05),
                                                        scale=(0.9, 1.15), shear=(-5, 5))], p=0.6),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.35, contrast=0.4,
                                                       saturation=0.3, hue=0.05)], p=0.6),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.5))], p=0.4),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.3),
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=1.8)], p=0.5),
        transforms.RandomApply([transforms.RandomChoice([
            transforms.Lambda(lambda img: TF.adjust_contrast(img, 1.3)),
            transforms.Lambda(lambda img: TF.adjust_gamma(img, 0.9)),
            transforms.Lambda(lambda img: TF.adjust_saturation(img, 0.8)),
        ])], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        transforms.RandomErasing(p=0.35, scale=(0.01, 0.12), value='random'),
    ])


def build_superres_tta(size: int) -> List[Callable[[Image.Image], torch.Tensor]]:
    resize_large = int(size * 1.4)

    def compose(ops: List[Callable[[Image.Image], Image.Image]]) -> Callable[[Image.Image], torch.Tensor]:
        def apply(img: Image.Image) -> torch.Tensor:
            for op in ops:
                img = op(img)
            tensor = transforms.ToTensor()(img)
            tensor = transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)(tensor)
            return tensor
        return apply

    return [
        compose([transforms.Resize((size, size))]),
        compose([transforms.Resize(resize_large), transforms.CenterCrop((size, size))]),
        compose([transforms.Resize((size, size)),
                 lambda x: TF.adjust_sharpness(x, 1.4),
                 lambda x: TF.adjust_contrast(x, 1.15)]),
    ]


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None,
                 reduction: str = 'mean', class_weight: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.register_buffer('class_weight', None)
        if class_weight is not None:
            weight_tensor = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer('class_weight', weight_tensor)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(inputs, dim=1)
        prob = log_prob.exp()
        focal = (1 - prob) ** self.gamma
        loss = -focal * log_prob
        if self.class_weight is not None:
            loss = loss * self.class_weight.unsqueeze(0)
        if self.alpha is not None:
            alpha_factor = torch.ones_like(inputs) * self.alpha
            alpha_factor.scatter_(1, targets.unsqueeze(1), 1 - self.alpha)
            loss = loss * alpha_factor
        loss = loss.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


@dataclass
class ExperimentConfig:
    name: str
    model_name: str
    image_size: int
    epochs: int
    optimizer: str
    lr: float
    weight_decay: float
    scheduler: str
    train_transform_builder: Optional[Callable[[int], transforms.Compose]] = None
    valid_transform_builder: Optional[Callable[[int], transforms.Compose]] = None
    tta_builder: Optional[Callable[[int], List[Callable[[Image.Image], torch.Tensor]]]] = None
    criterion: str = 'ce'
    criterion_kwargs: Dict[str, float] = field(default_factory=dict)
    sampler: str = 'none'
    batch_size: Optional[int] = None
    use_focus_loss: bool = False
    focus_weight: float = 4.0
    warmup_ratio: float = 0.1
    notes: str = ''


class ExperimentLogger:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, 'train.log')

    def log(self, message: str) -> None:
        timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(timestamped)
        with open(self.log_path, 'a') as f:
            f.write(timestamped + '\n')


def build_optimizer(model: nn.Module, config: ExperimentConfig, lr: float, weight_decay: float) -> optim.Optimizer:
    opt_name = config.optimizer.lower()
    if opt_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: optim.Optimizer, config: ExperimentConfig, epochs: int):
    if config.scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if config.scheduler == 'cosine_warmup':
        warmup_epochs = max(1, int(epochs * config.warmup_ratio))
        if warmup_epochs >= epochs:
            return CosineAnnealingLR(optimizer, T_max=epochs)
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    if config.scheduler == 'step30':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(epochs * 0.6), int(epochs * 0.85)],
            gamma=0.1
        )
    return None


def build_criterion(config: ExperimentConfig, device: torch.device,
                    criterion_kwargs: Dict[str, float], class_weights: Optional[List[float]] = None) -> nn.Module:
    if config.criterion == 'focal':
        if class_weights is not None:
            criterion_kwargs = {**criterion_kwargs, 'class_weight': class_weights}
        crit = FocalLoss(**criterion_kwargs)
    else:
        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
            crit = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            crit = nn.CrossEntropyLoss()
    return crit.to(device)


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            imgs, targets = batch[:2]
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def build_sampler(dataset: DigitData, focus_examples: Iterable[str], focus_weight: float) -> Optional[WeightedRandomSampler]:
    focus_set = set(focus_examples)
    if not focus_set or focus_weight <= 1.0:
        return None
    weights = torch.ones(len(dataset), dtype=torch.double)
    for idx, rel_path in enumerate(dataset.image_files):
        if rel_path in focus_set:
            weights[idx] = focus_weight
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_class_weights(dataset: DigitData, focus_examples: Set[str], focus_weight: float,
                          num_classes: int = 10) -> List[float]:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for rel_path in dataset.image_files:
        label = int(rel_path.split('/')[0])
        weight = focus_weight if rel_path in focus_examples else 1.0
        counts[label] += weight
    counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    inv = counts.sum() / counts
    normalized = inv / inv.mean()
    return normalized.tolist()


def evaluate_with_metrics(model: nn.Module, dataset: DigitData, device: torch.device, batch_size: int,
                          num_workers: int, tta_transforms: Optional[List[Callable[[Image.Image], torch.Tensor]]],
                          small_set: Iterable[str], noisy_set: Iterable[str],
                          max_error_log: int, output_dir: str) -> Dict[str, object]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == 'cuda')
    small_lookup = set(small_set)
    noisy_lookup = set(noisy_set)
    total = correct = 0
    small_total = small_correct = 0
    noisy_total = noisy_correct = 0
    error_log: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            imgs, targets, paths = batch
            targets = targets.to(device)
            if tta_transforms:
                preds_list: List[torch.Tensor] = []
                for rel_path in paths:
                    full_path = os.path.join(dataset.path, rel_path)
                    with Image.open(full_path) as pil_img:
                        pil_img = pil_img.convert('RGB')
                        logits: List[torch.Tensor] = []
                        for tta in tta_transforms:
                            tensor = tta(pil_img).unsqueeze(0).to(device)
                            logits.append(model(tensor))
                        stacked = torch.stack(logits, dim=0)
                        preds_list.append(stacked.mean(dim=0))
                outputs = torch.cat(preds_list, dim=0)
            else:
                imgs = imgs.to(device)
                outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            matches = preds.eq(targets)
            batch_size_curr = targets.size(0)
            correct += matches.sum().item()
            total += batch_size_curr
            for idx, rel_path in enumerate(paths):
                match = matches[idx].item()
                if rel_path in small_lookup:
                    small_total += 1
                    small_correct += match
                if rel_path in noisy_lookup:
                    noisy_total += 1
                    noisy_correct += match
                if not match and len(error_log) < max_error_log:
                    error_log.append({
                        'path': rel_path,
                        'target': int(targets[idx].item()),
                        'pred': int(preds[idx].item()),
                    })
    metrics = {
        'overall_acc': correct / total if total else 0.0,
        'small_acc': (small_correct / small_total) if small_total else None,
        'noisy_acc': (noisy_correct / noisy_total) if noisy_total else None,
        'total_samples': total,
        'small_samples': small_total,
        'noisy_samples': noisy_total,
        'error_examples': error_log,
    }
    metrics_path = os.path.join(output_dir, 'eval_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_experiment(config: ExperimentConfig, args, device: torch.device,
                   small_valid_set: Sequence[str], noisy_set: Sequence[str],
                   hard_set: Sequence[str], small_train_set: Sequence[str]) -> Dict[str, object]:
    output_dir = os.path.join(args.output_dir, config.name)
    os.makedirs(output_dir, exist_ok=True)
    logger = ExperimentLogger(output_dir)
    logger.log(f"Running experiment '{config.name}' ({config.notes})")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    epochs = args.epochs if args.epochs is not None else config.epochs
    lr = args.lr if args.lr is not None else config.lr
    weight_decay = args.weight_decay if args.weight_decay is not None else config.weight_decay

    train_transform = config.train_transform_builder(config.image_size) if config.train_transform_builder else None
    valid_transform = config.valid_transform_builder(config.image_size) if config.valid_transform_builder else build_eval_transform(config.image_size)
    tta_transforms = config.tta_builder(config.image_size) if config.tta_builder else None

    train_dataset = DigitData(args.data_dir, config.image_size, 'train', transform=train_transform)
    valid_dataset = DigitData(args.data_dir, config.image_size, 'valid', transform=valid_transform)
    eval_dataset = DigitData(args.data_dir, config.image_size, 'valid', transform=valid_transform, return_path=True)

    focus_examples = set()
    focus_examples.update([p for p in small_train_set])
    focus_examples.update([p for p in hard_set])
    focus_matches = {p for p in train_dataset.image_files if p in focus_examples}

    sampler_weight = args.focus_sampler_weight if getattr(args, 'focus_sampler_weight', None) is not None else args.hard_example_weight
    if sampler_weight is None:
        sampler_weight = config.focus_weight
    sampler: Optional[WeightedRandomSampler] = None
    if config.sampler == 'hard':
        sampler = build_sampler(train_dataset, focus_matches, sampler_weight)
        if sampler:
            logger.log(f"Using WeightedRandomSampler with weight={sampler_weight} (matched {len(focus_matches)} samples)")
        else:
            logger.log("Focus sampler requested but no matching samples found; falling back to shuffle.")

    batch_size = config.batch_size or args.batch_size

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=sampler is None, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    model = models.__dict__[config.model_name](num_classes=10).to(device)
    optimizer = build_optimizer(model, config, lr, weight_decay)
    scheduler = build_scheduler(optimizer, config, epochs)
    criterion_kwargs = dict(config.criterion_kwargs)
    if config.criterion == 'focal' and args.focal_alpha is not None:
        criterion_kwargs['alpha'] = args.focal_alpha
    class_weights = None
    loss_focus_weight = None
    if focus_matches and (config.use_focus_loss or getattr(args, 'use_focus_loss', False)):
        loss_focus_weight = args.focus_loss_weight if getattr(args, 'focus_loss_weight', None) is not None else config.focus_weight
        class_weights = compute_class_weights(train_dataset, focus_matches, loss_focus_weight)
        logger.log(f"Applying class-weighted loss (focus_weight={loss_focus_weight}, weights={['{:.2f}'.format(w) for w in class_weights]})")
    criterion = build_criterion(config, device, criterion_kwargs, class_weights)
    if config.criterion == 'focal':
        logger.log(f"FocalLoss settings: gamma={criterion_kwargs.get('gamma', 2.0)}, alpha={criterion_kwargs.get('alpha')}")

    logger.log(f"Model: {config.model_name}, optimizer: {config.optimizer}, lr: {lr}, wd: {weight_decay}, "
               f"epochs: {epochs}, batch_size: {batch_size}")
    best_state = None
    best_acc = 0.0
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device.type == 'cuda')
        val_acc = evaluate_accuracy(model, valid_loader, device)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc})
        logger.log(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if scheduler:
            scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_state = {
                'model_state': cpu_state,
                'epoch': epoch + 1,
                'val_acc': best_acc,
                'lr': optimizer.param_groups[0]['lr'],
            }
            torch.save(best_state, os.path.join(output_dir, 'best_model.pth'))

    if best_state:
        model.load_state_dict(best_state['model_state'])

    metrics = evaluate_with_metrics(model, eval_dataset, device, batch_size,
                                    args.num_workers, tta_transforms,
                                    small_valid_set, noisy_set, args.max_error_log, output_dir)

    summary = {
        'best_val_acc': best_acc,
        'final_metrics': metrics,
        'history': history,
        'config': {
            'lr': lr,
            'weight_decay': weight_decay,
            'epochs': epochs,
            'batch_size': batch_size,
            'criterion_kwargs': criterion_kwargs,
            'optimizer': config.optimizer,
            'scheduler': config.scheduler,
            'image_size': config.image_size,
            'sampler_weight': sampler_weight if config.sampler == 'hard' else None,
            'focus_matches': len(focus_matches),
            'use_focus_loss': bool(config.use_focus_loss or getattr(args, 'use_focus_loss', False)),
            'focus_loss_weight': loss_focus_weight,
            'focus_weight': config.focus_weight,
        },
        'notes': config.notes,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def build_experiments(args) -> List[ExperimentConfig]:
    base_epochs = args.epochs if args.epochs is not None else 60
    base_batch = args.batch_size
    base_scheduler = 'cosine'

    return [
        ExperimentConfig(
            name='baseline_resnet32',
            model_name='resnet32',
            image_size=64,
            epochs=base_epochs,
            optimizer='sgd',
            lr=0.1,
            weight_decay=5e-4,
            scheduler=base_scheduler,
            batch_size=base_batch,
            notes='Baseline ResNet32 + SGD(momentum) + 기본 전처리',
        ),
        ExperimentConfig(
            name='baseline_resnet50',
            model_name='resnet50',
            image_size=64,
            epochs=base_epochs,
            optimizer='sgd',
            lr=0.05,
            weight_decay=5e-4,
            scheduler=base_scheduler,
            batch_size=base_batch,
            notes='ResNet50 + SGD(momentum) + 기본 전처리',
        ),
        ExperimentConfig(
            name='augmented_resnet20',
            model_name='resnet20',
            image_size=64,
            epochs=base_epochs,
            optimizer='adam',
            lr=0.001,
            weight_decay=5e-4,
            scheduler=base_scheduler,
            train_transform_builder=build_heavy_aug_transform,
            notes='강화된 데이터 증강 + 하드 샘플 가중',
            sampler='hard',
            batch_size=base_batch,
        ),
        ExperimentConfig(
            name='superres_tta_resnet20',
            model_name='resnet20',
            image_size=96,
            epochs=base_epochs,
            optimizer='adamw',
            lr=0.0007,
            weight_decay=3e-4,
            scheduler=base_scheduler,
            train_transform_builder=build_superres_train_transform,
            valid_transform_builder=build_eval_transform,
            tta_builder=build_superres_tta,
            notes='고해상도 학습 + 다중 스케일 TTA',
            sampler='hard',
            batch_size=base_batch,
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description='Run multiple digit classification experiments and compare metrics.')
    parser.add_argument('--data_dir', type=str, default='../../digit_data', help='Path to digit_data root directory.')
    parser.add_argument('--output_dir', type=str, default='experiment_runs', help='Directory to store experiment outputs.')
    parser.add_argument('--epochs', type=int, default=None, help='Override epoch count for all experiments.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate for all experiments.')
    parser.add_argument('--weight_decay', type=float, default=None, help='Override weight decay for all experiments.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None, help='cuda, cpu or auto detect.')
    parser.add_argument('--small_list', type=str, default=None, help='Path to txt file listing small-digit validation samples.')
    parser.add_argument('--noisy_list', type=str, default=None, help='Path to txt file listing noisy validation samples.')
    parser.add_argument('--hard_list', type=str, default=None, help='Path to txt file listing hard training samples.')
    parser.add_argument('--hard_example_weight', type=float, default=3.0)
    parser.add_argument('--small_train_list', type=str, default=None, help='Path to txt file listing small-digit training samples.')
    parser.add_argument('--focus_sampler_weight', type=float, default=None,
                        help='Override the sampler weight applied to focus (small/hard) samples.')
    parser.add_argument('--use_focus_loss', action='store_true',
                        help='Force class-weighted loss using focus sample statistics.')
    parser.add_argument('--focus_loss_weight', type=float, default=None,
                        help='Override the focus weighting factor when building class-weighted loss.')
    parser.add_argument('--max_error_log', type=int, default=50, help='Maximum number of misclassified samples to log per experiment.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiments', type=str, default=None,
                        help='Comma separated list of experiment names to run (subset of predefined experiments).')
    parser.add_argument('--focal_alpha', type=float, default=None,
                        help='Alpha value for focal loss (balance hard/easy samples).')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    os.makedirs(args.output_dir, exist_ok=True)

    # Fairness guard: constrain shared batch size to avoid OOM and keep identical across experiments
    if args.batch_size > 64:
        print(f"[Fairness] Requested batch size {args.batch_size} is capped to 64 for all experiments "
              "to ensure equal training budget and stable GPU memory usage.")
        args.batch_size = 64

    small_set = load_path_list(args.small_list)
    noisy_set = load_path_list(args.noisy_list)
    hard_set = load_path_list(args.hard_list)
    small_train_set = load_path_list(args.small_train_list)

    experiments = build_experiments(args)
    if args.experiments:
        selected = set(name.strip() for name in args.experiments.split(',') if name.strip())
        experiments = [cfg for cfg in experiments if cfg.name in selected]
        if not experiments:
            raise ValueError('No matching experiments found for the provided --experiments argument.')

    summaries = {}
    for config in experiments:
        summaries[config.name] = run_experiment(config, args, device, small_set, noisy_set, hard_set, small_train_set)

    summary_path = os.path.join(args.output_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)

    print('\n=== Experiment Comparison ===')
    for name, result in summaries.items():
        metrics = result['final_metrics']
        print(f"{name:25s} | best_val_acc: {result['best_val_acc']:.4f} | "
              f"overall: {metrics['overall_acc']:.4f} | "
              f"small: {metrics['small_acc'] if metrics['small_acc'] is not None else 'n/a'} | "
              f"noisy: {metrics['noisy_acc'] if metrics['noisy_acc'] is not None else 'n/a'}")


if __name__ == '__main__':
    main()
