import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CLASSIFICATION_ROOT = REPO_ROOT / 'dataset' / 'classification'

for path in (SCRIPT_DIR, CLASSIFICATION_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import models  # type: ignore
from dataloader import DigitData  # type: ignore
from utils import train  # type: ignore
import se_resnet  # type: ignore


DATASET_MEAN = [0.80048384, 0.44734452, 0.50106468]
DATASET_STD = [0.22327253, 0.29523788, 0.24583565]


TransformBuilder = Callable[[int], transforms.Compose]
TTAFunction = Callable[[Image.Image], torch.Tensor]


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_identifier_list(path: Optional[str]) -> List[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def build_eval_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])


def build_light_aug_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size + 4, size + 4)),
        transforms.RandomCrop(size),
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])


def build_medium_aug_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.15)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                                       saturation=0.2, hue=0.04)], p=0.7),
        transforms.RandomRotation(degrees=6),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])


def build_strong_aug_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.8, 1.35)),
        transforms.RandomApply([transforms.RandomAffine(degrees=8, translate=(0.08, 0.08),
                                                        scale=(0.9, 1.15), shear=(-6, 6))], p=0.7),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.35, contrast=0.4,
                                                       saturation=0.3, hue=0.05)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.4),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.15)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.12), value='random'),
    ])


def build_resolution_focus_transform(size: int) -> transforms.Compose:
    upscale = int(size * 1.6)
    return transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.3)),
            transforms.Compose([
                transforms.Resize(upscale),
                transforms.RandomCrop(size),
            ]),
            transforms.Compose([
                transforms.Resize(int(size * 1.8)),
                transforms.CenterCrop(size),
            ]),
        ]),
        transforms.RandomApply([transforms.RandomAdjustSharpness(1.5)], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.35,
                                                       saturation=0.3, hue=0.04)], p=0.6),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])


def build_balanced_tta(size: int) -> List[TTAFunction]:
    upscale = int(size * 1.2)

    def compose(ops: List[Callable[[Image.Image], Image.Image]]) -> TTAFunction:
        def apply(img: Image.Image) -> torch.Tensor:
            for op in ops:
                img = op(img)
            tensor = transforms.ToTensor()(img)
            tensor = transforms.Normalize(DATASET_MEAN, DATASET_STD)(tensor)
            return tensor
        return apply

    return [
        compose([transforms.Resize((size, size))]),
        compose([transforms.Resize((upscale, upscale)), transforms.CenterCrop((size, size))]),
        compose([
            transforms.Resize((size, size)),
            lambda x: TF.adjust_brightness(x, 1.1),
            lambda x: TF.adjust_contrast(x, 1.15),
        ]),
        compose([
            transforms.Resize((size, size)),
            lambda x: TF.adjust_sharpness(x, 1.2),
        ]),
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
        log_prob = nn.functional.log_softmax(inputs, dim=1)
        prob = log_prob.exp()
        focal = (1.0 - prob) ** self.gamma
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
    image_size: int = 64
    epochs: int = 60
    optimizer: str = 'sgd'
    lr: float = 0.1
    weight_decay: float = 5e-4
    scheduler: str = 'cosine'
    batch_size: Optional[int] = None
    train_transform_builder: Optional[TransformBuilder] = None
    valid_transform_builder: Optional[TransformBuilder] = None
    tta_builder: Optional[Callable[[int], List[TTAFunction]]] = None
    sampler_focus: Dict[str, float] = field(default_factory=dict)
    sampler_strategy: str = 'none'
    class_weight_focus: bool = False
    class_weight_factor: float = 2.0
    loss_type: str = 'ce'
    loss_kwargs: Dict[str, float] = field(default_factory=dict)
    warmup_ratio: float = 0.1
    notes: str = ''


class ExperimentLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / 'train.log'

    def log(self, message: str) -> None:
        timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(timestamped)
        with open(self.log_path, 'a') as f:
            f.write(timestamped + '\n')


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg and device_arg != 'auto':
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_optimizer(model: nn.Module, config: ExperimentConfig, lr: float, weight_decay: float) -> optim.Optimizer:
    if config.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if config.optimizer.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: optim.Optimizer, config: ExperimentConfig, epochs: int):
    if config.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if config.scheduler == 'cosine_warmup':
        warmup_epochs = max(1, int(epochs * config.warmup_ratio))
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    if config.scheduler == 'step':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(epochs * 0.6), int(epochs * 0.85)],
            gamma=0.1,
        )
    return None


def build_criterion(config: ExperimentConfig, device: torch.device,
                    class_weights: Optional[List[float]]) -> nn.Module:
    if config.loss_type == 'focal':
        kwargs = dict(config.loss_kwargs)
        if class_weights is not None:
            kwargs['class_weight'] = class_weights
        criterion = FocalLoss(**kwargs)
    else:
        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
    return criterion.to(device)


def build_focus_lookup(sample_groups: Dict[str, Sequence[str]],
                       focus_config: Dict[str, float]) -> Dict[str, float]:
    lookup: Dict[str, float] = {}
    for group, weight in focus_config.items():
        samples = sample_groups.get(group, [])
        if not samples:
            continue
        for rel_path in samples:
            lookup[rel_path] = max(lookup.get(rel_path, 1.0), weight)
    return lookup


def build_sampler(dataset: DigitData, focus_lookup: Dict[str, float]) -> Optional[WeightedRandomSampler]:
    if not focus_lookup:
        return None
    weights = torch.ones(len(dataset), dtype=torch.double)
    matches = 0
    for idx, rel_path in enumerate(dataset.image_files):
        if rel_path in focus_lookup:
            weights[idx] = focus_lookup[rel_path]
            matches += 1
    if matches == 0:
        return None
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_class_weights(dataset: DigitData, focus_lookup: Dict[str, float],
                          focus_factor: float, num_classes: int = 10) -> List[float]:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for rel_path in dataset.image_files:
        label = int(rel_path.split('/')[0])
        weight = focus_factor if rel_path in focus_lookup else 1.0
        counts[label] += weight
    counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    inverse = counts.sum() / counts
    normalized = inverse / inverse.mean()
    return normalized.tolist()


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def evaluate_with_details(model: nn.Module, dataset: DigitData, device: torch.device,
                          batch_size: int, num_workers: int, tta_transforms: Optional[List[TTAFunction]],
                          subset_lookup: Dict[str, set], max_error_log: int) -> Dict[str, object]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == 'cuda')
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    subset_stats = {name: {'total': 0, 'correct': 0} for name in subset_lookup}
    total = correct = 0
    error_examples: List[Dict[str, int]] = []
    model.eval()
    with torch.no_grad():
        for imgs, targets, paths in loader:
            targets = targets.to(device)
            if tta_transforms:
                logits_list: List[torch.Tensor] = []
                for rel_path in paths:
                    full_path = os.path.join(dataset.path, rel_path)
                    with Image.open(full_path) as pil_img:
                        pil_img = pil_img.convert('RGB')
                        tta_logits: List[torch.Tensor] = []
                        for tta in tta_transforms:
                            tensor = tta(pil_img).unsqueeze(0).to(device)
                            tta_logits.append(model(tensor))
                        stacked = torch.stack(tta_logits, dim=0).mean(dim=0)
                        logits_list.append(stacked)
                outputs = torch.cat(logits_list, dim=0)
            else:
                imgs = imgs.to(device)
                outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            matches = preds.eq(targets)
            total += targets.size(0)
            correct += matches.sum().item()
            for idx, rel_path in enumerate(paths):
                target = int(targets[idx].item())
                pred = int(preds[idx].item())
                confusion[target, pred] += 1
                for subset_name, subset in subset_lookup.items():
                    if rel_path in subset:
                        subset_stats[subset_name]['total'] += 1
                        subset_stats[subset_name]['correct'] += int(pred == target)
                if pred != target and len(error_examples) < max_error_log:
                    error_examples.append({'path': rel_path, 'target': target, 'pred': pred})
    subset_metrics = {
        name: (stats['correct'] / stats['total']) if stats['total'] else None
        for name, stats in subset_stats.items()
    }
    per_class = []
    for cls in range(10):
        row_total = confusion[cls].sum().item()
        per_class.append((confusion[cls, cls].item() / row_total) if row_total else None)
    return {
        'overall_acc': correct / total if total else 0.0,
        'subset_acc': subset_metrics,
        'per_class_acc': per_class,
        'confusion_matrix': confusion.tolist(),
        'total_samples': total,
        'error_examples': error_examples,
    }


def load_sample_lists(args) -> Dict[str, List[str]]:
    return {
        'small_train': load_identifier_list(args.small_train_list),
        'small_valid': load_identifier_list(args.small_valid_list),
        'noisy_train': load_identifier_list(args.noisy_train_list),
        'noisy_valid': load_identifier_list(args.noisy_valid_list),
        'hard_train': load_identifier_list(args.hard_train_list),
        'hard_valid': load_identifier_list(args.hard_valid_list),
    }


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == 'resnet32_se':
        return se_resnet.seresnet32(num_classes=num_classes)
    if not hasattr(models, name):
        raise ValueError(f"Unknown model name: {name}")
    return getattr(models, name)(num_classes=num_classes)


def run_experiment(config: ExperimentConfig, args, device: torch.device,
                   sample_lists: Dict[str, List[str]]) -> Dict[str, object]:
    output_dir = Path(args.output_dir) / config.name
    logger = ExperimentLogger(output_dir)
    logger.log(f"Running experiment '{config.name}' ({config.notes})")

    train_transform = config.train_transform_builder(config.image_size) if config.train_transform_builder else build_eval_transform(config.image_size)
    valid_transform = config.valid_transform_builder(config.image_size) if config.valid_transform_builder else build_eval_transform(config.image_size)
    tta_transforms = config.tta_builder(config.image_size) if config.tta_builder else None

    train_dataset = DigitData(args.data_dir, config.image_size, 'train', transform=train_transform)
    valid_dataset = DigitData(args.data_dir, config.image_size, 'valid', transform=valid_transform)
    eval_dataset = DigitData(args.data_dir, config.image_size, 'valid', transform=valid_transform, return_path=True)

    focus_lookup = build_focus_lookup(
        {k: sample_lists.get(k, []) for k in ('small_train', 'noisy_train', 'hard_train')},
        config.sampler_focus)

    sampler = None
    if config.sampler_strategy == 'weighted':
        sampler = build_sampler(train_dataset, focus_lookup)
        if sampler:
            logger.log(f"Weighted sampler enabled: matched {len(focus_lookup)} focus samples.")
        else:
            logger.log("Sampler requested but no matching focus samples were found.")

    class_weights = None
    if config.class_weight_focus:
        class_weights = compute_class_weights(train_dataset, focus_lookup, config.class_weight_factor)
        logger.log(f"Class weights applied (focus_factor={config.class_weight_factor}): "
                   f"{['{:.2f}'.format(w) for w in class_weights]}")

    batch_size = config.batch_size or args.batch_size
    epochs = args.epochs or config.epochs
    lr = args.lr or config.lr
    weight_decay = args.weight_decay or config.weight_decay

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=args.num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_model(config.model_name).to(device)
    optimizer = build_optimizer(model, config, lr, weight_decay)
    scheduler = build_scheduler(optimizer, config, epochs)
    criterion = build_criterion(config, device, class_weights)

    history: List[Dict[str, float]] = []
    best_state = None
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device.type == 'cuda')
        val_acc = evaluate_accuracy(model, valid_loader, device)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc})
        logger.log(f"Epoch {epoch + 1}/{epochs} | Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")
        if scheduler:
            scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr'],
            }
            torch.save(best_state, output_dir / 'best_model.pth')

    if best_state:
        model.load_state_dict(best_state['model_state'])

    subset_lookup = {
        'small': set(sample_lists.get('small_valid', [])),
        'noisy': set(sample_lists.get('noisy_valid', [])),
        'hard': set(sample_lists.get('hard_valid', [])),
    }
    metrics = evaluate_with_details(model, eval_dataset, device, batch_size,
                                    args.num_workers, tta_transforms,
                                    subset_lookup, args.max_error_log)

    with open(output_dir / 'eval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    summary = {
        'name': config.name,
        'notes': config.notes,
        'best_val_acc': best_acc,
        'final_metrics': metrics,
        'history': history,
        'config': {
            'lr': lr,
            'weight_decay': weight_decay,
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': config.optimizer,
            'scheduler': config.scheduler,
            'image_size': config.image_size,
            'sampler_strategy': config.sampler_strategy,
            'focus_count': len(focus_lookup),
            'class_weight_focus': config.class_weight_focus,
            'loss_type': config.loss_type,
        },
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.log(f"Finished experiment '{config.name}' | Best Val Acc {best_acc:.4f} | "
               f"Eval Overall {metrics['overall_acc']:.4f} | Small {metrics['subset_acc'].get('small')} | "
               f"Noisy {metrics['subset_acc'].get('noisy')}")
    return summary


def build_phase3_experiments(args) -> Dict[str, List[ExperimentConfig]]:
    base = {
        'epochs': args.epochs or 60,
        'batch_size': args.batch_size,
    }
    return {
        'data': [
            ExperimentConfig(
                name='data_light_aug_resnet32',
                model_name='resnet32',
                train_transform_builder=build_light_aug_transform,
                notes='Light aug baseline to gauge reference performance.',
            ),
            ExperimentConfig(
                name='data_medium_aug_resnet32',
                model_name='resnet32',
                train_transform_builder=build_medium_aug_transform,
                notes='Medium augmentation for balanced robustness.',
            ),
            ExperimentConfig(
                name='data_strong_aug_resnet32',
                model_name='resnet32',
                train_transform_builder=build_strong_aug_transform,
                sampler_strategy='weighted',
                sampler_focus={'small_train': 2.0, 'noisy_train': 2.0, 'hard_train': 2.5},
                notes='Strong aug with focus sampler to stress-test realism.',
            ),
            ExperimentConfig(
                name='data_focus_sampler_resnet32',
                model_name='resnet32',
                train_transform_builder=build_medium_aug_transform,
                sampler_strategy='weighted',
                sampler_focus={'small_train': 2.0, 'noisy_train': 2.0, 'hard_train': 3.0},
                class_weight_focus=True,
                class_weight_factor=2.5,
                notes='Explicit hard-sample oversampling + class-weighted loss.',
            ),
        ],
        'model': [
            ExperimentConfig(
                name='model_resnet32_baseline',
                model_name='resnet32',
                train_transform_builder=build_light_aug_transform,
                notes='Reference ResNet32 baseline.',
            ),
            ExperimentConfig(
                name='model_resnet44_deeper',
                model_name='resnet44',
                train_transform_builder=build_light_aug_transform,
                notes='Deeper residual stack to test representation gain.',
            ),
            ExperimentConfig(
                name='model_resnet32x4_wide',
                model_name='resnet32x4',
                train_transform_builder=build_light_aug_transform,
                optimizer='adamw',
                lr=0.0008,
                notes='Wider network (x4 channels) for richer features.',
            ),
            ExperimentConfig(
                name='model_resnet32_se',
                model_name='resnet32_se',
                train_transform_builder=build_light_aug_transform,
                optimizer='adamw',
                lr=0.0007,
                notes='Channel attention (SE) injected into ResNet32 blocks.',
            ),
        ],
        'loss': [
            ExperimentConfig(
                name='loss_ce_cosine',
                model_name='resnet32',
                train_transform_builder=build_light_aug_transform,
                scheduler='cosine',
                notes='CrossEntropy + Cosine LR baseline.',
            ),
            ExperimentConfig(
                name='loss_ce_step',
                model_name='resnet32',
                train_transform_builder=build_light_aug_transform,
                scheduler='step',
                notes='CrossEntropy + Step LR (milestones 60/85%).',
            ),
            ExperimentConfig(
                name='loss_ce_cosine_warmup',
                model_name='resnet32',
                train_transform_builder=build_light_aug_transform,
                scheduler='cosine_warmup',
                warmup_ratio=0.15,
                notes='Cosine scheduler with warmup for smoother convergence.',
            ),
            ExperimentConfig(
                name='loss_class_weight_focus',
                model_name='resnet32',
                train_transform_builder=build_medium_aug_transform,
                class_weight_focus=True,
                class_weight_factor=3.0,
                sampler_focus={'small_train': 2.0, 'noisy_train': 2.0},
                notes='CrossEntropy + class weights for noisy/small emphasis.',
            ),
            ExperimentConfig(
                name='loss_focal_gamma2',
                model_name='resnet32',
                train_transform_builder=build_medium_aug_transform,
                loss_type='focal',
                loss_kwargs={'gamma': 2.0, 'alpha': 0.25},
                sampler_focus={'small_train': 2.0, 'noisy_train': 2.0},
                notes='Focal loss targeting hard samples.',
            ),
        ],
        'superres': [
            ExperimentConfig(
                name='superres_64_baseline',
                model_name='resnet32',
                image_size=64,
                train_transform_builder=build_resolution_focus_transform,
                notes='64x64 reference with focus-aware transforms.',
            ),
            ExperimentConfig(
                name='superres_96',
                model_name='resnet32',
                image_size=96,
                train_transform_builder=build_resolution_focus_transform,
                notes='96x96 input resolution comparison.',
            ),
            ExperimentConfig(
                name='superres_128',
                model_name='resnet32',
                image_size=128,
                train_transform_builder=build_resolution_focus_transform,
                notes='128x128 resolution stress test.',
            ),
            ExperimentConfig(
                name='superres_balanced_tta',
                model_name='resnet32',
                image_size=96,
                train_transform_builder=build_resolution_focus_transform,
                tta_builder=build_balanced_tta,
                notes='96x96 + brightness/contrast balanced TTA.',
            ),
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 3 digit classification experiments.')
    default_data = REPO_ROOT / 'dataset' / 'digit_data'
    parser.add_argument('--data_dir', type=str, default=str(default_data))
    parser.add_argument('--output_dir', type=str, default=str(REPO_ROOT / 'phase3_runs'))
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--groups', type=str, default='data,model,loss,superres',
                        help='Comma separated experiment groups to run (or "all").')
    parser.add_argument('--max_error_log', type=int, default=60)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--small_train_list', type=str, default=str(default_data / 'small_train.txt'))
    parser.add_argument('--small_valid_list', type=str, default=str(default_data / 'small_valid.txt'))
    parser.add_argument('--noisy_train_list', type=str, default=str(default_data / 'noisy_train.txt'))
    parser.add_argument('--noisy_valid_list', type=str, default=str(default_data / 'noisy_valid.txt'))
    parser.add_argument('--hard_train_list', type=str, default=str(default_data / 'hard_train.txt'))
    parser.add_argument('--hard_valid_list', type=str, default=str(default_data / 'hard_valid.txt'))
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    sample_lists = load_sample_lists(args)

    experiment_groups = build_phase3_experiments(args)
    requested = [g.strip().lower() for g in args.groups.split(',')]
    selected_groups = experiment_groups.keys() if 'all' in requested else requested

    all_results: List[Dict[str, object]] = []
    for group in selected_groups:
        if group not in experiment_groups:
            print(f"[WARN] Unknown group '{group}' skipped.")
            continue
        print(f"\n=== Running group: {group} ===")
        for config in experiment_groups[group]:
            result = run_experiment(config, args, device, sample_lists)
            all_results.append(result)

    summary_path = Path(args.output_dir) / 'phase3_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCompleted {len(all_results)} experiments. Summary saved to {summary_path}.")
    print("Topline results:")
    for result in all_results:
        metrics = result['final_metrics']
        subsets = metrics['subset_acc']
        print(f"{result['name']:30s} | overall {metrics['overall_acc']:.4f} | "
              f"small {subsets.get('small')} | noisy {subsets.get('noisy')} | "
              f"hard {subsets.get('hard')}")


if __name__ == '__main__':
    main()
