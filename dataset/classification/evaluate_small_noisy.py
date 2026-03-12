#!/usr/bin/env python3

import os
import torch
from torch.utils.data import DataLoader
from dataloader import DigitData

# 모델 import
from models.resnet import resnet20, resnet32
from models.convnext_custom import convnext_tiny_custom
from models.swin_custom import swin_tiny_custom

import torchvision.transforms as T
from typing import List, Sequence

# Dataset normalization (학습 때와 동일해야 함)
DATASET_MEAN = [0.80048384, 0.44734452, 0.50106468]
DATASET_STD  = [0.22327253, 0.29523788, 0.24583565]


# -------------------------------
# Helper functions
# -------------------------------

def load_path_list(path: str) -> Sequence[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_transform(model_name: str):
    """모델 구조에 따라 입력 크기와 전처리를 자동 설정한다."""
    if model_name in ["convnext", "convnext_tiny", "swin", "swin_tiny"]:
        size = 224
    else:
        size = 64
    
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(DATASET_MEAN, DATASET_STD)
    ]), size


def create_model(model_name: str):
    if model_name == "resnet20":
        return resnet20(num_classes=10)
    if model_name == "resnet32":
        return resnet32(num_classes=10)
    if model_name in ["convnext", "convnext_tiny"]:
        return convnext_tiny_custom(10)
    if model_name in ["swin", "swin_tiny"]:
        return swin_tiny_custom(10)

    raise ValueError("Unknown model name:", model_name)


# -------------------------------
# Accuracy evaluation
# -------------------------------

def evaluate_subset(model, dataset, subset_paths, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    subset = set(subset_paths)
    total, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs).argmax(1)

            for i, rel_path in enumerate(paths):
                if rel_path in subset:
                    total += 1
                    correct += int(preds[i] == labels[i])

    return correct / total if total > 0 else None


def evaluate_all(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    total, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs).argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total


# -------------------------------
# Main procedure
# -------------------------------

def main(model_name, model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"\n[Evaluation] Loading model: {model_name}")
    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load transforms & dataset
    transform, size = build_transform(model_name)
    eval_dataset = DigitData(data_dir, size=size, split="valid", transform=transform, return_path=True)

    # Load small/noisy list files
    small_list = load_path_list(os.path.join(data_dir, "small_valid.txt"))
    noisy_list = load_path_list(os.path.join(data_dir, "noisy_valid.txt"))

    # Evaluate
    acc_all = evaluate_all(model, eval_dataset, device)
    acc_small = evaluate_subset(model, eval_dataset, small_list, device)
    acc_noisy = evaluate_subset(model, eval_dataset, noisy_list, device)

    # Print results
    print(f"\n===== Evaluation Results ({model_name}) =====")
    print(f"Overall Accuracy:     {acc_all:.4f}")
    print(f"Small-digit Accuracy: {acc_small:.4f}")
    print(f"Noisy-digit Accuracy: {acc_noisy:.4f}")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("model_path")
    parser.add_argument("--data_dir", type=str, default="../digit_data")

    args = parser.parse_args()
    main(args.model_name, args.model_path, args.data_dir)