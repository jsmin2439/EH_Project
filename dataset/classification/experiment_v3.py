"""
experiment_v3.py
──────────────────────────────────────────────
3차 연구 전용 독립 실험 파일

지원 모델:
- resnet20
- resnet32
- convnext_tiny_custom
- swin_tiny_custom

기존 run_experiments.py 구조를 참고하지만,
강화된 config 시스템 없이 “단일 모델 실험”만 수행.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# ----------------------------
# models (3차 실험용)
# ----------------------------
from models.resnet import resnet20, resnet32
from models.convnext_custom import convnext_tiny_custom
from models.swin_custom import swin_tiny_custom

from dataloader import DigitData

# ----------------------------
# Dataset normalization
# ----------------------------
DATASET_MEAN = [0.80048384, 0.44734452, 0.50106468]
DATASET_STD  = [0.22327253, 0.29523788, 0.24583565]


# ===========================================
# 모델 선택
# ===========================================
def create_model(model_name: str, num_classes=10):
    model_name = model_name.lower()

    if model_name == "resnet20":
        return resnet20(num_classes=num_classes)
    elif model_name == "resnet32":
        return resnet32(num_classes=num_classes)
    elif model_name in ["convnext", "convnext_tiny"]:
        return convnext_tiny_custom(num_classes)
    elif model_name in ["swin", "swin_tiny"]:
        return swin_tiny_custom(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ===========================================
# Train function
# ===========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# ===========================================
# Eval function
# ===========================================
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Valid", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ===========================================
# Transform builder
# ===========================================
import torchvision.transforms as T

def build_transform(model_name):
    """
    ResNet → 64~96px  
    ConvNeXt/Swin → 224px 입력
    """
    if model_name in ["swin", "swin_tiny", "convnext", "convnext_tiny"]:
        size = 224
    else:
        size = 64

    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(DATASET_MEAN, DATASET_STD),
    ])


# ===========================================
# Run Single Experiment
# ===========================================
def run_experiment(
        model_name="resnet20",
        data_dir="../../digit_data",
        output_dir="./experiment_v3_results",
        epochs=20,
        batch_size=64,
        lr=1e-4
):
    # -----------------------
    # device 설정
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # 출력 디렉토리 생성
    # -----------------------
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)

    # -----------------------
    # 모델 불러오기
    # -----------------------
    print(f"\n[3차 연구] Loading model: {model_name}")
    model = create_model(model_name).to(device)

    # -----------------------
    # 데이터셋 로드
    # -----------------------
    transform = build_transform(model_name)
    train_dataset = DigitData(data_dir, "train", transform=transform)
    valid_dataset = DigitData(data_dir, "valid", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # -----------------------
    # Loss / Optimizer
    # -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------
    # Train Loop
    # -----------------------
    history = []

    for ep in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, valid_loader, device)

        print(f"[Epoch {ep}/{epochs}] "
              f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        # save best
        if ep == epochs:
            torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pth"))

    # -----------------------
    # Save Summary
    # -----------------------
    summary = {
        "model": model_name,
        "epochs": epochs,
        "final_val_acc": history[-1]["val_acc"],
        "history": history
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Done] {model_name} final_val_acc={summary['final_val_acc']:.4f}")
    print(f"Saved to: {run_dir}")
    return summary


# ===========================================
# CLI
# ===========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet20",
                        help="resnet20, resnet32, convnext, swin")
    parser.add_argument("--data_dir", type=str, default="../../digit_data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./experiment_v3_results")

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )