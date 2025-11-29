#!/usr/bin/env python3
"""
Run test evaluation using a best checkpoint and plot confusion matrix + metrics.
Usage:
    python run_test_best_model.py --checkpoint ./x_ray_checkpoints/resnet18_full_best.pth

This script expects the test images to be at ./chest_xray/test (ImageFolder structure: class subfolders).
If your dataset path differs, pass --test_dir.
"""
import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test_best_model")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_resnet18(num_classes: int, mode: str = "full"):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if mode == "frozen":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif mode == "partial":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True
    elif mode == "full":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("mode must be one of ['frozen','partial','full']")

    return model


def evaluate_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # prob for class 1
            _, preds = torch.max(outputs, 1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC only if both classes present
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }


def plot_and_save_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth (best)')
    parser.add_argument('--model', type=str, default='resnet18_full', help='Model identifier for naming')
    parser.add_argument('--test_dir', type=str, default='./chest_xray/test', help='Path to test folder (ImageFolder)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}\nPlease ensure dataset downloaded and located here or pass --test_dir")
        return

    device = get_device()
    logger.info(f"Using device: {device}")

    # transforms
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = datasets.ImageFolder(str(test_dir), transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_names = test_dataset.classes
    num_classes = len(class_names)
    logger.info(f"Found classes: {class_names}")

    # model
    model = create_resnet18(num_classes=num_classes, mode='full')
    model.to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    # checkpoint may store dict with 'model_state' or directly state_dict
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)

    logger.info(f"Loaded checkpoint: {ckpt_path}")

    results = evaluate_metrics(model, test_loader, device)

    # print metrics
    logger.info("Test results:")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall: {results['recall']:.4f}")
    logger.info(f"F1: {results['f1']:.4f}")
    logger.info(f"ROC AUC: {results['roc_auc']:.4f}")

    out_dir = Path('./eval_results')
    out_dir.mkdir(parents=True, exist_ok=True)

    # save summary csv
    df = pd.DataFrame([{
        'model': args.model,
        'checkpoint': str(ckpt_path),
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'roc_auc': results['roc_auc'],
    }])
    csv_path = out_dir / f"summary_{args.model}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary to {csv_path}")

    # confusion matrix plot
    cm_path = out_dir / f"confusion_matrix_{args.model}.png"
    plot_and_save_confusion_matrix(results['confusion_matrix'], class_names, cm_path)

    # optionally save predictions
    np.save(out_dir / f"y_true_{args.model}.npy", results['y_true'])
    np.save(out_dir / f"y_pred_{args.model}.npy", results['y_pred'])
    np.save(out_dir / f"y_prob_{args.model}.npy", results['y_prob'])
    logger.info(f"Saved predictions to {out_dir}")


if __name__ == '__main__':
    main()
