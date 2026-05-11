import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from src.dataset import ImageDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(image_size):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_dataloaders(data_cfg, train_key, test_key, image_size, batch_size, num_workers):
    train_tf, eval_tf = build_transforms(image_size)

    train_ds = ImageDataset(
        csv_file=data_cfg[train_key]["csv"],
        root_dir=data_cfg[train_key]["root"],
        transform=train_tf,
        data_subdir=data_cfg[train_key].get("subdir"),
    )
    test_ds = ImageDataset(
        csv_file=data_cfg[test_key]["csv"],
        root_dir=data_cfg[test_key]["root"],
        transform=eval_tf,
        data_subdir=data_cfg[test_key].get("subdir"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader, len(train_ds), len(test_ds)


def build_model(pretrained=True, num_classes=2):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _compute_metrics(loss_sum, n_seen, preds, labels):
    epoch_loss = loss_sum / max(n_seen, 1)
    epoch_acc = accuracy_score(labels, preds)
    epoch_f1 = f1_score(labels, preds, average="binary", zero_division=0)
    epoch_precision = precision_score(labels, preds, average="binary", zero_division=0)
    epoch_recall = recall_score(labels, preds, average="binary", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall


def train_one_epoch(model, loader, criterion, optimizer, device, desc="Training"):
    model.train()
    loss_sum = 0.0
    n_seen = 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bsz = images.size(0)
        loss_sum += loss.item() * bsz
        n_seen += bsz
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return _compute_metrics(loss_sum, n_seen, all_preds, all_labels)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Validation"):
    model.eval()
    loss_sum = 0.0
    n_seen = 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        bsz = images.size(0)
        loss_sum += loss.item() * bsz
        n_seen += bsz
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return _compute_metrics(loss_sum, n_seen, all_preds, all_labels)


def save_metrics_json(path, payload):
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"metrics saved -> {path}")
