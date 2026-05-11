"""Дообучение базовой модели на Train_2, валидация на Test_2.

Базовая модель скачивается из S3, новая версия выкладывается обратно в S3.
Метрики и параметры дообучения логируются в TensorBoard.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from src import s3_utils
from src.training import (
    build_dataloaders,
    build_model,
    evaluate,
    get_device,
    save_metrics_json,
    set_seed,
    train_one_epoch,
)


def main(params_path: str):
    with open(params_path) as f:
        params = yaml.safe_load(f)

    data_cfg = params["data"]
    cfg = params["finetune"]
    s3_cfg = params["s3"]
    paths = params["paths"]

    set_seed(cfg["seed"])
    device = get_device()
    print(f"device: {device}")

    train_loader, test_loader, n_train, n_test = build_dataloaders(
        data_cfg=data_cfg,
        train_key="train_2",
        test_key="test_2",
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print(f"train size: {n_train}, test size: {n_test}")

    ckpt_dir = Path(paths["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_local = ckpt_dir / "resnet18_base_from_s3.pt"
    s3_utils.download_file(s3_cfg, s3_cfg["base_model_key"], str(base_local))

    model = build_model(pretrained=False)
    state_dict = torch.load(base_local, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("base model loaded from S3")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"]
    )

    log_dir = Path(paths["tensorboard_dir"]) / "finetune"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    hparams = {
        "stage": "finetune",
        "model": "resnet18",
        "pretrained_source": f"s3://{s3_cfg['bucket']}/{s3_cfg['base_model_key']}",
        "image_size": cfg["image_size"],
        "batch_size": cfg["batch_size"],
        "num_epochs": cfg["num_epochs"],
        "lr": cfg["lr"],
        "step_size": cfg["step_size"],
        "gamma": cfg["gamma"],
        "seed": cfg["seed"],
        "train_size": n_train,
        "test_size": n_test,
    }
    writer.add_text("hparams", "\n".join(f"{k}: {v}" for k, v in hparams.items()))

    last_train = {}
    for epoch in range(cfg["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['num_epochs']}")
        print("-" * 50)

        tr_loss, tr_acc, tr_f1, tr_prec, tr_rec = train_one_epoch(
            model, train_loader, criterion, optimizer, device, desc="Finetune"
        )
        scheduler.step()

        writer.add_scalar("train/loss", tr_loss, epoch)
        writer.add_scalar("train/accuracy", tr_acc, epoch)
        writer.add_scalar("train/f1", tr_f1, epoch)
        writer.add_scalar("train/precision", tr_prec, epoch)
        writer.add_scalar("train/recall", tr_rec, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        print(
            f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} | "
            f"F1: {tr_f1:.4f} | P: {tr_prec:.4f} | R: {tr_rec:.4f}"
        )
        last_train = {
            "loss": tr_loss,
            "accuracy": tr_acc,
            "f1": tr_f1,
            "precision": tr_prec,
            "recall": tr_rec,
        }

    print("\nEvaluating on Test_2 ...")
    te_loss, te_acc, te_f1, te_prec, te_rec = evaluate(
        model, test_loader, criterion, device, desc="Test_2"
    )
    print(
        f"Test  Loss: {te_loss:.4f} | Acc: {te_acc:.4f} | "
        f"F1: {te_f1:.4f} | P: {te_prec:.4f} | R: {te_rec:.4f}"
    )

    final_epoch = cfg["num_epochs"]
    writer.add_scalar("test/loss", te_loss, final_epoch)
    writer.add_scalar("test/accuracy", te_acc, final_epoch)
    writer.add_scalar("test/f1", te_f1, final_epoch)
    writer.add_scalar("test/precision", te_prec, final_epoch)
    writer.add_scalar("test/recall", te_rec, final_epoch)
    writer.flush()
    writer.close()

    local_ckpt = ckpt_dir / "resnet18_finetuned.pt"
    torch.save(model.state_dict(), local_ckpt)
    print(f"saved local checkpoint: {local_ckpt}")

    s3_utils.upload_file(s3_cfg, str(local_ckpt), s3_cfg["finetuned_model_key"])

    save_metrics_json(
        Path(paths["metrics_dir"]) / "finetune.json",
        {
            "train": last_train,
            "test": {
                "loss": te_loss,
                "accuracy": te_acc,
                "f1": te_f1,
                "precision": te_prec,
                "recall": te_rec,
            },
            "hparams": hparams,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)
