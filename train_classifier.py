from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.utils.seed import set_seed
from src.utils.logger import build_logger
from src.utils.ckpt import save_checkpoint, Checkpoint
from src.data.build import DataConfig, build_dataloaders
from src.models.build import build_model
from src.engine.train_eval import TrainConfig, train_one_epoch, evaluate

def parse_args():
    p = argparse.ArgumentParser("Train classifier baseline")
    p.add_argument("--dataset", type=str, required=True, choices=["cifar10","cifar100","gtsrb","mnist"])
    p.add_argument("--model", type=str, required=True, choices=["lenet","resnet18","resnet20","vgg16"])
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.out_dir) / f"{args.dataset}_{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("train", run_dir)

    dcfg = DataConfig(dataset=args.dataset, data_root=args.data_root, batch_size=args.batch_size)
    train_loader, test_loader, num_classes, in_ch = build_dataloaders(dcfg)

    model = build_model(args.model, num_classes=num_classes, in_ch=in_ch).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best = -1.0
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optim, args.device)
        metrics = evaluate(model, test_loader, args.device)
        logger.info(f"Epoch {epoch}/{args.epochs} loss={loss:.4f} acc={metrics['acc']:.4f}")

        if metrics["acc"] > best:
            best = metrics["acc"]
            save_checkpoint(best_path, Checkpoint(
                model_state=model.state_dict(),
                optim_state=optim.state_dict(),
                epoch=epoch,
                best_metric=best,
                cfg={
                    "dataset": args.dataset,
                    "model": args.model,
                    "num_classes": num_classes,
                    "in_ch": in_ch,
                    "seed": args.seed,
                }
            ))
            logger.info(f"Saved best to {best_path} (acc={best:.4f})")

    logger.info(f"Done. Best acc={best:.4f}")

if __name__ == "__main__":
    main()
