import torch
import time
import mlflow
from loguru import logger
from app.evaluation.metrics import compute_metrics
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self, model, optimizer, criterion, device, callbacks=None, log_dir="logs/"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks or []
        for cb in self.callbacks:
            cb.trainer = self
        self.writer = SummaryWriter(log_dir=log_dir)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        proximal_mu: float = 0.0,
        global_params: list | None = None,
    ):
        logger.info(f"Starting training for {epochs} epochs on {self.device}")

        # Log model parameters count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        mlflow.log_metric("total_params", total_params)
        mlflow.log_metric("trainable_params", trainable_params)

        for cb in self.callbacks:
            cb.on_train_begin()

        for epoch in range(epochs):
            for cb in self.callbacks:
                cb.on_epoch_begin(epoch)

            train_start = time.time()
            train_logs = self._train_epoch(
                train_loader, epoch, proximal_mu, global_params
            )
            train_time = time.time() - train_start

            val_start = time.time()
            val_logs = self._validate_epoch(val_loader, epoch)
            val_time = time.time() - val_start

            logs = {
                **train_logs,
                **val_logs,
                "train_time": train_time,
                "val_time": val_time,
            }

            # Log to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": logs.get("train_loss", 0),
                    "val_loss": logs.get("val_loss", 0),
                    "val_f1": logs.get("val_f1", 0),
                    "train_time": train_time,
                    "val_time": val_time,
                },
                step=epoch,
            )

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", logs.get("train_loss", 0), epoch)
            self.writer.add_scalar("Loss/val", logs.get("val_loss", 0), epoch)
            self.writer.add_scalar("Metrics/val_f1", logs.get("val_f1", 0), epoch)

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)

            stop_training = any(
                getattr(cb, "stop_training", False) for cb in self.callbacks
            )
            if stop_training:
                break

        for cb in self.callbacks:
            cb.on_train_end()

        self.writer.close()

    def _train_epoch(self, loader, epoch, proximal_mu=0.0, global_params=None):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            if output.shape[-1] == 1:
                output = output.squeeze(-1)
                target = target.float()

            loss = self.criterion(output, target)

            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(
                    self.model.parameters(), global_params
                ):
                    proximal_term += ((local_param - global_param) ** 2).sum()
                loss += (proximal_mu / 2) * proximal_term

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return {"train_loss": total_loss / len(loader)}

    def _validate_epoch(self, loader, epoch):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                if output.shape[-1] == 1:
                    output = output.squeeze(-1)
                    target_float = target.float()
                    loss = self.criterion(output, target_float)
                    probs = torch.sigmoid(output)
                    preds = (probs > 0.5).long()
                else:
                    loss = self.criterion(output, target)
                    probs = (
                        torch.softmax(output, dim=-1)[:, 1]
                        if output.shape[-1] > 1
                        else output
                    )
                    preds = torch.argmax(output, dim=-1)

                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        if len(all_targets) == 0 or len(all_preds) == 0:
            return {"val_loss": total_loss / max(1, len(loader)), "val_accuracy": 0.0}

        metrics = compute_metrics(all_targets, all_preds, all_probs)
        metrics = {f"val_{k}": v for k, v in metrics.items()}
        metrics["val_loss"] = total_loss / max(1, len(loader))
        return metrics
