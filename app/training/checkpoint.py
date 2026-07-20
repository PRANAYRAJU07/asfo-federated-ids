import os
import torch
from app.training.callbacks import Callback
from loguru import logger
from pathlib import Path

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
        if self.mode == 'min':
            self.monitor_op = lambda current, best: current < best
        else:
            self.monitor_op = lambda current, best: current > best
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if not hasattr(self, 'trainer'):
            return
            
        model_state = self.trainer.model.state_dict()
        
        last_path = self.filepath.parent / "last.pt"
        torch.save(model_state, last_path)
        
        if current is None:
            return
            
        if self.best_score is None or self.monitor_op(current, self.best_score):
            self.best_score = current
            best_path = self.filepath.parent / "best.pt"
            torch.save(model_state, best_path)
            logger.info(f"ModelCheckpoint: {self.monitor} improved to {current:.4f}. Saved best model to {best_path}")
