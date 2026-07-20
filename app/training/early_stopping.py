from app.training.callbacks import Callback
from loguru import logger

class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', min_delta=0.0, patience=5, mode='min'):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        
        if self.mode == 'min':
            self.monitor_op = lambda current, best: current < best - self.min_delta
        else:
            self.monitor_op = lambda current, best: current > best + self.min_delta

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            logger.warning(f"EarlyStopping monitoring {self.monitor} which is not available.")
            return

        if self.best_score is None:
            self.best_score = current
        elif self.monitor_op(current, self.best_score):
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                logger.info(f"Early stopping triggered at epoch {epoch}")
