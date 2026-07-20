import os
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from loguru import logger
from tabulate import tabulate
import mlflow

from app.core.utils import set_seed
from app.datasets.registry import DatasetRegistry

# Import datasets to register
from app.preprocessing.pipeline import (
    Pipeline,
    MissingValueCleaner,
    DuplicateRemover,
    Encoder,
    Normalizer,
)
from app.datasets.loader import IntrusionDataModule

from app.models.registry import ModelRegistry

# Import models to register

from app.training.trainer import Trainer
from app.training.early_stopping import EarlyStopping
from app.training.checkpoint import ModelCheckpoint


@hydra.main(version_base=None, config_path="../app/configs", config_name="config")
def benchmark_models(cfg: DictConfig):
    set_seed(cfg.core.seed)
    logger.info(
        f"Starting Baseline Models Benchmark. Config:\n{OmegaConf.to_yaml(cfg)}"
    )

    # Load Data
    dataset_cls = DatasetRegistry.get_dataset(cfg.datasets.name, cfg.datasets)
    df = dataset_cls.load()

    # Split Data (Simple 80/20 train/val for baselines)
    train_df = df.sample(frac=0.8, random_state=cfg.core.seed)
    val_df = df.drop(train_df.index)

    # Preprocess
    pipeline = Pipeline(
        [
            MissingValueCleaner(),
            DuplicateRemover(),
            Encoder(categorical_cols=cfg.datasets.get("categorical_cols", [])),
            Encoder(categorical_cols=[cfg.datasets.label_col]),
            Normalizer(numerical_cols=cfg.datasets.get("numerical_cols", [])),
        ]
    )

    # Fit on train, transform on both to prevent leakage
    pipeline.fit(train_df)
    train_df_processed = pipeline.transform(train_df)
    val_df_processed = pipeline.transform(val_df)

    label_col = cfg.datasets.label_col
    data_module = IntrusionDataModule(
        train_df_processed,
        val_df_processed,
        label_col,
        batch_size=cfg.training.batch_size,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    input_dim = len(train_df_processed.columns) - 1
    output_dim = train_df_processed[label_col].nunique()

    models_to_test = ["mlp", "cnn", "lstm", "transformer"]
    results = []

    device = torch.device(
        cfg.training.device
        if torch.cuda.is_available() and cfg.training.device == "cuda"
        else "cpu"
    )

    mlflow.set_experiment(f"ASFO_Baselines_{cfg.datasets.name}")

    for model_name in models_to_test:
        with mlflow.start_run(run_name=model_name):
            logger.info(f"--- Training {model_name.upper()} ---")

            # Use default model configurations for benchmarking
            model_kwargs = {"input_dim": input_dim, "output_dim": output_dim}
            model = ModelRegistry.get_model(model_name, **model_kwargs)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Use CrossEntropyLoss natively for 2 or more classes
            criterion = torch.nn.CrossEntropyLoss()

            checkpoint_dir = os.path.join(cfg.training.checkpoint_dir, model_name)
            callbacks = [
                EarlyStopping(patience=cfg.training.early_stopping_patience),
                ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "model.pt")),
            ]

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                callbacks=callbacks,
                log_dir=os.path.join(cfg.training.log_dir, model_name),
            )

            start_time = time.time()
            trainer.fit(train_loader, val_loader, epochs=cfg.training.epochs)
            train_duration = time.time() - start_time

            # Final Evaluation
            val_metrics = trainer._validate_epoch(val_loader, 0)

            mlflow.log_param("model", model_name)
            mlflow.log_metrics(val_metrics)

            results.append(
                {
                    "Model": model_name.upper(),
                    "Val Loss": f"{val_metrics['val_loss']:.4f}",
                    "F1 Score": f"{val_metrics['val_f1']:.4f}",
                    "Accuracy": f"{val_metrics['val_accuracy']:.4f}",
                    "ROC AUC": f"{val_metrics.get('val_roc_auc', 0):.4f}",
                    "MCC": f"{val_metrics.get('val_mcc', 0):.4f}",
                    "Time (s)": f"{train_duration:.2f}",
                }
            )

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)
    print(tabulate(results, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    benchmark_models()
