import os
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl
import torch
from loguru import logger
import mlflow
import time
import copy

from app.core.utils import set_seed
from app.datasets.registry import DatasetRegistry
import app.datasets.edge_iiot
import app.datasets.unsw_nb15
import app.datasets.cicids2018
from app.preprocessing.pipeline import Pipeline, MissingValueCleaner, DuplicateRemover, Encoder, Normalizer
from app.datasets.loader import IntrusionDataModule
from app.federated.partitioner import Partitioner

def print_data_validation_report(df: pd.DataFrame, dataset_name: str, label_col: str):
    logger.info(f"\n--- DATA VALIDATION REPORT: {dataset_name.upper()} ---")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Columns: {len(df.columns)}")
    if label_col in df.columns:
        attack_classes = df[label_col].nunique()
        counts = df[label_col].value_counts(normalize=True) * 100
        # Assume 'Normal' or 0 is benign. We'll just print the top distributions
        logger.info(f"Attack Classes: {attack_classes}")
        for cls, pct in counts.items():
            logger.info(f"  - {cls}: {pct:.2f}%")
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    logger.info(f"Missing Data: {missing_pct:.2f}%")
    duplicates = df.duplicated().sum()
    logger.info(f"Duplicates: {duplicates} ({(duplicates/len(df))*100:.2f}%)")
    logger.info("-" * 40 + "\n")

from app.models.registry import ModelRegistry
import app.models.mlp
import app.models.cnn
import app.models.lstm
import app.models.transformer

from app.federated.client import IDSClient
from app.federated.server import get_strategy

def client_fn_factory(cfg, partitions_train, partitions_val, input_dim, output_dim, device, model_name):
    def client_fn(cid: str):
        # Ray workers need these imports locally to register models
        import app.models.mlp
        import app.models.cnn
        import app.models.lstm
        import app.models.transformer
        
        cid_int = int(cid)
        train_df = partitions_train[cid_int]
        val_df = partitions_val[cid_int]
        
        data_module = IntrusionDataModule(train_df, val_df, cfg.datasets.label_col, batch_size=cfg.training.batch_size)
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        model = ModelRegistry.get_model(model_name, input_dim=input_dim, output_dim=output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        metadata = {
            "partition_type": cfg.partition.type,
            "dropout_probability": cfg.federated.get("dropout_probability", 0.0)
        }
        
        return IDSClient(
            cid=cid,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=cfg.training.epochs,
            metadata=metadata
        ).to_client()
    return client_fn

@hydra.main(version_base=None, config_path="../app/configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.core.seed)
    logger.info(f"Starting Federated Simulation. Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # 1. Load Dataset
    dataset = DatasetRegistry.get_dataset(cfg.datasets.name, cfg.datasets)
    df = dataset.load()
    
    print_data_validation_report(df, cfg.datasets.name, cfg.datasets.label_col)

    # 2. Preprocess Data
    pipeline = Pipeline([
        MissingValueCleaner(),
        DuplicateRemover(),
        Encoder(cfg.datasets.categorical_cols + [cfg.datasets.label_col]),
        Normalizer(cfg.datasets.numerical_cols)
    ])
    from sklearn.model_selection import train_test_split
    # Stratified split to maintain attack proportions in train/val sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=cfg.core.seed, stratify=df[cfg.datasets.label_col] if cfg.datasets.label_col in df.columns else None)
    
    train_df_processed = pipeline.fit_transform(train_df)
    val_df_processed = pipeline.transform(val_df)
    
    import os
    os.makedirs("artifacts", exist_ok=True)
    pipeline.save("artifacts/preprocessing_pipeline.pkl")
    
    import json
    metadata_file = dataset.metadata_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            # Create a simple experiment manifest
            manifest = {
                "dataset": cfg.datasets.name,
                "strategy": cfg.strategy.name,
                "num_clients": cfg.federated.num_clients,
                "dataset_metadata": metadata,
            }
            with open("artifacts/experiment_manifest.json", "w") as mf:
                json.dump(manifest, mf, indent=2)

    # 3. Partition Data
    num_clients = cfg.federated.num_clients
    if cfg.partition.type == "iid":
        partitions_train = Partitioner.iid_partition(train_df_processed, num_clients)
        partitions_val = Partitioner.iid_partition(val_df_processed, num_clients)
    else:
        partitions_train = Partitioner.dirichlet_partition(train_df_processed, cfg.datasets.label_col, num_clients, alpha=0.5)
        partitions_val = Partitioner.dirichlet_partition(val_df_processed, cfg.datasets.label_col, num_clients, alpha=0.5)
        
    logger.info("\n--- FEDERATED PARTITION VALIDATION ---")
    dist_data = []
    for cid in range(num_clients):
        client_df = partitions_train[cid]
        logger.info(f"Client {cid}: {len(client_df)} samples")
        if cfg.datasets.label_col in client_df.columns:
            dist = client_df[cfg.datasets.label_col].value_counts(normalize=True).to_dict()
            dist_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in dist.items()])
            logger.info(f"  Distribution: {dist_str}")
            dist_data.append({"Client": f"Client {cid}", **dist})
    logger.info("-" * 40 + "\n")
    
    # Save visualization of partition statistics
    if dist_data:
        import matplotlib.pyplot as plt
        import pandas as pd
        dist_df = pd.DataFrame(dist_data).set_index("Client").fillna(0)
        dist_df.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
        plt.title(f"Class Distribution per Client ({cfg.partition.type})")
        plt.ylabel("Proportion")
        plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("artifacts/partition_stats.png")
        plt.close()
    
    input_dim = len(train_df_processed.columns) - 1
    output_dim = max(2, train_df_processed[cfg.datasets.label_col].nunique())
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu")
    
    mlflow.set_experiment(f"ASFO_Federated_{cfg.datasets.name}")
    
    # 4. Benchmarking strategies (FedAvg, FedProx, FedNova)
    strategies_to_test = ["fedavg", "fedprox", "fednova"]
    model_to_test = "mlp" # Use MLP for quick benchmark
    
    for strategy_name in strategies_to_test:
        logger.info(f"--- Running {strategy_name.upper()} ---")
        
        # Override config strategy dynamically
        current_cfg = copy.deepcopy(cfg)
        OmegaConf.set_struct(current_cfg, False)
        current_cfg.strategy.name = strategy_name
        if strategy_name == "fedprox":
            current_cfg.strategy.proximal_mu = 0.1
            
        with mlflow.start_run(run_name=f"{strategy_name}_{model_to_test}"):
            mlflow.log_params(OmegaConf.to_container(current_cfg, resolve=True))
            
            global_model = ModelRegistry.get_model(model_to_test, input_dim=input_dim, output_dim=output_dim)
            checkpoint_dir = os.path.join(current_cfg.training.checkpoint_dir, "federated", strategy_name, model_to_test)
            
            strategy = get_strategy(current_cfg, global_model, checkpoint_dir)
            
            client_fn = client_fn_factory(current_cfg, partitions_train, partitions_val, input_dim, output_dim, device, model_to_test)
            
            # Start simulation (3 rounds for quick testing)
            fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=num_clients,
                config=fl.server.ServerConfig(num_rounds=3),
                strategy=strategy,
                client_resources={"num_cpus": 1, "num_gpus": 0.0},
            )

if __name__ == "__main__":
    main()
