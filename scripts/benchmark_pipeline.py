import time
import psutil
from loguru import logger
import hydra
from omegaconf import DictConfig
from ydata_profiling import ProfileReport
import pandas as pd

# Mock imports from app
from app.datasets.registry import DatasetRegistry
# Import datasets to register them
import app.datasets.edge_iiot
import app.datasets.unsw_nb15
import app.datasets.cicids2018
from app.preprocessing.pipeline import Pipeline, MissingValueCleaner, DuplicateRemover, Encoder, Normalizer
from app.federated.partitioner import Partitioner

@hydra.main(version_base=None, config_path="../app/configs", config_name="config")
def benchmark(cfg: DictConfig):
    logger.info("Starting benchmark pipeline...")
    
    start_time = time.time()
    
    dataset_name = cfg.datasets.name
    dataset_cls = DatasetRegistry.get_dataset(dataset_name, cfg.datasets)
    
    # 1. Download & Load
    df = dataset_cls.load()
    load_time = time.time() - start_time
    logger.info(f"Loading Time: {load_time:.2f}s")
    
    # 2. Profile before preprocessing
    logger.info("Generating data profile report...")
    profile = ProfileReport(df, title=f"{dataset_name} Profiling Report", minimal=True)
    profile.to_file(f"datasets/reports/{dataset_name}_report.html")
    
    # 3. Preprocess
    prep_start = time.time()
    pipeline = Pipeline([
        MissingValueCleaner(),
        DuplicateRemover(),
        Encoder(categorical_cols=cfg.datasets.get("categorical_cols", [])),
        Normalizer(numerical_cols=cfg.datasets.get("numerical_cols", []))
    ])
    df_processed = pipeline.run(df)
    prep_time = time.time() - prep_start
    logger.info(f"Preprocessing Time: {prep_time:.2f}s")
    
    # 4. Partition
    if cfg.partition.type == "iid":
        partitions = Partitioner.iid_partition(df_processed, cfg.partition.num_clients)
    elif cfg.partition.type == "dirichlet":
        partitions = Partitioner.dirichlet_partition(df_processed, cfg.datasets.label_col, cfg.partition.num_clients, cfg.partition.alpha)
    else:
        partitions = Partitioner.iid_partition(df_processed, cfg.partition.num_clients)
        
    logger.info(f"Tensor Shape (total): {df_processed.shape}")
    logger.info(f"Client Distribution: {[len(p) for p in partitions]}")
    
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    logger.info(f"Memory Usage: {mem:.2f} MB")
    logger.info("Benchmark complete.")

if __name__ == "__main__":
    benchmark()
