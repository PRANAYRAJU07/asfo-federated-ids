import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from loguru import logger

class Partitioner:
    @staticmethod
    def iid_partition(df: pd.DataFrame, num_clients: int) -> List[pd.DataFrame]:
        logger.info(f"Partitioning data into {num_clients} IID clients")
        shuffled = df.sample(frac=1).reset_index(drop=True)
        return np.array_split(shuffled, num_clients)

    @staticmethod
    def dirichlet_partition(df: pd.DataFrame, label_col: str, num_clients: int, alpha: float) -> List[pd.DataFrame]:
        logger.info(f"Partitioning data using Dirichlet distribution (alpha={alpha}) into {num_clients} clients")
        # Simplified simulation of Dirichlet partitioning
        labels = df[label_col].unique()
        partitions = [pd.DataFrame()] * num_clients
        
        for label in labels:
            label_data = df[df[label_col] == label]
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            splits = np.split(label_data.sample(frac=1), (np.cumsum(proportions)[:-1] * len(label_data)).astype(int))
            for i in range(num_clients):
                partitions[i] = pd.concat([partitions[i], splits[i]])
                
        return partitions

    @staticmethod
    def domain_partition(df: pd.DataFrame, domain_col: str, num_clients: int) -> List[pd.DataFrame]:
        logger.info(f"Partitioning data by domain column {domain_col}")
        domains = df[domain_col].unique()
        partitions = []
        for d in domains:
            partitions.append(df[df[domain_col] == d])
        # If num_clients > len(domains), we might need to split domains further.
        # This is a simplified version.
        return partitions[:num_clients]

    @staticmethod
    def temporal_partition(df: pd.DataFrame, time_col: str, num_clients: int) -> List[pd.DataFrame]:
        logger.info(f"Partitioning data temporally based on {time_col}")
        sorted_df = df.sort_values(by=time_col).reset_index(drop=True)
        return np.array_split(sorted_df, num_clients)
