from typing import List, Dict
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PreprocessingStep(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


class MissingValueCleaner(PreprocessingStep):
    def fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning missing values")
        return df.dropna()


class DuplicateRemover(PreprocessingStep):
    def fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing duplicates")
        return df.drop_duplicates()


class OutlierCleaner(PreprocessingStep):
    def fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning outliers (placeholder)")
        return df


class Encoder(PreprocessingStep):
    def __init__(self, categorical_cols: List[str] | None = None):
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.encoders: Dict[str, LabelEncoder] = {}

    def fit(self, df: pd.DataFrame) -> None:
        if not self.categorical_cols:
            self.categorical_cols = df.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
        logger.info(
            f"Fitting categorical encoders for {len(self.categorical_cols)} columns"
        )
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                classes = list(le.classes_)
                if "<UNKNOWN>" not in classes:
                    classes.append("<UNKNOWN>")
                le.classes_ = np.array(classes)
                self.encoders[col] = le

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in self.categorical_cols:
            if col in df_out.columns and col in self.encoders:
                le = self.encoders[col]
                series = df_out[col].astype(str)
                series = series.map(lambda s: s if s in le.classes_ else "<UNKNOWN>")
                df_out[col] = le.transform(series)
        return df_out


class Normalizer(PreprocessingStep):
    def __init__(self, numerical_cols: List[str] | None = None):
        self.numerical_cols = numerical_cols if numerical_cols is not None else []
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> None:
        if not self.numerical_cols:
            self.numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        logger.info(
            f"Fitting normalizer for {len(self.numerical_cols)} numerical columns"
        )
        cols_to_scale = [c for c in self.numerical_cols if c in df.columns]
        if cols_to_scale:
            self.scaler.fit(df[cols_to_scale])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        cols_to_scale = [c for c in self.numerical_cols if c in df_out.columns]
        if cols_to_scale:
            df_out[cols_to_scale] = self.scaler.transform(df_out[cols_to_scale])
            df_out[cols_to_scale] = df_out[cols_to_scale].astype(np.float32)
        return df_out


class Pipeline:
    def __init__(self, steps: List[PreprocessingStep]):
        self.steps = steps

    def fit(self, df: pd.DataFrame) -> None:
        logger.info(f"Fitting pipeline with {len(self.steps)} steps")
        # Sequential fit-transform internally to fit subsequent steps on transformed data
        df_temp = df.copy()
        for step in self.steps:
            step.fit(df_temp)
            df_temp = step.transform(df_temp)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming with pipeline")
        df_out = df.copy()
        for step in self.steps:
            df_out = step.transform(df_out)
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, filepath: str) -> None:
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Pipeline artifacts saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Pipeline":
        import pickle

        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"Pipeline artifacts loaded from {filepath}")
        return pipeline
