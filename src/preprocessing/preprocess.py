import os
from typing import Any, Tuple, Union
import pandas as pd
from mlbox.encoding import NA_encoder, Categorical_encoder
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from config import paths
from logger import get_logger
from schema.data_schema import RegressionSchema

logger = get_logger(task_name="preprocess")


def impute_missing(
    input_data: pd.DataFrame, numeric_strategy: Union[str, float, int] = "median", training: bool = True
) -> Tuple[pd.DataFrame, Any]:
    """
    Imputes the missing numeric and categorical values in the given dataframe.

    Args:
        input_data (pd.DataFrame): The data to be imputed.
        numeric_strategy (str): The strategy to use during encoding numeric values. Possiable values ['mean', 'median', 'most_frequent', float/int value].
        training (bool): Indicates wheather imputation is done during training or testing phase.
    Returns:
        pd.DataFrame: Dataframe after imputation.
    """


    if training:
        imputer = NA_encoder(numerical_strategy=numeric_strategy)
        data = imputer.fit_transform(input_data)
        dump(imputer, paths.IMPUTER_FILE_PATH)

    else:
        imputer = load(paths.IMPUTER_FILE_PATH)
        data = imputer.transform(input_data)

    return data

    
def encode(
    input_data: pd.DataFrame, target:pd.Series = None, training: bool = True
) -> pd.DataFrame:
    """
    Performs one-hot encoding categorical features of a given dataframe.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.
        target (pd.Series): Target feature.
        schema (RegressionSchema): The schema of the given data.
        training (bool): Indicates wheather encoding is done during training or testing phase.

    Returns:
        A dataframe after performing one-hot encoding
    """
    if training:
        encoder = Categorical_encoder(strategy="dummification")
        data = encoder.fit_transform(input_data, target)
        dump(encoder, paths.ENCODER_FILE)
    
    else:
        encoder = load(paths.ENCODER_FILE)
        data = encoder.transform(input_data)
    
    return data


def normalize(
    input_data: pd.DataFrame, schema: RegressionSchema, training: bool = True
) -> pd.DataFrame:
    """
    Performs MinMax normalization on numeric features of a given dataframe.

    Args:
        input_data (pd.DataFrame): The data to be normalized.
        schema (RegressionSchema): The schema of the given data.
        training (bool): Indicates wheather normalization is done during training or testing phase.

    Returns:
        (pd.DataFrame)  Dataframe after MinMax normalization
    """

    input_data = input_data.copy()
    numeric_features = schema.numeric_features
    if not numeric_features:
        return input_data
    numeric_features = [f for f in numeric_features if f in input_data.columns]
    if training:
        scaler = MinMaxScaler()
        scaler.fit(input_data[numeric_features])
        dump(scaler, paths.SCALER_FILE)
    else:
        scaler = load(paths.SCALER_FILE)
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    return input_data