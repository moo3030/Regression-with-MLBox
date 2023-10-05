import pandas as pd

from preprocessing.preprocess import (
    encode,
    impute_missing,
    normalize,
)
from schema.data_schema import RegressionSchema

def run_pipeline(
    input_data: pd.DataFrame,
    schema: RegressionSchema,
    training: bool = True,
    target: pd.Series = None
) -> pd.DataFrame:
    """
    Apply transformations to the input data (Imputations, encoding and normalization).

    Args:
        input_data (pd.DataFrame): Data to be processed.
        target (pd.Series): Target feature.
        schema (RegressionSchema): RegressionSchema object carrying data about the schema
        training (bool): Should be set to true if the data is for the training process.
    Returns:
        pd.DataFrame: The data after applying the transformations
    """

    data = impute_missing(input_data=input_data, training=training)
    data = normalize(data, schema, training)
    data = encode(data, target, training)
    return data