from typing import List
import pandas as pd

class FeatureEngineering:
    """
    A class for organizing functions/methods for performing feature engineering on bank transaction data.
    Most of the functions are static functions because it is intended to use this function to build a pipleine 
    and all of the functions perform feature engineering on the passed data and return it without the need to keep the state in the instance.
    """

    @staticmethod
    def extract_date_features(data: pd.DataFrame, date_column : str = 'TransactionStartTime') -> pd.DataFrame:
        """"""

    @staticmethod
    def encode_categorical_data(data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """"""
    
    @staticmethod
    def handle_missing_data(data: pd.DataFrame) -> pd.DataFrame:
        """"""
    
    @staticmethod
    def aggregate_customer_data(data: pd.DataFrame) -> pd.DataFrame:
        """"""

    @staticmethod
    def normalize_numerical_features(data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """"""

