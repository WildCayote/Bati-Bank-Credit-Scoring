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
        """
        A function that will breakdown the given date column into hour, day, month and year features.

        Args:
            data(pd.DataFrame): a dataframe containing the time/date column
            date_column(str): the name of the column that contains the date feature, default is TransactionStartTime
        Returns:
            pd.DataFrame: the resulting data frame with the new date features
        """

        # convert the date data to a datetime object
        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

        # break down the data
        data['Hour'] = data['TransactionStartTime'].dt.hour
        data['Day'] = data['TransactionStartTime'].dt.day
        data['Month'] = data['TransactionStartTime'].dt.month
        data['Year'] = data['TransactionStartTime'].dt.year

        return data
    
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

