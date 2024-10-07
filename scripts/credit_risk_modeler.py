import pandas as pd


class CreditScoreEngine:
    """
    A class for organizing the functions that are concerned with generating credit score of user(s) from given transactions of the user(s).
    """

    def __init__(self, transaction_data: pd.DataFrame):
        """
        Initializes the class.

        Args:
            transaction_data(pd.DataFame): the transactional data from which credit scores are going to be determined.
        """

        self.data = transaction_data
    
    @staticmethod
    def calculate_recency(transaction_dates: pd.Series, latest_date: pd.Timestamp=None) -> int:
        """
        A function that calculates the recency of a users transaction.

        Args:
            transaction_dates(pd.Series): a series containing the dates of the user's transactions.
            latest_date(pd.Timestamp): the time from which we want to calculate the recency value. Default is the time where the function is run.

        Returns:
            int: an integer that represents the difference between the latest_date and the most recent date from the transaction date series. 
        """
        
        # If the latest date isn't passed, use today's date
        if latest_date is None:
            latest_date = pd.Timestamp.utcnow()

        # Ensure the series is in datetime format
        transaction_dates = pd.to_datetime(transaction_dates)

        # Obtain the most recent transaction date from the series
        user_latest = transaction_dates.max()

        # Calculate the time delta
        time_delta = latest_date - user_latest

        # Return the number of days in the time delta
        return time_delta.days
    
    def calcualte_rfms(self) -> pd.DataFrame:
        """
        A method that calcualtes the RFMS values of users.

        Returns:
            pd.DataFrame: a dataframe that contains the RFMS values corresponding to each user.
        """

        # convert the date string to a pd.datetime object
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'])
        
        # first group the data for each data
        user_groupings = self.data.groupby(by="CustomerId")

        # now aggregate all the RFMS values
        rfms_data = user_groupings.agg(
            Recency=("TransactionStartTime", CreditScoreEngine.calculate_recency),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum"),
            Std_Deviation=("Value", "std")
        )

        # now fill the NA values in standard deviation with zero
        rfms_data['Std_Deviation'] = rfms_data['Std_Deviation'].fillna(value=0)

        return rfms_data