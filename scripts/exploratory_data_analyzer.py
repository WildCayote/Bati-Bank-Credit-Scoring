import pandas as pd
import seaborn as sns

sns.set_theme()

class EDAAnalyzer:
    """
    A class for organizing functions/methods for performing EDA on bank transaction data.
    """
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDAAnalyzer class

        Args:
            data(pd.DataFrame): the dataframe that contains bank transactional data
        """
        self.data = data
    
    def basic_overview(self):
        """
        A function that creates basic overview of the data like - data type of columns, the shape of the data(i.e the number of rows and columns) 
        """
    
    def basic_summary_statistics(self):
        """
        A function that generates 5 number summary(descriptive statistics) of the dataframe
        """
    
    def numerical_distribution(self):
        """
        A function that will give histogram plots of numerical data with a density curve that shows the distribution of data
        """

    def categorical_distribution(self):
        """
        A function that will give bar plots of categorical data
        """

    def describe_skewness(self):
        """
        A function that will describe the skewness of the different columns
        """

    

