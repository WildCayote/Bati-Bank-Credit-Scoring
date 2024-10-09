from typing import List
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action="ignore")

class WOE_Binner:
    def __init__(self, data: pd.DataFrame, target: str):
        """
        Initialize a WOE Binner

        Args:
            data(pd.DataFrame): the data to perform WOE binning on
            target(str): a string identifying the column where the binary target is found in.
        """
        self.data = data
        self.target = target
        self.numerical_columns = self.obtain_numerical_cols()
        self.categorical_columns = self.obtain_categorical_cols()

    def obtain_numerical_cols(self) -> List[str]:
        """
        A function that returns the numerical columns from the classes dataframe, won't include the target column
        """
        numerical_columns = self.data._get_numeric_data().columns
        numerical_columns = [column for column in numerical_columns if column != self.target]

        return numerical_columns

    def obtain_categorical_cols(self) -> List[str]:
        """
        A function that returns the categorical columns from the classes dataframe, won't include the target column
        """
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target in categorical_columns:
            categorical_columns.remove(self.target)

        return categorical_columns

    def bin_numerical_cols(self, columns_to_ignore: List[str] = [], n_bins: int = 5) -> dict:
        """
        A function that will bin numeric data into a given amount of bins/groups
        """
        bins = {}
        for column in self.numerical_columns:
            if column in columns_to_ignore: continue
            col_bins = pd.qcut(x=self.data[column], q=5, duplicates='drop')
            bins[column] = col_bins

        return bins

    def bin_categorical_cols(self, ignore_cols: List[str] = []):
        """
        A function that will bin categorical data into given amount of bins/groups
        """
        # Ensure 'bin' is not part of the data being processed
        if 'bin' in self.data.columns:
            self.data = self.data.drop(columns=['bin'])
        
        bins = {}
        for column in self.categorical_columns:
            # check if the number of unique values is greater than 10
            n_unique = self.data[column].nunique()

            # if it greater than 10 skip it
            if n_unique > 10 or column in ignore_cols: continue
            
            # get the unique values for the column
            col_bins = self.data[column].unique().tolist()

            bins[column] = col_bins
        
        return bins

    def obtain_counts(self, bins: dict, good_label: any, numeric: bool = True) -> dict:
        """
        A function that will count the good and bad values for every cut

        Args:
            bins(dict): a dictionary that contains keys that are column names and values that contain the bins
            good_label(any): a value that indicates what the good value is in the target column
            numeric(bool): a bool that determines whether the bins passed are for numeric or non numeric columns
        
        Returns:
            dict: a dict that contains columns where the keys are column names and the value are dictionaries containing values for good and bad counts
        """
        counts = {}
        if numeric:
            for column, binnig in bins.items():
                self.data['bin'] = binnig

                # group the bins and count occurences of each target value
                grouped_data = self.data.groupby('bin')[self.target].value_counts().unstack(fill_value=0)

                # Store the counts in the dictionary
                counts[column] = grouped_data.to_dict(orient='index')

            # Drop the temporary 'bin' column
            self.data = self.data.drop(columns=['bin'])
        else:
            for column, binnig in bins.items():
                # group the bins and count occurences of each target value
                grouped_data = self.data.groupby(column)[self.target].value_counts().unstack(fill_value=0)

                # Store the counts in the dictionary
                counts[column] = grouped_data.to_dict(orient='index')
        return counts

    def calculate_woe(self, counts: dict) -> dict:
        """
        A function that calculates the WOE for a given counts dictionary

        Args:
            counts(dict): a dict containing names of columns and within another dictionary that contains keys as bins and values as dicts containing counts
        
        Returns:
            dict: a dict that contains key as columns and then values as dicts that they themselves contain floats for the woe value
        """
        woe_dict = {}

        # Calculate total good (1) and bad (0) across all bins
        for column, bins in counts.items():
            total_good = sum(bins[bin].get(1, 0) for bin in bins)  # Total good counts
            total_bad = sum(bins[bin].get(0, 0) for bin in bins)    # Total bad counts

            # Store the WOE for each bin
            woe_dict[column] = {}

            for bin, bin_counts in bins.items():
                good_count = bin_counts.get(1, 0)
                bad_count = bin_counts.get(0, 0)

                # Avoid division by zero by adding small constant
                good_ratio = (good_count + 0.5) / (total_good + 0.5)
                bad_ratio = (bad_count + 0.5) / (total_bad + 0.5)

                # Calculate WOE
                woe = np.log(good_ratio / bad_ratio)
                woe_dict[column][bin] = woe

        return woe_dict
    
    def bad_probability(self, counts:dict) -> dict:
        """
        A function that calculates the bad probability given counts dictionary of bins

        Args:
            counts(dict): a dict containing names of columns and within another dictionary that contains keys as bins and values as dicts containing counts

        Returns:
            dict: a dict that contains key as columns and then values as dicts that they themselves contain floats for the bad probabilities            
        """
        bad_prob_dict = {}

        # Iterate over each column in the counts dictionary
        for column, bins in counts.items():
            bad_prob_dict[column] = {}
    
            # Calculate bad probability for each bin
            for bin, bin_counts in bins.items():
                bad_count = bin_counts.get('Bad')  # Count of bad (0) values in the current bin
                total_count = sum(bin_counts.values())  # Total count of good + bad values in the bin
    
                # Avoid division by zero
                if total_count > 0:
                    bad_prob = bad_count / total_count
                else:
                    bad_prob = 0  # If no values in the bin, bad probability is 0
    
                bad_prob_dict[column][bin] = bad_prob
    
        return bad_prob_dict