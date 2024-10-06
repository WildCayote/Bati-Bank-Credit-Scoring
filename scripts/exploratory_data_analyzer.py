import math, warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore")

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
        # print out the shape
        print(f"The data has a shape of: {self.data.shape}")

        # print out the column info
        print(self.data.info())

    def summary_statistics(self):
        """
        A function that generates 5 number summary(descriptive statistics) of the dataframe
        """
        print(self.data.describe())
    
    def missing_values(self):
        """
        A function that checks for columns with missing value and then returns ones with greater than 0 with the percentage of missing values.
        """

        # obtain missing value percentage
        missing = self.data.isna().mean() * 100
        missing = missing [missing > 0]
        
        # print out the result
        print(f"These are columns with missing values greater than 0%:\n{missing}")

    def numerical_distribution(self):
        """
        A function that will give histogram plots of numerical data with a density curve that shows the distribution of data
        """
        
        # determine the numerical columns and data
        numerical_data = self.data._get_numeric_data()
        numerical_cols = numerical_data.columns

        # detrmine number of rows and columns for 
        num_cols = math.ceil(len(numerical_cols) ** 0.5)

        # calculate the number of rows
        num_rows = math.ceil(len(numerical_cols) / num_cols)

        # create subpltos
        fig, axes = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=(14, 9))

        # flatten the axes
        axes = axes.flatten()

        for idx, column in enumerate(numerical_cols):
            # calculate the median and mean to use in the plots
            median = self.data[column].median()
            mean = self.data[column].mean()

            # plot the histplot for that column with a density curve overlayed on it
            sns.histplot(self.data[column], bins=15, kde=True, ax=axes[idx])

            # add title for the subplot
            axes[idx].set_title(f"Distribution plot of {column}", fontsize=10)

            # set the x and y labels
            axes[idx].set_xlabel(column, fontsize=9)
            axes[idx].set_ylabel("Frequency", fontsize=9)

            # add a lines for indicating the mean and median for the distribution
            axes[idx].axvline(mean, color='black', linewidth=1, label='Mean') # the line to indicate the mean
            axes[idx].axvline(median, color='red', linewidth=1, label='Median') # the line to indivate the median 

            # add legends for the mean and median
            axes[idx].legend()

        # remove unused subplots
        for unused in range(idx + 1, len(axes)):
            plt.delaxes(ax=axes[unused])
        
        # create a tight layout
        plt.tight_layout()

        # show the plot
        plt.show()

    def describe_skewness(self):
        """
        A function that will describe the skewness of the numerical columns
        """
        # obtain the skewness of the data
        skewness_data = self.data.skew(numeric_only=True).sort_values().round(decimals=3)
        columns = skewness_data.keys()

        ax = sns.barplot(skewness_data, palette='husl')
        ax.set_title("Plot of Skewness values of Numerical Columns", pad=20)
        ax.set_xlabel("Numerical Columns", weight='bold')
        ax.set_ylabel("Skewness", weight="bold")
        ax.tick_params(axis='x', labelrotation=45)


        for idx, patch in enumerate(ax.patches):
            # get the corrdinates to write the values
            x_coordinate = patch.get_x() + patch.get_width() / 2
            y_coordinate = patch.get_height()

            # get the value of the coordinate
            value = skewness_data[columns[idx]]
            ax.text(x=x_coordinate, y=y_coordinate, s=value, ha='center', va='bottom', weight='bold')

    def categorical_distribution(self):
        """
        A function that will give bar plots of categorical data
        """
        # define the categorical columns that are worth investigating, 
        # i.e avoid columns which hold ids because the information they provide when looking at their distribution isn't useful
        columns_of_interest = ["CurrencyCode", "ProviderId", "ProductCategory", "ChannelId"]

        # create subplots for each categorical variable
        num_columns = math.ceil(len(columns_of_interest) ** 0.5)
        num_rows = math.ceil(len(columns_of_interest) / num_columns) 

        fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=(14,9))

        axes = axes.flatten()

        for idx, column in enumerate(columns_of_interest):
            # group the data around that column and count instances with respective values
            column_grouping = self.data.groupby(by=column)
            grouping_counts = column_grouping.size().sort_values()

            # create the bar plot
            sns_plot = sns.barplot(data=grouping_counts, ax=axes[idx], palette='husl')
            sns_plot.tick_params(axis='x', labelrotation=45)
            sns_plot.set_ylabel(ylabel="Count", weight='bold')
            sns_plot.set_xlabel(xlabel=column, weight='bold', loc='center', labelpad=5)

            category_values = grouping_counts.keys()
            for idx, patch in enumerate(sns_plot.patches):
                # get the corrdinates to write the values 
                x_coordinate = patch.get_x() + patch.get_width() / 2
                y_coordinate = patch.get_height()

                # get the value to be written
                value = grouping_counts[category_values[idx]]
                sns_plot.text(x=x_coordinate, y=y_coordinate, s=value, ha='center', va='bottom', weight='bold')


        # clean up the unused plots
        for unused in range(idx+1, len(columns_of_interest)):
            plt.delaxes(ax=axes[unused])

        fig.suptitle(t="Distribution of Categorical Columns", weight='bold', fontsize=18)
        plt.tight_layout(pad=1)
    
    