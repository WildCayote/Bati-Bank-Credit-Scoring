import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def visualize_numerical_distribution(data: pd.DataFrame) -> None:
    """
    A function that visualizes the distribution of numerical columns given a dataframe.

    Args:
        data(pd.DataFrame): the data whoes numerical columns are going to get visualized
    
    Returns:
        None:
    """

    # determine the numerical columns and data
    numerical_data = data._get_numeric_data()
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
        median = data[column].median()
        mean = data[column].mean()

        # plot the histplot for that column with a density curve overlayed on it
        sns.histplot(data[column], bins=15, kde=True, ax=axes[idx])

        # add title for the subplot
        axes[idx].set_title(f"Distribution plot of {column}", fontsize=10)

        # set the x and y labels
        axes[idx].set_xlabel(column, fontsize=9)
        axes[idx].set_ylabel("Frequency", fontsize=9)

        # add a lines for indicating the mean and median for the distribution
        axes[idx].axvline(mean, color='black', linewidth=1, label=f'Mean = {round(mean , 2)}') # the line to indicate the mean
        axes[idx].axvline(median, color='red', linewidth=1, label=f'Median = {round(median, 2)}') # the line to indivate the median 

        # add legends for the mean and median
        axes[idx].legend()

    # remove unused subplots
    for unused in range(idx + 1, len(axes)):
        plt.delaxes(ax=axes[unused])
        
    # create a tight layout
    plt.tight_layout()

    # show the plot
    plt.show()    
