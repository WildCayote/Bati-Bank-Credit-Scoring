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

def plot_bar_chart_with_values(counts, title='Count of Risk Labels', xlabel='Risk Label', ylabel='Count', figsize=(8, 5)):
    """
    Plots a bar chart with values displayed on top of each bar, ensuring 'Good' is green and 'Bad' is red.

    Parameters:
    - counts: Pandas Series containing the counts for each category. Index should have 'Good' and 'Bad'.
    - title: Title of the bar chart.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - figsize: Size of the figure.
    """
    # Define color mapping based on risk label
    color_mapping = {'Good': 'green', 'Bad': 'red'}
    bar_colors = [color_mapping.get(label, 'gray') for label in counts.index]

    # Create the bar chart
    plt.figure(figsize=figsize)
    bars = counts.plot(kind='bar', color=bar_colors, alpha=0.7)

    # Add the value on top of each bar
    for bar in bars.patches:
        yval = bar.get_height()  # Get the height of each bar
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), 
                 ha='center', va='bottom', fontsize=12)  # Add text at the center-top of the bar

    plt.title(title, weight='bold')
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel(ylabel, weight='bold')
    plt.xticks(rotation=0)
    plt.grid(axis='y')

    plt.show()

