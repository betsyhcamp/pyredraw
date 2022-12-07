import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_replicates(original_series, replicates_df, field_name, num_replicates=10):
    """Plot lines to compare original time series to replicate time series.
    
    Parameters
    ----------
    original_series : pandas.Series
        Pandas series with original time series. Index should be datetime
    replicates_df : pandas.DataFrame
        Pandas dataframe with field_name to be plotted and field called "replicate". 
        Index should be datetime.
    num_replicates : int, optional
        Plot the first num_replicates. By default, 10.

    Returns
    -------
    Nothing returned

    Examples
    --------
    >>> from pyredraw.plotting import plot_replicates
    >>> plot_replicates(original_series, replicates_df, "monthly_expenditures")
    """
    # to set the plot size
    plt.figure(figsize=(16, 8))
    
    # plotting replicates
    number_replicates_array = np.arange(0, num_replicates)
    for n in number_replicates_array:
        label_str = "rep" + str(n)
        (replicates_df.loc[replicates_df["replicate"]==n, 
                          field_name]
                      .plot(label=label_str))
    
    # plotting original time series    
    original_series.plot(label='original ts', color='black')  
    
    # adding Label to the y-axis
    plt.ylabel(f'{field_name}')
    
    # adding legend to the curve
    plt.legend()
    plt.show()