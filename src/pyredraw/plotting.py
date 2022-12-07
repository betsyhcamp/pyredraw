import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_words(word_counts, n=10):
    """Plot a bar chart of word counts.
    
    Parameters
    ----------
    word_counts : collections.Counter
        Counter object of word counts.
    n : int, optional
        Plot the top n words. By default, 10.

    Returns
    -------
    matplotlib.container.BarContainer
        Bar chart of word counts.

    Examples
    --------
    >>> from pycounts.pycounts import count_words
    >>> from pycounts.plotting import plot_words
    >>> counts = count_words("text.txt")
    >>> plot_words(counts)
    """
    top_n_words = word_counts.most_common(n)
    word, count = zip(*top_n_words)
    fig = plt.bar(range(n), count)
    plt.xticks(range(n), labels=word, rotation=45)
    plt.xlabel("Word")
    plt.ylabel("Count")
    return fig
 
    
def plot_replicates(original_series, replicates_df, field_name, num_replicates=10):
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