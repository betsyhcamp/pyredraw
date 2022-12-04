import numpy as np

def moving_block_selection(data_array, block_window_length):
    """Select sample of blocks from input time series array via moving block bootstrap.
       Result is a single replicate of the input array.

    Parameters
    ----------
    data_array: numpy ndarray (1D)
        Time series data on which user would like perform moving block bootstrap.
    
    block_window_length: int
        Number of data points in a block bootstrap sample. Length of block sample. If 
        input is float, will be converted to integer.

    Returns
    -------
    numpy ndarray (1D)
        Moving block boostrap result from input data_array. Result is same size
        as input data given by data_array and is a single boostrap replicate of
        the input data_array.

    Examples
    --------
    >>> data_array = [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    >>> block_window_length = 3
    >>> moving_block_selection(data_array, block_window_length)
    
    References
    ---------- 
    Bergmeir C., Hyndman R.J., Benitez J.M.: Bagging exponential smoothing methods 
    using STL decomposition and Box-cox Transformation. International Journal of 
    Forecasting, 32(2):303-312, (2016)
    """

    # number of blocks to choose and create array of this length to iterate through
    number_blocks = int(np.floor(len(data_array) / block_window_length) + 2)
    number_blocks_array = np.arange(0, number_blocks)

    # initialize array to hold chosen blocks of data
    chosen_blocks = np.zeros((number_blocks, block_window_length))

    # collect randomly sampled blocks (with replacement)
    for block_number in number_blocks_array:
        # randomly chosen block start index 
        block_start_index = int(
                    np.random.randint(0,
                                     (len(data_array) - block_window_length+1), 
                                     size = 1))
    
        # get block of data points 
        temp_block = data_array[block_start_index:
                        (block_start_index + block_window_length)]
    
        # assign block of data points to set of chosen blocks
        chosen_blocks[block_number,]=temp_block
    
    # Make 2D array of chosen blocks into 1D array
    chosen_blocks = chosen_blocks.flatten(order="C") 
    
    # Clip blocks to ensure that returned array result is correct length
    subset_start_index = int(np.random.randint(0,
                                  block_window_length, 
                                  size = 1))
    clipped_blocks = chosen_blocks[subset_start_index : (subset_start_index + len(data_array))]
    
    return clipped_blocks


def moving_block_bootstrap(data_array, 
                           block_window_length, 
                           number_replicates, 
                           seed=1027):
    """Create replicates of time series array via moving block bootstrap.

    Parameters
    ----------
    data_array: list, pandas Series, numpy ndarray (1D)
        Time series data on which user would like perform moving block bootstrap.
    
    block_window_length: int or float
        Number of data points in a block bootstrap sample. Length of block sample. If 
        input is float, will be converted to integer.
        
    number_replicates: int or float
        Number of bootstrap replicates of input data to create.
    
    seed: int
        Value of random seed for random number generator for reproducibility. 
        Default value of random seed is 1027 which will be used if no value 
        is provided.  

    Returns
    -------
    numpy ndarray (2D)
        Moving block boostrap replicates from input data_array. Resulting array
        has number of columns equal to length of input data and number of rows
        is equal to number of replicates. Each row is a single bootstrap replicate.

    Examples
    --------
    >>> data_array = [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    >>> block_window_length = 3
    >>> number_replicates = 10
    >>> moving_block_bootstrap(data_array, block_window_length, number_replicates)
    """
    
    # set random seed for reproducibility
    np.random.seed(seed)
    
    # make input data into numpy array (lists and pandas Series acceptable input types)
    data_array = np.array(data_array)
    
    # number of data points in a block
    block_window_length = int(block_window_length)
    
    # Int conversion number of replicates to create 
    number_replicates = int(number_replicates)
    
    # create array of this length to iterate through
    number_replicates_array = np.arange(0, number_replicates)

    # initialize array to hold moving block boostrap replicates
    bootstrap_replicates = np.zeros((number_replicates, len(data_array)))
    
    for replicate_num in number_replicates_array:
        # generate moving block boostrap replicate
        replicate = moving_block_selection(data_array, block_window_length)
        
        # write replicate to output array
        bootstrap_replicates[replicate_num,] = replicate
    
    return bootstrap_replicates