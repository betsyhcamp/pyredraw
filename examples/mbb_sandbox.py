# %%
import pandas as pd


# %%
import numpy as np


# %%
df = pd.read_csv("iceland_montly_retail_debit_card_expenditures.csv")

# %%
df.head()

# %%
data_array = df['monthly_expenditure'].values
data_array

# %%
series = df['monthly_expenditure']
series

# %%
np.array(series)

# %%
np.array([10, 12, 13])

# %%
# number of data points in a block
block_window_length = 12
block_window_length = int(block_window_length)

# %%
# number of blocks to choose
number_blocks = int(np.floor(len(data_array) / block_window_length) + 2)
number_blocks_array = np.arange(0, number_blocks + 1)

# %%
number_blocks_array

# %%
# initialize array to hold chosen blocks of data
chosen_blocks = np.zeros((number_blocks, block_window_length))

# %%
chosen_blocks

# %%
chosen_blocks[1,]


# %%
i = 1

# %%

# %%
(len(data_array) - block_window_length+1)

# %%
# randomly chosen block start index 
block_start_index = int(
                    np.random.randint(0,
                                     (len(data_array) - block_window_length+1), 
                                     size = 1))
block_start_index

# %%
len(data_array)

# %%
(len(data_array) - block_window_length+1)

# %%
temp_block = data_array[block_start_index:
                        (block_start_index + block_window_length)]

# %%
temp_block

# %%
chosen_blocks[i,]=temp_block

# %%
block_window_length = 12 # input parameter

np.random.seed(1027)

# number of data points in a block
block_window_length = int(block_window_length)

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
    
chosen_blocks = chosen_blocks.flatten(order="C")    

# %%
subset_start_index = int(np.random.randint(0,
                                  block_window_length, 
                                  size = 1))
clipped_blocks =chosen_blocks[subset_start_index: (subset_start_index + len(data_array))]

# %%
len(clipped_blocks)


# %%
def moving_block_bootstrap(data_array, block_window_length, seed=1027):
    """Select sample of blocks from an input time series array via moving block bootstrap"""
    
    # make input data into numpy array (lists and pandas Series acceptable input types)
    data_array = np.array(data_array)
    
    # set random seed for reproducibility
    np.random.seed(seed)
    
    # number of data points in a block
    block_window_length = int(block_window_length)

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