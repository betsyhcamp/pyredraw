# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("iceland_montly_retail_debit_card_expenditures.csv", 
                  index_col='date',
                  parse_dates=True
                 )

# %%
df

# %%
sr =df.squeeze()
sr

# %%
sr = sr.asfreq('MS')

# %%
sr.plot()

# %%
data_array = df['monthly_expenditure'].values
data_array

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
i = 1

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
block_start_index=152

# %%
(len(data_array) - block_window_length+1)

# %%
temp_block = data_array[block_start_index:
                        (block_start_index + block_window_length)]

# %%
len(temp_block)

# %%
chosen_blocks[i,]=temp_block

# %%
chosen_blocks

# %%
block_window_length = 14 # input parameter

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
# Check: acceptable input data structures for data_array: list, pandas Series, numpy ndarray (1D)



# %%
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


# %%
number_replicates = 10
seed=1027

# %%
# set random seed for reproducibility
np.random.seed(seed)
# Int conversion number of replicates to create 
number_replicates = int(number_replicates)
# create array of this length to iterate through
number_replicates_array = np.arange(0, number_replicates)

# initialize array to hold moving block boostrap replicates
bootstrap_replicates = np.zeros((number_replicates, len(data_array)))

# %%
bootstrap_replicates.shape

# %%
for replicate_index in number_replicates_array:
    replicate = moving_block_selection(data_array, block_window_length)
    bootstrap_replicates[replicate_index,] = replicate

# %%
bootstrap_replicates


# %%

# %%

# %%
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


# %%
block_window_length = 12
number_replicates = 10
seed=1027
result = moving_block_bootstrap(data_array, block_window_length, number_replicates)

# %%
result

# %%
result.shape

# %%
data_array

# %%
# do box-cox transform
boxcox_array, lambda_val = boxcox(data_array)

# %%
# make series from array resulting from box-cox 
boxcox_series = pd.Series(
    boxcox_array, index=pd.date_range("1-1-2000", periods=len(boxcox_array), freq="MS")
    , name="monthly_expenditures"
)

# %%
np.array(boxcox_series)

# %%
boxcox_series.describe()

# %%
boxcox_series

# %%
# do STL decomp (note seasonal = period+1, seasonal must be odd)
stl = STL(boxcox_series, seasonal=13)
res = stl.fit()
fig = res.plot()

# %%

res.seasonal # seasonal component of STL decomp
#res.trend

# %%
res.trend # trend component of STL decomp

# %%
res.resid # residual component of STL decomp

# %%
# do moving block bootstrap of STL residuals 
residual_replicates = moving_block_bootstrap(res.resid, 
                           12*2, 
                           10, 
                           seed=1027)

# %%
residual_replicates[0,]

# %%
# make residual array into pandas Series
rr0 = pd.Series(residual_replicates[0,], index=pd.date_range("1-1-2000", periods=len(residual_replicates[0,]), freq="MS")
    , name="resid"
)
rr1 = pd.Series(residual_replicates[1,], index=pd.date_range("1-1-2000", periods=len(residual_replicates[1,]), freq="MS")
    , name="resid"
)
rr2 = pd.Series(residual_replicates[2,], index=pd.date_range("1-1-2000", periods=len(residual_replicates[2,]), freq="MS")
    , name="resid"
)
rr3 = pd.Series(residual_replicates[3,], index=pd.date_range("1-1-2000", periods=len(residual_replicates[3,]), freq="MS")
    , name="resid"
)
rr4 = pd.Series(residual_replicates[4,], index=pd.date_range("1-1-2000", periods=len(residual_replicates[4,]), freq="MS")
    , name="resid"
)
rr5 = pd.Series(residual_replicates[5,], index=pd.date_range("1-1-2000", periods=len(residual_replicates[5,]), freq="MS")
    , name="resid"
)
rr6 = pd.Series(residual_replicates[6,], index=pd.date_range("1-1-2000", periods=len(residual_replicates[6,]), freq="MS")
    , name="resid"
)
rr7 = pd.Series(residual_replicates[7,], index=pd.date_range("1-1-2000", periods=len(residual_replicates[7,]), freq="MS")
    , name="resid"
)
rr8 = pd.Series(residual_replicates[8,], index=pd.date_range("1-1-2000", periods=len(residual_replicates[8,]), freq="MS")
    , name="resid"
)
rr9 = pd.Series(residual_replicates[9,], index=pd.date_range("1-1-2000", periods=len(residual_replicates[9,]), freq="MS")
    , name="resid"
)

# %%
rr0

# %%
# get total series (bootstrap residual + seasonal STL component + trend STL component)
t0 =rr0 + res.seasonal +res.trend
t1 =rr1 + res.seasonal +res.trend
t2 =rr2 + res.seasonal +res.trend
t3 =rr3 + res.seasonal +res.trend
t4 =rr4 + res.seasonal +res.trend
t5 =rr5 + res.seasonal +res.trend
t6 =rr6 + res.seasonal +res.trend
t7 =rr7 + res.seasonal +res.trend
t8 =rr8 + res.seasonal +res.trend
t9 =rr9 + res.seasonal +res.trend

# %%
# Do inverse Box-Cox of total series (bootstrap residual + seasonal STL component + trend STL component)
y0 = inv_boxcox(t0, lambda_val)
y1 = inv_boxcox(t1, lambda_val)
y2 = inv_boxcox(t2, lambda_val)
y3 = inv_boxcox(t3, lambda_val)
y4 = inv_boxcox(t4, lambda_val)
y5 = inv_boxcox(t5, lambda_val)
y6 = inv_boxcox(t6, lambda_val)
y7 = inv_boxcox(t7, lambda_val)
y8 = inv_boxcox(t8, lambda_val)
y9 = inv_boxcox(t9, lambda_val)

# %%
# Plot replicates with original time series to see if replicates make sense.

# to set the plot size
plt.figure(figsize=(16, 8), dpi=150)
  
# using plot method to plot open prices.
# in plot method we set the label and color of the curve.
#sr.plot(label='original ts', color='orange')
y0.plot(label='rep0')
y1.plot(label='rep1')
y2.plot(label='rep2')
y3.plot(label='rep3')
y4.plot(label='rep4')
y5.plot(label='rep5')
y6.plot(label='rep6')
y7.plot(label='rep7')
y8.plot(label='rep8')
y9.plot(label='rep9')
sr.plot(label='original ts', color='black')  
# adding title to the plot
#plt.title('Open Price Plot')
  
# adding Label to the x-axis
#plt.xlabel('Years')
  
# adding legend to the curve
plt.legend()


# %%
