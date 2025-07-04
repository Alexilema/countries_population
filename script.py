# -*- coding: utf-8 -*-

# Libraries
import numpy as np
import os
import pandas as pd
import requests
import traceback
import urllib3
# import warnings

from concurrent.futures import ThreadPoolExecutor
from scipy.stats import normaltest
from tqdm import tqdm

# Disable the InsecureRequestWarning for verify SSL Certificate = False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)

#%%
#
# API start
#

# Functions
def read_data(
        csv_file: str,
        first_as_index: bool = True
    ) -> pd.DataFrame:
    '''
    Reads a csv file onto a df.
    
    Params:
    ------
    csv_file : str
        File key (path + name) to be read.
    first_as_index : bool
        If True, first column of the csv is to be used as index col.
    '''
    
    # Depending on first_as_index, choose the corresponding input value
    index_col = 0
    if not first_as_index:
        index_col = None
    
    df = pd.read_csv(csv_file, index_col=index_col)
    return df


# Normalization function
def normalize_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes* each numeric column of a df.
    
    *Typically:
    - Normalization means min-max reescaling.
    - Standarization means reescaling so that the mean is 0 and the std is 1.
    
    This function performs the second transformation.
    
    Params:
    ------
    df : pd.DataFrame
        df to be transformed
    """
    # Select numeric cols
    num_cols = df.select_dtypes(include='number').columns
    
    # Get the numeric values
    data = df[num_cols].values
    
    # Calculate each column mean, and substract it to each column value so that
    # the new means are cero
    data = data - np.mean(data, axis=0)
    
    # Calculate each column standard deviation, and divide each value by it so
    # that the new stds are one
    data = data / np.std(data, axis=0)
    
    # Turn back to df
    numeric_df = pd.DataFrame(columns=num_cols, data=data)
    
    # Select the non-numeric cols
    other_cols = [col for col in df.columns if col not in num_cols]
    
    # Include the non-numeric data
    df_norm = pd.concat([numeric_df, df[other_cols]], axis=1)  # axis=1 needed
    
    # Keep original columns order
    original_cols = df.columns
    df_norm = df_norm[original_cols]
    
    return df_norm


# Cleansing function
def remove_boring(df: pd.DataFrame,
                  alpha: float = 0.05) -> pd.DataFrame:
    '''
    Removes all "boring" feature (df column).
    A feature is considered boring if its data distribution is not Gaussian and 
    its name is of the form 'boring{i:04d}' for some integer `i`.
    
    Params:
    ------
    df : pd.DataFrame
        df to be transformed
    alpha: float
        Significance threshold for the normality test (p-value).
    '''
    # Column names error correction
    df.columns = df.columns.str.strip().str.lower()
    
    # @@ DOUBT cols ' boring' y 'Boring' must be removed ??
    
    # Select numeric cols
    num_cols = df.select_dtypes(include='number').columns
    
    # 1) Filter by the col name condition
    boring_cols = [
        col
        for col in num_cols
        if col.startswith("boring")  
        and len(col) == len("boring") + 4 # the extra part has exactly 4 digits
        and col[len("boring"):].isdigit() # the extra part is numeric
        ]
    
    # @@ DOUBT: {i:03d} o i:04d ??? in the csv file they all have 4 digits...
    
    # 2) Filter by the non Gaussian condition
    non_gaussian_cols = []
    
    for col in boring_cols:
        
        # @@ DOUBT: how do we want to treat NAs values on the input data?
        
        # Drop NA values and turn into array type
        array = df[col].dropna().values
        
        # Note: normaltest requires at least 8 samples, otherwise it raises an error
        if len(array) < 8:
            
            # Show a warning
            print("WARNING: Can't perform the Gaussian test with less than 8 data samples.")
            
            # Consider the sample as non-Gaussian
            non_gaussian_cols.append(col)
            continue
        
        # Perform the gaussian distribution test
        _, pvalue = normaltest(array)
        
        # Compare the p-value of the test with the significance threshold to 
        # decide if the distribution is considered gaussian
        if pvalue < alpha:  # condition for non-Gaussian
            non_gaussian_cols.append(col)
    
    # Drop the boring and non-Gaussian columns
    df = df.drop(columns=non_gaussian_cols)
    
    return df


# REST API countries (V3.1)
'''
Main page link:
https://restcountries.com/#api-endpoints-v3-all

We will be using version 3.1:
https://restcountries.com/v3.1/

This database consist of a JSON structured database with data from all the
countries around the world. The list of content or "fields" of info available
for each country can be found getting back the whole DB and directly taking a
look at it.

Fortunately, somebody has done that before, so the full list of
available fields can be checked at:
https://gitlab.com/restcountries/restcountries/-/blob/master/FIELDS.md

In this case, we just want the name of the country and its population
'''

# Create a session with retry policy: when requests.get() fails, it will check
# the policy in 'retry_policy', and if allowed, it will retry the query
session = requests.Session()
retry_policy = urllib3.util.retry.Retry(
    total = 1,                                # Retry up to 5 times
    backoff_factor = 1,                       # Wait time between retry attempts: wait 1s, 2s, 4s, ... between retries
    status_forcelist = [500, 502, 503, 504],  # HTTP codes taht will trigger a retry (*)
    allowed_methods = ["GET"]                 # Retry only allowed on requests of type .get()
)
adapter = requests.adapters.HTTPAdapter(max_retries = retry_policy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# (*) HTTPS codes (some examples)
#  400 Bad Request
#  401 Unauthorized
#  403 Forbidden
#  404 Not Found
#  408 Request Timeout
#  429 Too Many Requests
#  500 Internal Server Error
#  502 Bad Gateway
#  503 Service Unavailable
#  504 Gateway Timeout
#  505 Check http version (http, https, ...)

def get_country_population(country_ref: str) -> pd.DataFrame:
    '''
    Makes a query to the REST countries API to get the population of a country.
    
    Params:
    ------
    country_ref : str
        The value indicated here as an input must be either the 'common' name,
        the 'official' name of the 'nativeName' name, according to the API
        documentation.
        
        Example:
            For Spain, valid refs are:
                - Spain
                - España
                - Kingdom of Spain
                - Reino de España
                - spa
                
        Note: the value of the 'country_names' column is the 'common' country
        name, as it the one that matches our external data.csv info.
        
    Returns:
    -------
    Population value (numeric).
    '''
    
    # Reviewing the API documentation, the easiest way to search for a field
    # value for a particular country is the following url structure:
    link = f'https://restcountries.com/v3.1/name/{country_ref}?fields=name,population'
    
    # # If we want to retrive data from all the countries, we could use:
    # link = 'https://restcountries.com/v3.1/all?fields=name,population'
    
    # Use a try-except block to help handle API downtimes
    try:
        # Send a GET request with timeout and retries
        response = session.get(link, timeout=3, verify=False, headers={'User-Agent': 'Mozilla/5.0'})
        
        # Optional: use a header to emulate using an web browser to do the http
        # call, which can sometimes prevent connection issues.
        #   session.get(..., headers={'User-Agent': 'Mozilla/5.0'})
        
        # Raise and exception if the response contains an error code (ex: 404 not found)
        response.raise_for_status()
        
        # Parse JSON response (countries DB is in JSON format)
        data = response.json()[0]
        # The json dict comes inside a list, due to the DB structure, but as we are
        # requesting one country only, we can just select the list first item
        
        # Extract the population
        population = data.get('population')
        
        # # Extract the 'common' country name (if neccesary)
        # country_name = data.get('name').get('common')
        
        return population
        
    # If all retries fail, return an empty df and print a warning
    except requests.exceptions.RequestException as e:
        print(f"\n\nWARNING: Error connecting to the API: {e} \n\n")
        population = None
        return population


def add_population(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calls the function get_country_population to add the column 'population' to
    the input df, so that each row will have its country's population value.
    
    Params:
    ------
    df : pd.DataFrame
        df to be transformed
        
    Returns:
    -------
    DataFrame containing the new colum 'population'.
    '''

    # # Side note: my first idea was to simply use apply with get_country_population
    # df['population'] = df.apply(get_country_population)
    
    # However, I realized this would make an API query for each row, which is
    # suboptimal, very slow and unnecesary, as we can preferably make a query only
    # once per each country in our df, safe the population values in a separate
    # item, then combine it with the original df
    
    if 'country_names' not in df.columns:
        raise ValueError("Input df missing key column 'country_names'.")
    
    # Create the list of countries present in the df
    countries = df['country_names'].unique()

    # Initialize empty dict
    query_dict = {}
    
    # Make the query for each country and safe the returned values in a dict
    
    # We can try to speed up the queries by using local parallel computation,
    # also known as MultiThreading
    
    ## - Classic ejecution
    # for country in tqdm(countries, desc='API population query', total=len(countries)):
    #     query_dict[country] = get_country_population(country)
    
    ## - MultiThreading
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Don't use too many workers:
            # - always less than the number of cores of the computer
            # - try to avoid the "429 Too Many Requests" code error
            
            futures = []
            with tqdm(desc='API population query', total=len(countries)) as pbar:
                for country in countries:
                    future = executor.submit(get_country_population, country)
                    futures.append([country, future])
                
                for country, future in futures:
                    try:
                        result = future.result()
                        if result:  # Don't include the result if its None
                            query_dict.update({country: result})
                        pbar.update(1)
                    
                    except Exception as e:
                        print(f"Error while processing {country}: {e}")  
    
    except Exception:
        print(traceback.format_exc())
        raise
    
    # Build the population column by mapping the dict info onto the df
    df['population'] = df['country_names'].map(query_dict)
    
    # @@ DOUBT: "Make sure your code handle API downtimes gracefully"
    # Does this mean that if the API is down, the code must constantly retry until
    # the connection works? Or does it simply mean that the code must not return an 
    # error, but rather something like a warning message + an empty population column?
    # Or somethig else?
    
    return df

#
# API end
#
#%%
# Here you can write code to consume the API to read the data, 
# normalize the data, and filter the features. 

# Use classic __main__ line to prevent to code from running when importing its
# functions from another script file
if __name__ == '__main__':

    # Specify the data file path
    input_path = r'C:\Users\alejandro.lema\Downloads\Challenge Data Engineer\data.csv'
    
    # @@ pending: propose a way to pass the path without hardcoding (many options available)
    
    # Read the data
    df = read_data(input_path)
    
    # Normalize the data
    df = normalize_mean_std(df)
    
    # Remove boring features
    df = remove_boring(df)
    
    # Get the population values from the REST countries API
    df = add_population(df)
    
    # Finally, safe the processed data
    output_path = os.path.dirname(input_path)
    df.to_csv(os.path.join(output_path, 'data_processed.csv'), index=False)

#%%
    # Check if population is a "boring" feature
    
    # Select population column
    df_pop = df[['country_names', 'population']]
    
    # Normalize it
    df_pop = normalize_mean_std(df_pop)
    
    # Rename it to "boring_9999", otherwise our function will never classify it as
    # boring even if its distribution is non-Gaussian
    df_pop = df_pop.rename(columns={'population': 'boring9999'})
    
    # Apply the function. If the column is removed, it means the population is
    # indeed a "boring" feature
    df_pop = remove_boring(df_pop)
    
    if 'boring9999' in df_pop.columns:
        print('Nope, population is not a "boring" feature.')
    else:
        print('Yes, population is indeed a "boring" feature.')
    
    # # Alternatively, we could just apply the normaltest directly to the column
    # _, pvalue = normaltest(df['population'].values)
    # if pvalue < 0.05:  # condition for non-Gaussian
    #     print('Yes, population is indeed a "boring" feature.')
    # else:
    #     print('Nope, population is not a "boring" feature.')
