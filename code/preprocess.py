import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from helpers import normalize, join, split_on_date
import time
import random
import os

api_key = 'A759HSE0DRDPALHC'
path_to_data = "../data/stocks/"
ts = TimeSeries(key=api_key, output_format='pandas')

def get_covid_data():
    df = get_data("../data/covid.csv")
    return df

def get_data_paths():
    """
    Gets list of paths to stock data points
    
    :return: python list of strings containing path to csv files for all stocks
    """
    file = open("../data/stocks.txt", 'r')
    stock_files = []
    for l in file:
        path = path_to_data + l.strip() + ".csv"
        stock_files.append(path)
    return stock_files
    
def adjust_for_splits(df : pd.DataFrame):
    """
    Adjusts stock price based on the stock splits
    """
    print(df)
    print(df['Split Coefficient'].cumprod())
    
    return df

def clean_data():
    """
    Renames all of the column names in the stock files for clarity and to match
    the index funds. Additionally reverses so the rows go from oldest to newest
    """
    rename_mapping = {"date": "Date",
                      "1. open": "Open",
                      "2. high": "High",
                      "3. low": "Low",
                      "4. close": "Close",
                      "5. adjusted close": "Adjusted Close",
                      "6. volume": "Volume",
                      "7. dividend amount": "Dividend Amount",
                      "8. split coefficient": "Split Coefficient"}
    stocks = get_data_paths()
    
    for filename in stocks:
        df = pd.read_csv(filename)
        df = df.rename(columns=rename_mapping)
        # df = adjust_for_splits(df)
        df = reverse_df(df)
        df.to_csv(filename, index=False)

def install_data(stock_ticker):
    """
    Gets stock data given the stock ticker using alpha vantage. Saves data as csv
    
    :param stock_ticker: String of the stock ticker to download
    :return: None
    """
    file_path = path_to_data + stock_ticker + ".csv"
    data, meta_data = ts.get_daily_adjusted(symbol=stock_ticker, outputsize='full')
    # print(data)
    data.to_csv(file_path)
    
def reverse_df(df):
    """
    Reverses a dataframe
    
    :param df: pandas dataframe to be reversed
    :return: reversed dataframe
    """
    df = df[::-1]
    return df

def get_data(filename):
    """
    Reads in a path to a csv file and returns a pandas dataframe
    
    :param filename: path to the file
    :return: pandas dataframe of the file
    """
    return pd.read_csv(filename)

def get_all_stocks(covid_data=None, random_seed = None):
    """
    Function that gets a list of all the stocks
    
    :return: python list of tuples. Each tuple has stock ticker in first slot, and 
             pandas dataframe of the values in the secon slot
    """
    file = open("../data/stocks.txt", 'r')
    l = []
    for stock in file:
        path = path_to_data + stock.strip() + ".csv"
        data = get_data(path)
        data = join(data, covid_data)
        # data = split_on_date(data, "2020-01-06")[1]

        data = normalize(data)
        l.append((stock, data))
    
    random.seed(random_seed)
    random.shuffle(l)
    train_data = l[:449]
    test_data = l[449:]
    return train_data, test_data

def download_all_data():
    """
    Downloads all of the stocks given in ../data/stocks.txt and saves them as csv files.
    """
    file = open("../data/stocks.txt", 'r')
    for l in file:
        print("Downloaded:", l.strip())
        install_data(l.strip())
        time.sleep(12.1)