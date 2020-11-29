import pandas
from alpha_vantage.timeseries import TimeSeries
import time

api_key = 'A759HSE0DRDPALHC'
path_to_data = "../data/stocks/"

ts = TimeSeries(key=api_key, output_format='pandas')

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
    

def normalize(df : pandas.DataFrame):
    """
    
    """
    df = df.drop(columns="Date")
    df = df / df.max(0)
    return df

def clean_data():
    """
    Renames all of the column names in the stock files for clarity and to match
    the index funds
    """
    rename_mapping = {"date": "Date",
                      "1. open": "Open",
                      "2. high": "High",
                      "3. low": "Low",
                      "4. close": "Close",
                      "5. volume": "Volume"}
    stocks = get_data_paths()
    
    for filename in stocks:
        df = pandas.read_csv(filename)
        df = df.rename(columns=rename_mapping)
        df.to_csv(filename, index=False)
    return

def install_data(stock_ticker):
    file_path = path_to_data + stock_ticker + ".csv"
    data, meta_data = ts.get_daily(symbol=stock_ticker, outputsize='full')
    data.to_csv(file_path)
    

def get_data(filename):
    """
    Reads in a path to a csv file and returns a pandas dataframe
    
    :param filename: path to the file
    :return: pandas dataframe of the file
    """
    return pandas.read_csv(filename)

def get_all_stocks():
    file = open("../data/stocks.txt", 'r')
    l = []
    for stock in file:
        l.append((stock, get_data(path_to_data + stock.strip() + ".csv")))
    return l

def download_all_data():
    """
    Downloads all of the stocks given in ../data/stocks.txt and saves them as csv files.
    """
    file = open("../data/stocks.txt", 'r')
    for l in file:
        install_data(l.strip())
        time.sleep(12.1)