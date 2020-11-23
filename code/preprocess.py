import pandas
from alpha_vantage.timeseries import TimeSeries

api_key = 'A759HSE0DRDPALHC'
path_to_data = "../data/"
ts = TimeSeries(key=api_key, output_format='pandas')
def install_data(stock_ticker):
    file_path = path_to_data + stock_ticker + ".csv"
    data, meta_data = ts.get_daily(symbol=stock_ticker, outputsize='full')
    print(meta_data)
    data.to_csv(file_path)
    

def get_data(filename):
    """
    Reads in a path to a csv file and returns a pandas dataframe
    
    :param filename: path to the file
    :return: pandas dataframe of the file
    """
    return pandas.read_csv(filename)

install_data("AAPL")