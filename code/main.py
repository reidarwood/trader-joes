from preprocess import get_data, download_all_data, get_all_stocks, clean_data
from helpers import split_on_date, normalize, train, test
import pandas
import tensorflow as tf
import numpy as np
from historic import Historic
import argparse

def main(arguments):
    
    if arguments.download_data:
        download_all_data()
        print("Downloaded Data!")
        clean_data()
        print("Cleaned Data")
    
    model = Historic()
    stocks = get_all_stocks()
    for stock in stocks:
        data = stock[1]
        
        train_data, test_data = split_on_date(data, "2019-01-02")
        data = normalize(train_data)
        print("Ticker:", stock[0])
        train(model, data, 20)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-data",
                        help="Download data from /data/stocks.txt",
                        action="store_true")
    arguments = parser.parse_args()
    main(arguments)