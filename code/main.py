from preprocess import get_data, download_all_data, get_all_stocks, clean_data, get_covid_data
from helpers import split_on_date, normalize, train, test, join
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
    covid_data = get_covid_data()
    model = Historic()
    train_data, test_data = get_all_stocks(covid_data)
    
    for i in range(0, model.num_epochs):
        train(model, train_data, 20)
        print(test(model, test_data, 20))
        # for stock in stocks:
        #     data = stock[1]
        #     print("Ticker:", stock[0])
        #     #print(data[0:5]["Dividend Amount"])
        #     # train_data, test_data = split_on_date(data, "2020-04-01")
        #     train(model, data, 20)
        #     # loss = test(model, test_data, 20)
        #     # print(loss)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-data",
                        help="Download data from /data/stocks.txt",
                        action="store_true")
    arguments = parser.parse_args()
    main(arguments)
