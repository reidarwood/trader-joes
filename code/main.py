from preprocess import get_data, download_all_data, get_all_stocks, clean_data, get_covid_data, install_data
from helpers import split_on_date, normalize, train, test, join
import pandas
import tensorflow as tf
import numpy as np
from historic import Historic
import argparse

def main(arguments):
    """
    Main function to run the stock market tester
    """
    if arguments.download_data:
        download_all_data()
        print("Downloaded Data!")
        # clean_data()
        # print("Cleaned Data")
        return
    if arguments.download_stock is not None:
        print(arguments.download_stock)
        install_data(arguments.download_stock)
        return
    
    DAYS_OUT = 20
    covid_data = get_covid_data()
    model = Historic()
    train_data, test_data = get_all_stocks(covid_data, random_seed=0)
    losses = []
    for i in range(0, model.num_epochs):
        train_loss = train(model, train_data, DAYS_OUT)
        print("EPOCH {} training loss: {}".format(i, train_loss))

        test_loss = test(model, test_data, DAYS_OUT)
        print("EPOCH {} Test loss: {}".format(i, test_loss))

        losses.append(test_loss)
    print("Last 10 Epochs Average MAPE: {}".format(sum(losses[-10:]) / 10))
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-data",
                        help="Download data from /data/stocks.txt",
                        action="store_true")
    parser.add_argument("--download-stock",
                        help="Download stock")
    arguments = parser.parse_args()
    main(arguments)
