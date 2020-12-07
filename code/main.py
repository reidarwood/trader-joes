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
    train_data, test_data = get_all_stocks(covid_data, random_seed=0)
    losses = []
    for i in range(0, model.num_epochs):
        train_loss = train(model, train_data, 20)
        print("EPOCH {} training loss: {}".format(i, train_loss))
        test_loss = test(model, test_data, 20)
        losses.append(test_loss)
        print("EPOCH {} Test loss: {}".format(i, test_loss))
    print("Last 10 Epochs Average MAPE: {}".format(sum(losses[-10:]) / 10))
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-data",
                        help="Download data from /data/stocks.txt",
                        action="store_true")
    arguments = parser.parse_args()
    main(arguments)
