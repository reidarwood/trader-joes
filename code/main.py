from preprocess import get_data, download_all_data, get_all_stocks, clean_data, get_covid_data, install_data
from helpers import split_on_date, normalize, train, test, join, test_stock, predict
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from historic import Historic
import argparse

def visualize_predictions(predictions, labels):
    x = [i for i in range(len(predictions))]
    plt.plot(x, predictions, label='Predicted')
    plt.plot(x, labels, label='Actual')
    plt.legend()
    plt.title('S&P 500 Price Prediction')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.show()  

def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAPE)')
    plt.show()  

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
    
    DAYS_OUT = 5
    covid_data = get_covid_data()
    model = Historic()
    train_data, test_data = get_all_stocks(covid_data)
    losses = []
    for i in range(0, model.num_epochs):
        train_loss = train(model, train_data, DAYS_OUT)
        print("EPOCH {} training loss: {}".format(i, train_loss))

        test_loss = test(model, test_data, DAYS_OUT)
        print("EPOCH {} Test loss: {}".format(i, test_loss))

        losses.append(test_loss)
    visualize_loss(losses)
    print("Last 10 Epochs Average MAPE: {}".format(sum(losses[-10:]) / 10))

    sp_data = pd.read_csv("../data/^GSPC.csv")
    sp_data = join(sp_data, covid_data)
    sp_data = normalize(sp_data)
    base_data = sp_data.iloc[:-DAYS_OUT]
    base_data = tf.convert_to_tensor(base_data)
    base_data = tf.expand_dims(base_data, 0)
    labels = sp_data.iloc[DAYS_OUT:]
    labels = labels["Adjusted Close"]
    labels = labels.values.tolist()
    # print(labels)
    predictions = predict(model, base_data)
    # print(len(predictions))
    visualize_predictions(predictions, labels)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-data",
                        help="Download data from /data/stocks.txt",
                        action="store_true")
    parser.add_argument("--download-stock",
                        help="Download stock")
    arguments = parser.parse_args()
    main(arguments)
