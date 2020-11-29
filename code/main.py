from preprocess import get_data, download_all_data, normalize, get_all_stocks, clean_data
import pandas
import tensorflow as tf
import numpy as np
from historic import Historic
import argparse

def train(model, data, days):
    """
    Trains a model to predict __days___ in advance for one epoch
    
    :param model: the model to train
    :param data: pandas dataframe with the data to train on
    :param days: how many days out the model should predict
    :return: None
    """
    train_data = data.iloc[:-days]
    train_labels = data.iloc[days:]
    train_data = tf.convert_to_tensor(train_data)
    # print(train_labels)
    train_labels = tf.convert_to_tensor(train_labels["Close"])
    cell_state = None
    for i in range(0, train_data.shape[0], model.batch_size):
        batch = train_data[i:i + model.batch_size, :]
        label = train_labels[i:i + model.batch_size]
        
        
        with tf.GradientTape() as tape:
            predictions, cell_state = model(tf.expand_dims(batch, 0), cell_state)
            loss = model.loss(predictions, tf.expand_dims(label, 0))
            
        if i//model.batch_size % 10 == 0:
             print("Loss on training set after {} training steps: {}".format(i//model.batch_size, loss))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    pass

def test(model, data, days):
    """
    Tests a model on predicting __days___ in advance
    
    :param model: the model to test
    :param data: pandas dataframe with the data to test on
    :param days: how many days out the model should predict
    :return: tuple of MAE, RMSE
    """
    pass

def main(arguments):
    
    if arguments.download_data:
        download_all_data()
        print("Downloaded Data!")
        return
    clean_data()
    print("Cleaned Data")
    # Example code to train on Dow
    # data = get_data("../data/^DJI.csv")
    # data = normalize(data)
    # return
    # print("Training on the Dow Jones Industrial Average")
    
    # dates = data.pop("Date")
    # train_data = data.iloc[:-1]
    # train_labels = data.iloc[1:]
    
    model = Historic()
    stocks = get_all_stocks()
    for stock in stocks:
        data = stock[1]
        data = normalize(data)
        print("Ticker:", stock[0])
        train(model, data, 20)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-data",
                        help="Download data from /data/stocks.txt",
                        action="store_true")
    arguments = parser.parse_args()
    main(arguments)