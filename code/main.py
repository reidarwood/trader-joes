from preprocess import get_data
import pandas
import tensorflow as tf
import numpy as np
from historic import Historic

def train(model, data, days):
    """
    Trains a model to predict __days___ in advance for one epoch
    
    :param model: the model to train
    :param data: pandas dataframe with the data to train on
    :param days: how many days out the model should predict
    :return: None
    """
    # Get labels and inputs
    # Batch and train
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

def main():
    # Example code to train on Dow
    data = get_data("../data/^DJI.csv")
    print("Training on the Dow Jones Industrial Average")
    data.head()
    # split up the data to training and testing data
    # train_data = data
    # test_data = data
    # 
    # model = Historic()
    # train(model, train_data, 10)
    # print("Accuracy: {}".format(test(model, test_data, 10)))


if __name__ == '__main__':
    main()