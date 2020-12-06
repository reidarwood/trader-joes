import pandas as pd
import tensorflow as tf

def test(model, data, days):
    """
    Tests a model on predicting __days___ in advance
    
    :param model: the model to test
    :param data: pandas dataframe with the data to test on
    :param days: how many days out the model should predict
    :return: tuple of MAPE, RMSE
    """
    test_data = data.iloc[:-days]
    test_labels = data.iloc[days:]
    test_data = tf.convert_to_tensor(test_data)
    test_labels = tf.convert_to_tensor(test_labels["Adjusted Close"])
    cell_state = None
    
    losses = 0
    accuracies = 0
    total = 0
    for i in range(0, test_data.shape[0], model.batch_size):
        batch = test_data[i:i + model.batch_size, :]
        label = test_labels[i:i + model.batch_size]
        
        predictions, cell_state = model(tf.expand_dims(batch, 0), cell_state)
        loss = model.loss(predictions, tf.expand_dims(label, 0))
        total += 1
        losses += loss

        if i//model.batch_size % 10 == 0:
             print("Loss on testing set after {} steps: {}".format(i//model.batch_size, loss))
    
    losses = losses / total
    return losses # MAPE and RMSE?

def train(model, data, days):
    """
    Trains a model to predict __days__ in advance for one epoch
    
    :param model: the model to train
    :param data: pandas dataframe with the data to train on
    :param days: how many days out the model should predict
    :return: None
    """
    train_data = data.iloc[:-days]
    train_labels = data.iloc[days:]
    train_data = tf.convert_to_tensor(train_data)
    train_labels = tf.convert_to_tensor(train_labels["Adjusted Close"])
    cell_state = None
    
    for i in range(0, train_data.shape[0], model.batch_size):
        batch = train_data[i:i + model.batch_size, :]
        label = train_labels[i:i + model.batch_size]
        
        
        with tf.GradientTape() as tape:
            predictions, cell_state = model(tf.expand_dims(batch, 0), cell_state)
            loss = model.loss(predictions, tf.expand_dims(label, 0))
        
        if i//model.batch_size % 1 == 0:
             print("Loss on training set after {} training steps: {}".format(i//model.batch_size, loss))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def normalize(df : pd.DataFrame):
    """
    normalizes data in a pandas df to values between 0 and 1
    
    NOTE: Also drops the date column as that cant be normalized
    
    :param df: dataframe to normalize
    :return: normalized dataframe
    """
    
    # Can add back the date column if needed
    df = df.drop(columns="Date")
    df = df / df.max(0)
    df = df.fillna(0)
    return df

def split_on_date(df : pd.DataFrame, date):
    """
    Splits a dataframe on the given date, and returns 2 dataframes of one
    before the given date, and one after
    
    :param df: dataframe of stock data
    :param date: string of date to split on
    :return: pandas df of data from before date
    :return: pandas df of data from after date
    """
    index = df.index[df["Date"]==date].to_list()[0]
    train = df.iloc[:index]
    test = df.iloc[index:]
    return train, test

def join(stock_data, covid_data):
    """
    Joins covid data with stock data on the Date key
    """
    data = [stock_data, covid_data]
    df = pd.merge(stock_data, covid_data, how='inner', on='Date')
    df = df.fillna(0)
    return df
