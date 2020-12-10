import pandas as pd
import tensorflow as tf
import numpy as np


# Lines added since lstm didnt work for GPU originally
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def predict(model, stock_data):
    """
    Predicts stock price for a given stock and model. Will predict
    however many days out the model is trained on.

    :param model: The model that predicts prices
    :stock_data: pandas df of stock and covid data to predict on
    """
    cell_state = None
    ret_predictions = []
    for j in range(0, stock_data.shape[1], model.window_size):
        batch = stock_data[:,j:j+model.window_size, :]
        predictions, cell_state = model(batch, cell_state)
        predictions = predictions.numpy()
        predictions = predictions.flatten().tolist()
        ret_predictions.extend(predictions)
    return ret_predictions


def apply_noise(tensor):
    """
    Applies noise to a tensor by randomly scaling the value by some multiple.
    Uses random normal scaling with mean 1 (scaling by factor of 1 is identity
    function) and stddev=0.1

    :param tensor: tensor to apply noise to
    :return: the tensor with random scaling
    """

    shape = tensor.shape
    random_scale = tf.random.normal(shape, mean=1, stddev=0.08)
    return tensor * random_scale

def test_stock(model, data, days):
    """
    Tests a model on predicting __days__ in advance
    
    :param model: the model to test
    :param data: Stock data as a pandas df
    :param days: how many days out the model should predict
    :return: Average MAPE across the data
    """
    cell_state = None
    total_loss = 0
    num_batches = 0

    inputs = data.iloc[:-days]
    inputs = tf.convert_to_tensor(inputs)
    inputs = tf.expand_dims(inputs, 0)
    labels = data.iloc[days:]
    labels = labels["Adjusted Close"]
    labels = tf.convert_to_tensor(labels)
    labels = tf.expand_dims(labels, 0)
    for j in range(0, inputs.shape[1], model.window_size):
        batch = inputs[:,j:j+model.window_size, :]
        label = labels[:,j:j+model.window_size]
        predictions, cell_state = model(batch, cell_state)
        loss = model.loss(predictions, label)
        total_loss += loss
        num_batches += 1
    return total_loss / num_batches


def test(model, data, days):
    """
    Tests a model on predicting __days__ in advance
    
    :param model: the model to test
    :param data: python list of pandas dataframes with the data to test on
    :param days: how many days out the model should predict
    :return: Average MAPE across the data
    """
    
    total_loss = 0
    num_batches = 0
    for i in range(0, len(data), model.batch_size):
        # create batches of model.batch_size
        batch_data = data[i:i+model.batch_size]
        batch_data = list(map(lambda x: x[1], batch_data))
        inputs = list(map(lambda x: x.iloc[:-days], batch_data))
        inputs = list(map(lambda x: tf.convert_to_tensor(x), inputs))
        labels = list(map(lambda x: x.iloc[days:], batch_data))
        labels = list(map(lambda x: x["Adjusted Close"], labels))

        inputs = tf.convert_to_tensor(inputs)
        labels = tf.convert_to_tensor(labels)
        labels = tf.expand_dims(labels, 2)

        cell_state = None

        for j in range(0, inputs.shape[1], model.window_size):
            batch = inputs[:,j:j+model.window_size, :]
            label = labels[:,j:j+model.window_size]

            predictions, cell_state = model(batch, cell_state)
            loss = model.loss(predictions, label)

            total_loss += loss
            num_batches += 1
    return total_loss / num_batches

def train(model, data, days):
    """
    Trains a model to predict __days__ in advance for one epoch
    
    :param model: the model to train
    :param data: pandas dataframe with the data to train on
    :param days: how many days out the model should predict
    :return: MAPE on the training data
    """
    total_loss = 0
    num_batches = 0

    for i in range(0, len(data), model.batch_size):
        # create batches of model.batch_size
        batch_data = data[i:i+model.batch_size]
        batch_data = list(map(lambda x: x[1], batch_data))
        inputs = list(map(lambda x: x.iloc[:-days], batch_data))
        labels = list(map(lambda x: x.iloc[days:], batch_data))
        labels = list(map(lambda x: x["Adjusted Close"], labels))
        inputs = list(map(lambda x: tf.convert_to_tensor(x, dtype="float32"), inputs))
        inputs = list(map(lambda x: apply_noise(x), inputs))
        inputs = tf.convert_to_tensor(inputs)
        labels = tf.convert_to_tensor(labels)
        labels = tf.expand_dims(labels, 2)
        labels = apply_noise(labels)
        cell_state = None

        for j in range(0, inputs.shape[1], model.window_size):
            batch = inputs[:,j:j+model.window_size, :]
            label = labels[:,j:j+model.window_size]

            with tf.GradientTape() as tape:
                predictions, cell_state = model(batch, cell_state)
                loss = model.loss(predictions, label)
            # print(loss)
            total_loss += loss
            num_batches += 1

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss / num_batches

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

    :param stock_data: the data for a stock
    :param covid_data: Historic COVID Data
    :return: pandas df of the two dataframes after an inner join on the "Date" key
    """
    data = [stock_data, covid_data]
    df = pd.merge(stock_data, covid_data, how='inner', on='Date')
    df = df.fillna(0)
    return df
