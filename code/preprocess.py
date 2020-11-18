import pandas

def get_data(filename):
    """
    Reads in a path to a csv file and returns a pandas dataframe
    
    :param filename: path to the file
    :return: pandas dataframe of the file
    """
    return pandas.read_csv(filename)