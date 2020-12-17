import pandas as pd
from scipy.io import arff


def arff2df(paths, include_path=False):
    """Convert .arff files to dataframes.
    Args:
        paths (list of str): The path to the .arff files to convert into pandas dataframes. 
        include_path (bool, optional): If set to True the list returned will include the path from where the dataframe was loaded.
                        Defaults to False.
    Returns:
        list of pandas dataframe: A list of dataframes converted from the specified .arff files.
    """
    df_list = []
    for i in range(len(paths)):
        if(include_path == True):
            df_list.append(
                (paths[i], pd.DataFrame(arff.loadarff(paths[i])[0])))
        else:
            df_list.append(pd.DataFrame(arff.loadarff(paths[i])[0]))
    return df_list


def df_to_dfs(dataframes):
    """ Checks if the dataframe/s passed is a list if not convert it to a list of pandas's dataframes. 

    Args:
        dataframes (pandas' dataframe or list of pandas' dataframe): A single instance of pandas' dataframe or a list.

    Returns:
        list of pandas' dataframes: Returns a list of pandas' dataframes if one instance is passed or returns the list passed instead.
    """

    # initialize list to hold pandas' dataframes
    dfs = []
    # if it is not a list add to list
    if isinstance(dataframes, pd.DataFrame):
        dfs = [dataframes]
    else:
        dfs = dataframes

    # return a list of pandas' dataframes
    return dfs
