import pandas as pd
from scipy.io import arff
import numpy as np
import missingno as msno
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from PIL import Image
import os


import convertutils as util

def missing_stats(dataframes):
    """ Output statistics on missing values. The following stats are shown;
    total instances, total instances with missing values, total instances without missing values 
    and the data loss percentage if the values with missing values were to be removed.  

    Args:
        dataframes (pandas' dataframe or a list of pandas' dataframes): The instances or a list of different instance to explore.

    Returns:
        pandas' dataframe: A pandas dataframe with missing values stats for each dataframe passed. 
    """

    # convert to pandas' list if required
    df_missing = util.df_to_dfs(dataframes)
    # initialize array to hold stats
    missing_stats = np.zeros((len(df_missing), 4))
    # loop and calculate statistics for each dataframe passed
    for i in range(len(df_missing)):
        instances_no_missing = df_missing[i].dropna().shape[0]
        missing_stats[i][0] = df_missing[i].shape[0]
        missing_stats[i][1] = df_missing[i].shape[0] - instances_no_missing
        missing_stats[i][2] = instances_no_missing
        missing_stats[i][3] = round(
            (missing_stats[i][1]/missing_stats[i][0]), 4)

    # create new pandas' dataframe which holds these stats
    columns = ["total_instances", "total_instances_with_missing_values",
               "total_instances_without_missing_values", "data_loss"]
    df_missing_stats = pd.DataFrame(data=missing_stats, columns=columns)

    # return missing values stats as a pandas' dataframe
    return df_missing_stats


def nullity_matrix(dataframes, figsize=(20, 5), include_all=False):
    """ Plots the nullity matrix of the missinggo library for the datasets.

    Args:
        dataframes (pandas' dataframe or a list of pandas' dataframes): The instances or a list of different instance to plot.
        figsize (tuple(int,int)): The size of the plot
        include_all (bool): if true show all features if false shows only features with missing values.
    """

    # convert to pandas' list if required
    dfs = util.df_to_dfs(dataframes)
    # loop and plot the nullity matrix for each dataframe passed
    for i in range(len(dataframes)):
        tmp_df = dfs[i] if include_all == True else dfs[i][dfs[i].columns[dfs[i].isna(
        ).any()].tolist()]
        map = msno.matrix(tmp_df, labels=True, figsize=figsize, inline=False)
        # map.figure.savefig("nullity_matrix.jpeg")
        map.figure.savefig(os.path.join(figPath(), 'nullity_matrix.jpeg'))


def nullity_heatmap(dataframes, figsize=(20, 20), include_all=False):
    """ Plots the nullity heatmap of the missinggo library for the datasets.

    Args:
        dataframes (pandas' dataframe or a list of pandas' dataframes): The instances or a list of different instance to plot.
        figsize (tuple(int,int)): The size of the plot
        include_all (bool): if true show all features if false shows only features with missing values.
    """

    # convert to pandas' list if required
    dfs = util.df_to_dfs(dataframes)
    # loop and plot the nullity heatmap for each dataframe passed
    for i in range(len(dataframes)):
        tmp_df = dfs[i] if include_all == True else dfs[i][dfs[i].columns[dfs[i].isna(
        ).any()].tolist()]
        map = msno.heatmap(tmp_df, labels=True, figsize=figsize, inline=False)
        # map.figure.savefig("nullity_heatmap.jpeg")
        map.figure.savefig(os.path.join(figPath(), 'nullity_heatmap.jpeg'))


def __sklearn_imputation(dataframes, strategy):
    dfs = util.df_to_dfs(dataframes)
    imp_sklearn_dfs = []
    if (strategy == "constant"):
        sklearn_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy, fill_value=0)
    else:
        sklearn_imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    for i in range(len(dfs)):
        imp_sklearn_dfs.append(
            pd.DataFrame(
                sklearn_imputer.fit_transform(dfs[i]), columns=dfs[i].columns
            ).astype(dfs[i].dtypes.to_dict()))

    return imp_sklearn_dfs


def mean_imputation(dataframes):
    """Imputes missing values found in pandas dataframe/s using sklearn mean imputation.

    Args:
        dataframes (pandas dataframe or list of dataframes): The dataframe/s to impute missing values for.

    Returns:
        list of pandas dataframe: A list of pandas dataframe imputted using mean imputation.
    """
    return __sklearn_imputation(dataframes, "mean")


def median_imputation(dataframes):
    return __sklearn_imputation(dataframes, "median")


def most_frequent_imputation(dataframes):
    return __sklearn_imputation(dataframes, "most_frequent")


def constant_imputation(dataframes):
    return __sklearn_imputation(dataframes, "constant")


def oversample_smote(dataframes, sampling_strategy="auto", random_state=40, k=8, columns=None, verbose=False):

    # convert df to dataframes
    dfs = util.df_to_dfs(dataframes)
    # initialize smote object
    smote = SMOTE(sampling_strategy=sampling_strategy,
                  random_state=random_state, k_neighbors=k)

    # loop in each dataframe
    oversampled_dfs = []
    for i in range(len(dfs)):
        n = dfs[i].shape[1] - 1

        # get the features for the df
        x = dfs[i].iloc[:, 0:n]
        # get the lables for the df
        y = dfs[i].iloc[:, n]
        y=y.astype('int') # test from Stack Ovreflow
        # output log (original)
        if(verbose):
            group, occurrences = np.unique(y, return_counts=True)
            outcomes = dict(zip(group, occurrences))
            print("original dataset (labels): " + str(outcomes))
            print("total: " + str(sum(outcomes.values())))

        # apply smote
        x_resampled, y_resampled = smote.fit_sample(x, y)

        # output log (oversampled)
        if(verbose):
            group, occurrences = np.unique(y_resampled, return_counts=True)
            outcomes = dict(zip(group, occurrences))
            print("resampled dataset (labels): " + str(outcomes))
            print("total: " + str(sum(outcomes.values())) + "\n")

        # convert oversampled arrays back to dataframes
        oversampled_instances = np.concatenate(
            (x_resampled, np.matrix(y_resampled).T), axis=1)
        oversampled_df = pd.DataFrame(
            data=oversampled_instances, columns=columns)
        oversampled_df.iloc[:, n] = oversampled_df.iloc[:, n].astype(int)
        oversampled_dfs.append(oversampled_df)

    # return oversampled dataframes
    return oversampled_dfs


def figPath():
    path = os.getcwd() 
    return path  + "/static"

