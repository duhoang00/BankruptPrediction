import numpy as np
import preprocess, process
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import OrderedDict
import pandas as pd


def main():
    print('running main')
    arrfurl = "1year.arff"
    df_list = preprocess.getDfList(arrfurl)
    meanDfList = preprocess.meanImputation(df_list)
    smoteDfList = preprocess.overSampleSmote(meanDfList)
    # features selector 
    df = smoteDfList[0]
    df_scaled = scaDf(df)
    labels = df.columns.values
    feature_chi_score, feature_chi_columns = chi2Score(df_scaled, df.values[:,-1], labels[:-1])
    top_feature_chi_columns = feature_chi_columns[:29]
    
    tmp_df = df[top_feature_chi_columns]
    tmp_df[30] = df[64]

    X_train, X_test, y_train, y_test = process.traintestsplitfast(tmp_df, 0.2)
    calAccuracy = process.randomForest(X_train, X_test, y_train, y_test)
    print(calAccuracy)


def fastTrain(df):
    # features selector
    df_scaled = scaDf(df)
    labels = df.columns.values
    feature_chi_score, feature_chi_columns = chi2Score(df_scaled, df.values[:,-1], labels[:-1])
    top_feature_chi_columns = feature_chi_columns[:29]
    tmp_df = df[top_feature_chi_columns]
    tmp_df[30] = df[64]
    X_train, X_test, y_train, y_test = process.traintestsplitfast(tmp_df, 0.2)
    confusion_matrix_result, accuracy_score_result, classification_report_result = process.randomForest(X_train, X_test, y_train, y_test)
    return confusion_matrix_result, accuracy_score_result, classification_report_result

def fastPredict(df, userinput):
    df_scaled = scaDf(df)
    labels = df.columns.values
    feature_chi_score, feature_chi_columns = chi2Score(df_scaled, df.values[:,-1], labels[:-1])
    top_feature_chi_columns = feature_chi_columns[:29]
    tmp_df = df[top_feature_chi_columns]
    tmp_df[30] = df[64]
    X_train, X_test, y_train, y_test = process.traintestsplitfast(tmp_df, 0.2)

    clf = process.RandomForestClassifier(random_state=99)
    clf.fit(X_train, y_train)
    userinputarray =userinput.split(",")
    x = np.array(userinputarray)
    y = x.astype(np.float)
    y_pred = clf.predict([y])

    return y_pred


def scaDf(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    dfScaled = scaler.transform(df)
    return dfScaled


def chi2Score(x, y, keys):
    selector = SelectKBest(chi2, k = "all").fit(x, y)
    scores = selector.scores_
    score_dictionary = dict(zip(keys, scores))
    sorted_by_value = sorted(score_dictionary.items(), key=lambda kv: kv[1], reverse=True)

    sorted_column_names = [sorted_by_value[i][0] for i in range(len(sorted_by_value))]
    return sorted_by_value, sorted_column_names


def getRawDfList(arrfurl):
    df_list = preprocess.getDfList(arrfurl)
    return df_list


def getMissingStats(df_list):
    missing_stats = preprocess.showDataStats(df_list)
    return missing_stats


if __name__ == "__main__":
    main()