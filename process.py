import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.datasets import load_iris
from graphviz import Digraph
import graphviz
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
import numpy as np


def processData(df_list):
    print('processing data...')


def kfold(df, k):
    kf = KFold(n_splits=2)
    kf.get_n_splits(X)
    print(kf) 

    KFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


def traintestsplit(df, test_size):
    print('using train test split...')
    # labels = ['net profit / total assets', 'Liabilities / total assets', 'working capital / total assets', 'current assets / short-term liabilities', '[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365', 'retained earnings / total assets', 'EBIT / total assets', 'book value of equity / total liabilities', 'sales / total assets', 'equity / total assets', '(gross profit + extraordinary items + financial expenses) / total assets', 'gross profit / short-term liabilities', '(gross profit + depreciation) / sales', '(gross profit + interest) / total assets', '(total liabilities * 365) / (gross profit + depreciation)', '(gross profit + depreciation) / total liabilities', 'total assets / total liabilities', 'gross profit / total assets', 'gross profit / sales', '(inventory * 365) / sales', 'sales (n) / sales (n-1)', 'sales (n) / sales (n-1)', 'net profit / sales', 'gross profit ( in 3 years) / total assets', '(equity - share capital) / total assets', '(net profit + depreciation) / total liabilities', 'profit on operating activities / financial expenses', 'working capital / fixed assets', 'logarithm of total assets', '(total liabilities - cash) / sales', '(gross profit + interest) / sales', '(current liabilities * 365) / cost of products sold', 'operating expenses / short-term liabilities', 'operating expenses / total liabilities', 'profit on sales / total assets', 'total sales / total assets', '(current assets - inventories) / long-term liabilities', 'constant capital / total assets', 'profit on sales / sales', '(current assets - inventory - receivables) / short-term liabilities', 'total liabilities / ((profit on operating activities + depreciation) * (12/365))', 'profit on operating activities / sales', 'rotation receivables + inventory turnover in days', '(receivables * 365) / sales', 'net profit / inventory', '(current assets - inventory) / short-term liabilities', '(inventory * 365) / cost of products sold', 'EBITDA (profit on operating activities - depreciation) / total assets', 'EBITDA (profit on operating activities - depreciation) / sales', 'current assets / total liabilities', 'short-term liabilities / total assets', '(short-term liabilities * 365) / cost of products sold)', 'equity / fixed assets', 'constant capital / fixed assets', 'working capital', '(sales - cost of products sold) / sales', '(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)', 'total costs / total sales', 'long-term liabilities / equity', 'sales / inventory', 'sales / receivables', '(short-term liabilities * 365) / sales', 'sales / short-term liabilities', 'sales / fixed assets']
    # df = df_list[0]
    features = list(df.columns[:64])
    X = df[features]
    y = df[64]
    
    if (test_size == "null"):
        test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return  X_train, X_test, y_train, y_test


def randomForest(X_train, X_test, y_train, y_test):
    print('using random forest...')
    clf = RandomForestClassifier(random_state=99)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return cal_accuracy(y_test, y_pred)


def randomForestPredict(X_train, X_test, y_train, y_test, userinput):
    print('using radom forest for prediction...')
    clf = RandomForestClassifier(random_state=99)
    clf.fit(X_train, y_train)
    userinputarray =userinput.split(",");
    x = np.array(userinputarray)
    y = x.astype(np.float)
    y_pred = clf.predict([y])

    return y_pred;


def decisionTree(X_train, X_test, y_train, y_test):
    print('using decision tree...')
    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return cal_accuracy(y_test, y_pred)
    

def decisionTreePredict(X_train, X_test, y_train, y_test, userinput):
    print('using decision tree for prediction...')
    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(X_train, y_train)
    userinputarray =userinput.split(",");
    x = np.array(userinputarray)
    y = x.astype(np.float)
    y_pred = dt.predict([y])

    return y_pred;

def cal_accuracy(y_test, y_pred):
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix: ", confusion_matrix_result)
    accuracy_score_result = accuracy_score(y_test,y_pred)*100
    # print ("Accuracy : ", accuracy_score_result) 
    classification_report_result = classification_report(y_test, y_pred, output_dict=True)
    # print("Report : ", classification_report_result)
    
    return confusion_matrix_result, accuracy_score_result, classification_report_result

def createDecisionTreeGraph(dt, labels):
    dot_data = export_graphviz(dt, feature_names=labels,filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("decision_tree_year1")