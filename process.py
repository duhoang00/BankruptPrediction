import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.datasets import load_iris
from graphviz import Digraph
import graphviz
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 


def processData(df_list):
    print('processing data...')
    # decisionTree(df_list)


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



def decisionTree(X_train, X_test, y_train, y_test):
    print('using decision tree...')
    # labels = ['net profit / total assets', 'Liabilities / total assets', 'working capital / total assets', 'current assets / short-term liabilities', '[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365', 'retained earnings / total assets', 'EBIT / total assets', 'book value of equity / total liabilities', 'sales / total assets', 'equity / total assets', '(gross profit + extraordinary items + financial expenses) / total assets', 'gross profit / short-term liabilities', '(gross profit + depreciation) / sales', '(gross profit + interest) / total assets', '(total liabilities * 365) / (gross profit + depreciation)', '(gross profit + depreciation) / total liabilities', 'total assets / total liabilities', 'gross profit / total assets', 'gross profit / sales', '(inventory * 365) / sales', 'sales (n) / sales (n-1)', 'sales (n) / sales (n-1)', 'net profit / sales', 'gross profit ( in 3 years) / total assets', '(equity - share capital) / total assets', '(net profit + depreciation) / total liabilities', 'profit on operating activities / financial expenses', 'working capital / fixed assets', 'logarithm of total assets', '(total liabilities - cash) / sales', '(gross profit + interest) / sales', '(current liabilities * 365) / cost of products sold', 'operating expenses / short-term liabilities', 'operating expenses / total liabilities', 'profit on sales / total assets', 'total sales / total assets', '(current assets - inventories) / long-term liabilities', 'constant capital / total assets', 'profit on sales / sales', '(current assets - inventory - receivables) / short-term liabilities', 'total liabilities / ((profit on operating activities + depreciation) * (12/365))', 'profit on operating activities / sales', 'rotation receivables + inventory turnover in days', '(receivables * 365) / sales', 'net profit / inventory', '(current assets - inventory) / short-term liabilities', '(inventory * 365) / cost of products sold', 'EBITDA (profit on operating activities - depreciation) / total assets', 'EBITDA (profit on operating activities - depreciation) / sales', 'current assets / total liabilities', 'short-term liabilities / total assets', '(short-term liabilities * 365) / cost of products sold)', 'equity / fixed assets', 'constant capital / fixed assets', 'working capital', '(sales - cost of products sold) / sales', '(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)', 'total costs / total sales', 'long-term liabilities / equity', 'sales / inventory', 'sales / receivables', '(short-term liabilities * 365) / sales', 'sales / short-term liabilities', 'sales / fixed assets']
    # df = df_list[0]
    # features = list(df.columns[:64])
    
    # X = df[features]
    # y = df[64]
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) # train/test split
    

    # X_train, X_test, y_train, y_test = traintestsplit(df)

    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(X_train, y_train)

    # test_array_Dat = [[20.1,5.2,3.1,3.2,8.1,15.45,2.56,3.25,4.89,-78.36,-89.35,85.36,5.56,4.96,8.32,7.35,4.58,9.1,9.9,-78.45,58.23,56.42,52.12,12.12,11.11,-48.45,89.56,0.32,0.96,0.78,0.56,0.63,0.99,1.56,2.35,3.3,3.45,-4.58,-98.54,20.36,31.32,33.35,34.56,35.59,89.56,0.2,0.45,42.36,25.37,60.61,78.79,79.80,-12.35,48.56,58.21,80.56,89.56,90.21,-90.36,23.25,99.56,46.20,58.78,50.37]]
    # test_array_Duc = [[0.92487,0.78606,0.68688,-0.82687,0.72373,0.75405,0.74261,0.58412,-0.31493,0.63758,-0.42777,0.54247,0.68963,0.84870,0.01022,0.57457,0.13920,0.95017,0.16257,0.59921,0.19386,0.47020,-0.26112,0.92701,0.00201,0.24212,0.70917,0.78829,0.39125,-0.74134,0.00985,0.78097,0.29665,0.46116,0.80078,0.05611,0.67901,-0.88062,0.25476,0.58680,0.95155,0.85256,0.47475,0.67420,-0.39732,0.71111,0.93372,0.31703,0.78347,0.72122,-0.44746,0.45480,0.58294,0.39582,-0.85312,0.19873,0.38607,0.43410,-0.69070,0.58623,0.63271,-0.88194,0.68870,0.00412]]

    y_pred = dt.predict(X_test)

    return cal_accuracy(y_test, y_pred)
    

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