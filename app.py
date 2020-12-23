import os
import arff
import pandas as pd

from flask import Flask, render_template, request
from scipy.io import arff
from io import StringIO
import main, preprocess, process
from PIL import Image

app=Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")
        

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template("about.html")

@app.route('/datainput', methods=['GET', 'POST'])
def datainput():
    df, Df_missing_stats = preProcessData("1year", "null", False)
    update_df = df.head(50)
    return render_template("datainput.html", 
        tables=[update_df.to_html(classes='table', header="true")], 
        missingstats = [Df_missing_stats.to_html(classes="table missing-stats", header="true")])


@app.route('/choosedata/<string:chosendata>/<string:chosenimputation>', methods=['GET', 'POST'])
def chooseData(chosendata, chosenimputation):
    df, Df_missing_stats = preProcessData(chosendata, chosenimputation, False)
    update_df = df.head(50)
    return render_template("datainput.html", 
        tables=[df.to_html(classes='table', header="true")], 
        missingstats = [Df_missing_stats.to_html(classes="table missing-stats", header="true")])


@app.route('/choosedatapreprocess/<string:chosendata>/<string:chosenimputation>', methods=['GET', 'POST'])
def datapreprocess(chosendata, chosenimputation):
    return render_template("datapreprocess.html",
        confusion_matrix = [], 
        accuracy_score = [], 
        classification_report = {})


@app.route('/choosedatatrain/<string:chosendata>/<string:chosenimputation>/<string:chosenmethod>/<string:chosentraintest>/<int:value>', methods=['GET', 'POST'])
def choosedatatrain(chosendata, chosenimputation, chosenmethod, chosentraintest, value):
    df, Df_missing_stats = preProcessData(chosendata, chosenimputation, True)
    if (chosentraintest == "split"):
        X_train, X_test, y_train, y_test = process.traintestsplit(df, value)
    if (chosentraintest == "kfold"):
        X_train, X_test, y_train, y_test = process.kfold(df, value)

    if (chosenmethod == "decisiontree"):
        confusion_matrix, accuracy_score, classification_report = process.decisionTree(X_train, X_test, y_train, y_test)
    elif (chosenmethod == "randomforest"):
        confusion_matrix, accuracy_score, classification_report = process.randomForest(X_train, X_test, y_train, y_test)
    elif (chosenmethod == "gaussiannb"):
        confusion_matrix, accuracy_score, classification_report = process.gaussianNB(X_train, X_test, y_train, y_test)
    elif (chosenmethod == "multinomialnb"):
        confusion_matrix, accuracy_score, classification_report = process.multinomialNB(X_train, X_test, y_train, y_test)
    elif (chosenmethod == "bernoullinb"):
        confusion_matrix, accuracy_score, classification_report = process.bernoulliNB(X_train, X_test, y_train, y_test)

    classification_report.__delitem__("accuracy")

    return render_template("datapreprocess.html", 
        confusion_matrix = confusion_matrix, 
        accuracy_score = accuracy_score, 
        classification_report = classification_report)


@app.route('/choosedatapredict/<string:chosendata>/<string:chosenimputation>/<string:chosenmethod>/<string:chosentraintest>/<int:value>', methods=['GET', 'POST'])
def choosedatapredict(chosendata, chosenimputation, chosenmethod, chosentraintest, value):
    return render_template("dataprocess.html")


@app.route('/predict/<string:chosendata>/<string:chosenimputation>/<string:chosenmethod>/<string:chosentraintest>/<int:value>/<string:userinput>', methods=['GET', 'POST'])
def predict(chosendata, chosenimputation, chosenmethod, chosentraintest, value, userinput):
    df, Df_missing_stats = preProcessData(chosendata, chosenimputation, True)
    if (chosentraintest == "split"):
        X_train, X_test, y_train, y_test = process.traintestsplit(df, value)
    if (chosentraintest == "kfold"):
        X_train, X_test, y_train, y_test = process.kfold(df, value)

    if (chosenmethod == "decisiontree"):
        result = process.decisionTreePredict(X_train, X_test, y_train, y_test, userinput)
    elif (chosenmethod == "randomforest"):
        result = process.randomForestPredict(X_train, X_test, y_train, y_test, userinput)
    elif (chosenmethod == "gaussiannb"):
        result = process.gaussianNBPredict(X_train, X_test, y_train, y_test, userinput)
    elif (chosenmethod == "multinomialnb"):
        result = process.multinomialNBPredict(X_train, X_test, y_train, y_test, userinput)
    elif (chosenmethod == "bernoullinb"):
        result = process.BernoulliNBPredict(X_train, X_test, y_train, y_test, userinput)
    
    return render_template("dataprocess.html", result = result)


def preProcessData(chosendata, chosenimputation, smote):
    print('preprocess data ' + chosendata + ' with ' + chosenimputation + ' and ' + str(smote) + ' smote')
    arrfurl = chosendata + ".arff"
    rawDfList = main.getRawDfList(arrfurl)
    if (chosenimputation != "null"):
        if (chosenimputation == "meanImp"):
            rawDfList = preprocess.meanImputation(rawDfList)
        elif (chosenimputation == "medianImp"):
            rawDfList = preprocess.medianImputation(rawDfList)
        elif (chosenimputation == "mostFrequentImp"):
            rawDfList = preprocess.mostFrequentImputation(rawDfList)
        elif (chosenimputation == "constantImp"):
            rawDfList = preprocess.constantImputation(rawDfList)
    else:
        processFig(rawDfList)
    if (smote == True):
        rawDfList = preprocess.overSampleSmote(rawDfList)
    Df_missing_stats = main.getMissingStats(rawDfList)
    return  rawDfList[0], Df_missing_stats


def processFig(rawDfList):
    print('processFig with length = ' + str(len(rawDfList[0].index)))
    preprocess.nullityMatrix(rawDfList)
    preprocess.nullityHeatmap(rawDfList)

@app.errorhandler(404)
def not_found(e):
    return render_template("nodata.html")


if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='localhost', port=port, debug=True)