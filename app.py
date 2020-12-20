import os
import arff
import pandas as pd

from flask import Flask, render_template, request
from scipy.io import arff
from io import StringIO
import main, preprocess, process

app=Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")
        
@app.route('/datainput', methods=['GET', 'POST'])
def datainput():
    df, Df_missing_stats = preProcessData("1year", "null", False)
    return render_template("datainput.html", 
        tables=[df.to_html(classes='table', header="true")], 
        missingstats = [Df_missing_stats.to_html(classes="table missing-stats", header="true")])


@app.route('/choosedata/<string:chosendata>/<string:chosenimputation>', methods=['GET', 'POST'])
def chooseData(chosendata, chosenimputation):
    df, Df_missing_stats = preProcessData(chosendata, chosenimputation, False)
    return render_template("datainput.html", 
        tables=[df.to_html(classes='table', header="true")], 
        missingstats = [Df_missing_stats.to_html(classes="table missing-stats", header="true")])


#test
@app.route('/datapreprocess', methods=['GET', 'POST'])
def datapreprocesstest():
    return render_template("datapreprocess.html",
        confusion_matrix = [], 
        accuracy_score = [], 
        classification_report = {})


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
        if (value != "null"):
            X_train, X_test, y_train, y_test = process.traintestsplit(df, value)
        else:
            X_train, X_test, y_train, y_test = process.traintestsplit(df)

    if (chosenmethod == "decisiontree"):
        confusion_matrix, accuracy_score, classification_report = process.decisionTree(X_train, X_test, y_train, y_test)

    classification_report.__delitem__("accuracy")

    return render_template("datapreprocess.html", 
        confusion_matrix = confusion_matrix, 
        accuracy_score = accuracy_score, 
        classification_report = classification_report)


@app.route('/dataprocess', methods=['GET', 'POST'])
def dataprocess():
    return render_template("dataprocess.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    return render_template("dataprocess.html")


def preProcessData(chosendata, chosenimputation, smote):
    arrfurl = chosendata + ".arff"
    rawDfList = main.getRawDfList(arrfurl);
    if (chosenimputation != "null"):
        rawDfList = preprocess.meanImputation(rawDfList)
    if (smote == True):
        rawDfList = preprocess.overSampleSmote(rawDfList)
    # rawDfList[0] = rawDfList[0][rawDfList[0]['Attr1'] > 0.3]  # less value for dataframe
    Df_missing_stats = main.getMissingStats(rawDfList)
    return  rawDfList[0], Df_missing_stats



@app.errorhandler(404)
def not_found(e):
    return render_template("nodata.html")


if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)