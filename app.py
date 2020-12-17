import os
import arff
import pandas as pd

from flask import Flask, render_template, request
from scipy.io import arff
from io import StringIO
import main, preprocess

app=Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")
        
@app.route('/datainput', methods=['GET', 'POST'])
def datainput():
    arrfurl = "1year.arff"
    rawDfList = main.getRawDfList(arrfurl);
    rawDfList[0] = rawDfList[0][rawDfList[0]['Attr1'] > 0.3]  # less value for dataframe
    Df_missing_stats = main.getMissingStats(rawDfList)
    return render_template("datainput.html", 
        tables=[rawDfList[0].to_html(classes='table', header="true")], 
        missingstats = [Df_missing_stats.to_html(classes="table missing-stats", header="true")])


@app.route('/choosedata/<string:chosendata>/<string:chosenimputation>', methods=['GET', 'POST'])
def chooseData(chosendata, chosenimputation):
    arrfurl = chosendata + ".arff"
    rawDfList = main.getRawDfList(arrfurl);
    if (chosenimputation != "null"):
        rawDfList = preprocess.meanImputation(rawDfList)
    rawDfList[0] = rawDfList[0][rawDfList[0]['Attr1'] > 0.3]  # less value for dataframe
    Df_missing_stats = main.getMissingStats(rawDfList)
    return render_template("datainput.html", 
        tables=[rawDfList[0].to_html(classes='table', header="true")], 
        missingstats = [Df_missing_stats.to_html(classes="table missing-stats", header="true")])


@app.route('/datapreprocess', methods=['GET', 'POST'])
def datapreprocess():
    return render_template("datapreprocess.html")


@app.route('/dataprocess', methods=['GET', 'POST'])
def dataprocess():
    return render_template("dataprocess.html")


if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)