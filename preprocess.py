import datautils, convertutils


# if (bool(isFlask) == True):
    # datautils = __import__('utils')


def getDfList(arrfurl):
    print('read arff files')
    return convertListArff2Df(arrfurl)


def showDataStats(df_list):
    print('show data stats...')
    showMissingStats(df_list)
    # showMatrix(df_list)
    # showHeatMap(df_list)


def processData(df_list):
    df_list_imp = meanImputation(df_list)
    return overSampleSmote(df_list_imp)


def convertListArff2Df(arrfurl):
    print('convert arff to df...')
    paths = []
    # paths.append("../data/1year.arff")
    paths.append(arrfurl)
    df_list = convertutils.arff2df(paths)
    print(df_list)
    return df_list


def showMissingStats(df_list):
    print('show missing stats...')
    missing_stats = datautils.missing_stats(df_list)
    print(missing_stats)


def showMatrix(df_list):
    print('show matrix...')
    datautils.nullity_matrix(df_list)


def showHeatMap(df_list):
    print('show heat map')
    datautils.nullity_heatmap(df_list)


def meanImputation(df_list):
    print("using mean imputation...")
    df_list = datautils.mean_imputation(df_list)
    return df_list


def overSampleSmote(df_list):
    print("over sample using smote...")
    df_list = datautils.oversample_smote(df_list)
    return df_list