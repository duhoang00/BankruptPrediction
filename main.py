import preprocess, process

def main():
    # raw data
    df_list = preprocess.getDfList(arrfurl)
    preprocess.showDataStats(df_list)
    # processed data
    df_list_imp = preprocess.processData(df_list)
    preprocess.showDataStats(df_list_imp)
    process.processData(df_list_imp)
    

def getRawDfList(arrfurl):
    df_list = preprocess.getDfList(arrfurl)
    return df_list

if __name__ == "__main__":
    main()