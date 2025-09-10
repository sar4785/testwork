
import numpy as np
from numpy import load
import pandas as pd
# import pickle

def load_data():
    data = load(r"C:\Users\Phimprasert\Desktop\Wafer-classification-DCNN\Kaggle_Dataset\Wafer_Map_Datasets.npz") #collect file path
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])


    with open(r"C:\Users\Phimprasert\Desktop\Wafer-classification-DCNN\Kaggle_Dataset\LSWMD.pkl", 'rb') as f: #collect file path
        df = pd.read_pickle(f)
        # data2 = pickle.load(f)

    print("Total wafers:",len(df))
    print("Column names:",(df.columns))

    return data,df




