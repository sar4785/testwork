
import numpy as np
from numpy import load
import pandas as pd
# import pickle

def load_data():
    data = load(r"C:\Users\Phimprasert\Documents\GitHub\testwork\little_wafer\Kaggle_Dataset\Wafer_Map_Datasets.npz") #collect file path
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])


    with open(r"C:\Users\Phimprasert\Documents\GitHub\testwork\little_wafer\Kaggle_Dataset\LSWMD.pkl", 'rb') as f: #collect file path
        df = pd.read_pickle(f)
        # data2 = pickle.load(f)

    print("Total wafers:",len(df))
    print("Column names:",(df.columns))

    return data,df

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Load the pickle file
with open(r"C:\Users\Phimprasert\Documents\GitHub\testwork\little_wafer\Kaggle_Dataset\LSWMD.pkl", 'rb') as f:
    df = pd.read_pickle(f)

# Function to convert failureType to string
def convert_failure_type(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return "Unknown"
        else:
            return str(x[0])  # Use first element
    elif pd.isna(x):
        return "Unknown"
    else:
        return str(x)

# Apply the conversion
df['failureType'] = df['failureType'].apply(convert_failure_type)

# Print unique failure types and count
print("Unique failure types:", df['failureType'].unique())
print("Number of unique failure types:", df['failureType'].nunique())
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['failureType'])
y_onehot = tf.keras.utils.to_categorical(df['label_id'])
print(le.classes_)  # ดูว่า class ไหนตรงกับเลขอะไร
df['label_id'] = le.fit_transform(df['failureType'])






