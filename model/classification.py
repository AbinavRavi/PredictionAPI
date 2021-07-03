import numpy as np
import pandas as pd
from sklearn import train_test_split

class classification_model:
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(self.dataset_path)

    def __preprocess(self):
        df = self.data
        features = df.drop(['Unnamed: 32','id'],axis=1)
        labels = df['diagnosis']
        return features,labels

    def model(self,train_data,train_labels):
        

