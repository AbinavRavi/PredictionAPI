import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def  read_data(dataset_path):
    data = pd.read_csv(dataset_path)
    return data

def preprocess(data):
    df = data
    features = df.drop(['Unnamed: 32','id'],axis=1)
    labels = df['diagnosis']
    return features,labels

def split_data(features,labels):
    train_data,test_data,train_labels,test_labels = train_test_split(features,labels,test_size=0.2)
    return train_data,test_data,train_labels,test_labels

def train_model(train_data,train_labels,test_data,test_labels):
    model = LogisticRegression(penalty='l1')
    model.fit(train_data,train_labels)
    predictions = model.predict(test_data)
    return predictions




