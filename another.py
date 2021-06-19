from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


diabetes = pd.read_csv("diabetes.csv",header=None)

X = diabetes.drop(columns=8,axis=0)
y=diabetes[8]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

model = LogisticRegression()

model = model.fit(X,y)

pickle = pickle.dump(model,open('diabetes.py','wb'))
