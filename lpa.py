"""
Original file is located at
    https://colab.research.google.com/drive/1wblzxUSxb1hY3QtJYA1NVLy4ZE74BT2s
"""



# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

train_data = pd.read_csv("train_dataset.csv")
train_data.head()

print(train_data.shape)

train_data.describe()

train_data.info()

def missing_values(df):
    a = num_null_values = df.isnull().sum()
    return a

missing_values(train_data)

train_data.drop(["Loan_ID", "Dependents"], axis=1, inplace=True)

train_data

### Working with Categorical null values  ###

cols = train_data[["Gender", "Married", "Self_Employed"]]
for i in cols:
    train_data[i].fillna(train_data[i].mode().iloc[0], inplace = True)

train_data.isnull().sum()

### Working with Numerical values missing_data ###

n_cols = train_data[["LoanAmount", "Loan_Amount_Term", "Credit_History"]]
for i in n_cols:
    train_data[i].fillna(train_data[i].mean(axis=0), inplace=True)

### Visualization ###

def bar_chart(col):
    Approved = train_data[train_data["Loan_Status"]=="Y"][col].value_counts()
    Disapproved = train_data[train_data["Loan_Status"]=="N"][col].value_counts()

    df1 = pd.DataFrame([Approved, Disapproved])
    df1.index = ["Approved", "Disapproved"]
    df1.plot(kind="bar")

bar_chart("Gender")

bar_chart("Married")

bar_chart("Self_Employed")

bar_chart("Education")

## Encoding categorical data ##

from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()

columns_to_encode = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]

train_data[columns_to_encode] = ord_enc.fit_transform(train_data[columns_to_encode])

train_data.head()

train_data[["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]].astype('int')

train_data

from sklearn.model_selection import train_test_split
X = train_data.drop("Loan_Status", axis=1)
y = train_data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

## Gaussian NB ML model ##

from sklearn.naive_bayes import GaussianNB

gfc = GaussianNB()
gfc.fit(X_train, y_train)
pred1 = gfc.predict(X_test)

## accuracy  ##

from sklearn.metrics import precision_score, recall_score, accuracy_score
def loss(y_true, y_pred):
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print(pre)
    print(rec)
    print(acc)

loss(y_test, pred1)

## Using SVC with Grid Search ##

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = { 'C' : [0.1, 1, 10, 100, 1000],
                'gamma' : [1, 0.1, 0.01, 0.001, 0.0001],
                 'kernel' :['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(X_train,y_train)

grid.best_params_

svc = SVC(C = 0.1, gamma = 1, kernel = 'rbf')
svc.fit(X_train, y_train)
pred2 = svc.predict(X_test)
loss(y_test,pred2)

#!pip install xgboost

## XGBoost Classifier ##

from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=2,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)

xgb.fit(X_train, y_train)
pred3 = xgb.predict(X_test)
loss_value = loss(y_test, pred3)

##  Decision Tree  aglorithm  ##

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

def randomized_search(params, runs=20, clf = DecisionTreeClassifier(random_state = 2)):
    rand_clf = RandomizedSearchCV(clf, params, n_iter= runs, n_jobs=-1, random_state = 2)
    rand_clf.fit(X_train,y_train)
    best_model = rand_clf.best_estimator_

    # Extra best score
    best_score = rand_clf.best_score_

    # Print best score
    print("Training score: {:.3f}".format(best_score))

    # Predict best set labels
    y_pred = best_model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test,y_pred)

    # Print accuracy
    print("Test score: {:.3f}".format(accuracy))

    return best_model

randomized_search(params = {
                            'criterion': ['entropy', 'gini'],
                            'splitter': ['random', 'best'],
                            'min_weight_fraction_leaf' : [0.0, 0.0025,0.005, 0.0075, 0.01],
                            'min_samples_split': [2, 3, 4, 5, 6, 8, 10],
                            'min_samples_leaf': [1, 0.01, 0.02, 0.03, 0.04],
                            'min_impurity_decrease' : [0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
                            'max_leaf_nodes' : [10, 15, 20, 25, 30, 35, 40, 45, 50, None],
                            'max_features': ['auto', 0.95, 0,90, 0.85, 0.80, 0.75, 0.70],
                            'max_depth': [None, 2, 4, 6, 8],
                            'min_weight_fraction_leaf' : [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05]
                             })

ds = DecisionTreeClassifier(max_depth=8, max_features=90, max_leaf_nodes=15,
                       min_impurity_decrease=0.005, min_samples_leaf=0.01,
                       min_weight_fraction_leaf=0.0075, random_state=2,
                       splitter='random')
ds.fit(X_train, y_train)
pred4 = ds.predict(X_test)
loss(y_test, pred4)

### Random Forest ML Algo ##

import joblib
joblib.dump(ds, "model.pkl")
model = joblib.load('model.pkl' )
model.predict(X_test)

