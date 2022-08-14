#Breat Cancer Support Vector Machine Identifier


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#Load data
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
df_feat = pd.DataFrame(data = cancer["data"], columns = cancer["feature_names"])


#train test split
from sklearn.model_selection import train_test_split

X = df_feat
y = cancer["target"]

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=101)


# Support Vector Machines


from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)
predictions = model.predict(X_test)


#Evaluate


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))



error_list = (y_test == predictions)


#Grid Search + Evaluation
from sklearn.model_selection import GridSearchCV


param_grid = {"C": [0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose = 3)

grid.fit(X_train, y_train)

grid.best_params_

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))

print(classification_report(y_test, grid_predictions))





