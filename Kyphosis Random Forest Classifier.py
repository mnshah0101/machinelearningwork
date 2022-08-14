#Kyphosis Prediction Using Random Forest


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("kyphosis.csv")


sns.pairplot(df, hue = "Kyphosis")
plt.show()

#Train Test Split
from sklearn.model_selection import train_test_split
X = df.drop("Kyphosis", axis  =1 )
y = df["Kyphosis"]
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3)


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

#decision tree evalutaion
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)


#RFC Evaluation

print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))


