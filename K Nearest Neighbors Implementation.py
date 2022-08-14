

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


#Get the Data

df = pd.read_csv("KNN_Project_Data")




# Exploratory Data Analysis



sns.pairplot(data = df, hue = "TARGET CLASS")
plt.show()

#standardize data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop("TARGET CLASS", axis =1))
transformed_data = scaler.transform(df.drop("TARGET CLASS", axis =1))



df_scaled = pd.DataFrame(data = transformed_data, columns = df.columns[:-1])


# # Train Test Split



from sklearn.model_selection import train_test_split
X = df_scaled
y = df["TARGET CLASS"]


X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=101)


# K Nearest Neighbots

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


#Prediction
predictions = knn.predict(X_test)


# Evaluate k nearest neighbors


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

#choosing k value
errors_list = []
for i in range (1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    errors_list.append(np.mean(predictions != y_test))
print(errors_list)




