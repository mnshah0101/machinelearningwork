#Linear Regression on Boston home data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


#load USA Housing data into DataFrame
df = pd.read_csv("USA_Housing.csv")


#Exploratory Data Analysis

sns.pairplot(df)
plt.show()

sns.distplot(df["Price"])
plt.show()

sns.heatmap(df.corr())
plt.show()

#X and Y arrays
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = df["Price"]

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


#linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.intercept_)
print(lm.coef_)

#coeffecient df
cdf = pd.DataFrame(lm.coef_, X.columns, columns = ["Coefficient"])


#boston home data
from sklearn.datasets import load_boston
boston = load_boston()
keys = list(boston['feature_names'])
bostondf = pd.DataFrame(data = boston["data"], columns = boston["feature_names"])
bostondf["Price"] = boston["target"]
Z = bostondf[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
V = bostondf["Price"]

#train test split
Z_train, Z_test, V_train, V_test = train_test_split(Z, V, test_size=0.4, random_state=101)


#lienar regression
lm1 = LinearRegression()
lm1.fit(Z_train,V_train)

#intercepts and predictions
intercepts = pd.DataFrame(data = lm1.coef_, index = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT'], columns = ["Intercept"])

predictions = lm.predict(X_test)


#linear regression evaluation
plt.figure(figsize = (12,12))
plt.xlabel("True Results")
plt.ylabel("Predicted Results")
sns.scatterplot(x = y_test, y = predictions)
plt.show()
sns.distplot(y_test - predictions)
plt.show()



from sklearn import metrics
metrics.mean_absolute_error(y_test, predictions)
metrics.mean_squared_error(y_test, predictions)

bostonpredictions = lm1.predict(Z_test)

#Linear Regression Evaluation
plt.figure(figsize = (10,10))
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
sns.scatterplot(x = V_test, y = bostonpredictions)
plt.show()
sns.distplot(V_test - bostonpredictions)
plt.show()
metrics.mean_squared_error(V_test, bostonpredictions)
metrics.mean_absolute_error(V_test, bostonpredictions)




