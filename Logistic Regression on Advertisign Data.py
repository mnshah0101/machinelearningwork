
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Get the Data
ad_data = pd.read_csv("advertising.csv")


sns.set_style("whitegrid")
sns.distplot(ad_data["Age"], kde = False, bins = 30)
plt.show()

sns.jointplot(x = "Area Income", y = "Age", data = ad_data)
plt.show()


sns.jointplot(y  = "Daily Time Spent on Site", x = "Age", data = ad_data, kind = "kde" )
plt.show()


sns.jointplot(x  = "Daily Time Spent on Site", y = "Daily Internet Usage", data = ad_data, kind = "scatter" )
plt.show()


sns.pairplot(data = ad_data, hue = "Clicked on Ad")
plt.show()


# Logistic Regression


from sklearn.model_selection import train_test_split
ad_data.columns

X = ad_data.drop(["Clicked on Ad", "Ad Topic Line", "City", "Country", "Timestamp"], axis = 1)
y = ad_data["Clicked on Ad"]

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=101)




from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)


predictions = logmodel.predict

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print(classification_
      report(y_test,predictions))
