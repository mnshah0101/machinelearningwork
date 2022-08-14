

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# Get the Data


customers = pd.read_csv("Ecommerce Customers")

#exploratory data anlysis


sns.jointplot(x = "Time on Website", y = "Yearly Amount Spent", data = customers)
plt.show()


sns.jointplot(x = "Time on App", y = "Yearly Amount Spent", data = customers)
plt.show()



sns.jointplot(x = "Time on App", y = "Length of Membership", data = customers, kind = "hex")
plt.show()


sns.pairplot(customers)
plt.show()



sns.lmplot(x = "Length of Membership", y = "Yearly Amount Spent", data = customers)
plt.show()


# Training and Testing Data



from sklearn.model_selection import train_test_split
customers.columns

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers["Yearly Amount Spent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)








# Training the Model on lin reg

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X, y)
predictions = lm.predict(X_test)
plt.ylabel("Predicted Yearly Amount Spent")
sns.scatterplot(x = y_test, y = predictions)
plt.show()


# Evaluating the Model

from sklearn import metrics
list = [metrics.mean_absolute_error(y_test, predictions), metrics.mean_squared_error(y_test, predictions),np.sqrt(metrics.mean_squared_error(y_test, predictions))]



# Residuals


sns.distplot(y_test - predictions)
plt.show()



coeffecients = pd.DataFrame(data = lm.coef_, index = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'], columns = ["Coeff"])
print(coeffecients)


