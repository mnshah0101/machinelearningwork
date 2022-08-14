
import pandas as pd
import numpy as no
import matplotlib.pyplot as plt
import seaborn as sns


#Read data


train = pd.read_csv("titanic_train.csv")



#Create dummy variables

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train["Age"]=train[["Age", "Pclass"]].apply(impute_age, axis = 1)


#Drop na values

sns.heatmap(train.isnull())
plt.show()
train.drop("Cabin', axis = 1, inplace = True)
train.dropna(inplace = True)

#Dummy variables
sex = pd.get_dummies(train["Sex"], drop_first = True)
embark = pd.get_dummies(train["Embarked"], drop_first = True)
train = pd.concat([train, sex, embark], axis = 1)

#drop irrelevant variables
train.drop(["Embarked", "Cabin", "Ticket", "Sex", "Name"], axis = 1, inplace = True)


# train test split

from sklearn.model_selection import train_test_split
X = train.drop("Survived", axis = 1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=101)


#Logistic Regression


from sklearn.linear_model import LogisticRegression
ogmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)


#evaluate

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)





