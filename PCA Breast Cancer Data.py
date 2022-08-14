#Breat Cancer PCA


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns

#Get data

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer["data"], columns = cancer["feature_names"])

#scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)


#PCA - find variables that affect variation
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)




plt.figure(figsize = (8,6))
plt.scatter(x_pca[:,0],x_pca[:,1], c = cancer["target"])
plt.xlabel("First Principal Compoenent")
plt.ylabel("second Principal Component")
plt.show()


print(pca.components_)

df_comp = pd.DataFrame(pca.components_, columns = cancer["feature_names"])
sns.heatmap(df_comp)
plt.show()




