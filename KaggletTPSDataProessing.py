#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("/Users/moksh/Downloads/tabular-playground-series-jul-2022/data.csv")


# In[3]:


df.shape


# In[4]:


df = df.drop('id', axis = 1)


# In[5]:



df.hist(figsize=(12,12))


# In[9]:


from sklearn.preprocessing import PowerTransformer


# In[7]:


scaler = PowerTransformer()


# In[8]:


df_transformed = scaler.fit_transform(df)


# In[45]:


df_transformed = pd.DataFrame(data = df_transformed)
df_transformed.hist(figsize=(12,12))


# In[9]:


from sklearn.mixture import BayesianGaussianMixture


# In[10]:


gm = BayesianGaussianMixture(
n_components=7,
verbose=1,
n_init =10,
verbose_interval=100,
random_state=2
)


# In[11]:


predicted = gm.fit_predict(df_transformed)
probs = gm.predict_proba(df_transformed)


# In[40]:


df_transformed = pd.DataFrame(df_transformed)
df_transformed['class'] = predicted


# In[41]:


df_transformed.head()


# In[43]:


df_transformed['max probability'] = np.max(probs, axis = 1)


# In[52]:


df_transformed.to_csv("data_preprocessed.csv")


# In[46]:





# In[49]:





# In[ ]:




