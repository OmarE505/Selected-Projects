#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[96]:


data = pd.read_csv("C:/Users/desit/Desktop/Selected/Numerical/Numerical ANN/Churn_Modelling.csv")


# In[97]:


X = data.iloc[:,3:-1].values


# In[98]:


Y = data.iloc[:,-1].values


# In[99]:


from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()
X[:,2] = np.array(LE1.fit_transform(X[:,2]))


# In[100]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct =ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder="passthrough")
X = np.array(ct.fit_transform(X))


# In[101]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[102]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[103]:


ann = tf.keras.models.Sequential()


# In[104]:


ann.add(tf.keras.layers.Dense(units=6,activation="relu"))


# In[105]:


ann.add(tf.keras.layers.Dense(units=6,activation="relu"))


# In[106]:


ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))


# In[107]:


ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


# In[108]:


ann.fit(X_train,Y_train,batch_size=32,epochs = 100)


# In[93]:


print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1,50000]])) > 0.5)


# In[94]:


ann.save("ANN.h5")


# In[ ]:




