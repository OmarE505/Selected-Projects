#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


bankdata = pd.read_csv("C:/Users/OmarAyman/PycharmProjects/Selected-Projects/Numerical SVM/bill_authentication.csv")


# In[5]:


bankdata.shape
bankdata.head()


# In[6]:


X = bankdata.drop('Class', axis=1)
y = bankdata['Class']


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[8]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


# In[9]:


y_pred = svclassifier.predict(X_test)


# In[10]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:




