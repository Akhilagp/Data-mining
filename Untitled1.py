#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[3]:


fil = pd.read_csv('fil.csv')
print(len(fil))
fil.shape


# In[8]:


X = fil.values[1:,0:17]
Y = fil.values[1:,17:]
print(X.shape,Y.shape)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[10]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[11]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


# In[ ]:




