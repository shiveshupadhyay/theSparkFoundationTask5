#!/usr/bin/env python
# coding: utf-8

# # Shivesh Upadhyay

# # Data Pre-Processing

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[4]:


print(X_train)


# In[5]:


print(y_train)


# In[6]:


print(X_test)


# In[7]:


print(y_test)


# ### Feature Scaling

# In[8]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[9]:


print(X_train)


# In[10]:


print(X_test)


# # Training and Predicting Decision Tree Model

# In[11]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[12]:


print(classifier.predict(sc.transform([[5.8, 2.8, 5.1, 2.4]])))


# In[13]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Looking for confusion matrix and accuracy

# In[14]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # Visualising Decision Tree

# In[15]:


from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import graphviz
import os


# In[17]:


dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, feature_names= ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

