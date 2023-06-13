#!/usr/bin/env python
# coding: utf-8

# # Importing essential libraries

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


# # Load the dataset

# In[2]:


cancer = datasets.load_breast_cancer()


# # Exploring Data /Data Analysis

# In[3]:


# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)


# # Split the data into training/testing sets

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.20,random_state=109) # 70% training and 30% test


# # Generating Model

# In[5]:


clf = svm.SVC(kernel='linear') # Linear Kernel


# # Train the model using the training sets

# In[6]:


clf.fit(X_train, y_train)


# # Predict the response for test dataset

# In[7]:


y_pred = clf.predict(X_test)


# # Evaluating the Model

# In[9]:


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

print("Confusiuon Matrix:",metrics.confusion_matrix(y_test, y_pred))


# In[ ]:





# In[ ]:




