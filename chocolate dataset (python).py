#!/usr/bin/env python
# coding: utf-8

# In[6]:


#downloading dataset, importing libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder






# In[7]:


#reading the dataset, deleting missing value entries
df = pd.read_csv('flavors_of_cacao.csv')

df.columns = ['Company', 'SpecificBean', 'Ref', 'ReviewDate','CocoaPercent', 'CompanyLocation','Rating',
                  'BeanType','BeanOrigin']


df.head()

df.dropna()


# In[14]:


#calculating how many tuples in dataset,
#calculating how many unique names in dataset
#how many reviews are made in 2013 in the dataset.
#calculating how many missing values in the BeanType Column.


print (df.shape)


data = pd.DataFrame()

df.head()

print (df['Company'].nunique())




print  (df['ReviewDate'].value_counts()[2013])

print (df['BeanType'].isnull().sum())


# In[9]:


#plotting histogram

plt.hist(x='Rating')


# In[10]:


#plotting converted numerical Cocoa Percent Values against Rating Values
#scatter plot
x = df['Rating'].values
df['CocoaPercent']= df['CocoaPercent'].str.rstrip('%').astype('float')/100.0
y = df['CocoaPercent']

plt.scatter(x,y)
plt.show()


# In[11]:


#normalize the Rating Column and printing results.

Scaler = MinMaxScaler()



df['Rating'] = Scaler.fit_transform(df[['Rating']])
display(df)


# In[13]:


#encoding two categorical columns with ordinal encoder

selected_columns = df[['Company','CompanyLocation']]

df[['Company', 'CompanyLocation']] = OrdinalEncoder().fit_transform(df[['Company', 'CompanyLocation']])


# In[ ]:




