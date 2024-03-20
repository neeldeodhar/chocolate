#!/usr/bin/env python
# coding: utf-8

# In[1]:


#downloading dataset, importing libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder











# In[2]:


#reading the dataset, deleting missing value entries
df = pd.read_csv('flavors_of_cacao.csv')

df.columns = ['Company', 'SpecificBean', 'Ref', 'ReviewDate','CocoaPercent', 'CompanyLocation','Rating',
                  'BeanType','BeanOrigin']


df.head()

df.dropna()


# In[3]:


#calculating how many tuples in dataset,
#calculating how many unique names in dataset
#how many reviews are made in 2013 in the dataset.
#calculating how many missing values in the BeanType Column.
df = pd.read_csv('flavors_of_cacao.csv')

df.columns = ['Company', 'SpecificBean', 'Ref', 'ReviewDate','CocoaPercent', 'CompanyLocation','Rating',
                  'BeanType','BeanOrigin']

print (df.shape[0]) # to count tuples


data = pd.DataFrame()

df.head()

print (df['Company'].nunique())




print  (df['ReviewDate'].value_counts()[2013])

df['BeanType'].value_counts()['\xa0']



# In[ ]:





# In[4]:


#plotting histogram

plt.hist(data = df, x='Rating')
plt.xlabel('Rating')
plt.ylabel('Chocolate Dataset')
plt.show()
print ("majority of the chocolate bars have been rated in the range 3.0-4.0; with 3.5 being most common rating")

x = df['Rating'].values
df['CocoaPercent']= df['CocoaPercent'].str.rstrip('%').astype('float')/100.0
y = df['CocoaPercent']



# In[5]:


#plotting converted numerical Cocoa Percent Values against Rating Values
#scatter plot
plt.scatter(x,y)
plt.xlabel("Rating")
plt.ylabel("Cocoa Percentage")
plt.show()
print ("when cocoa percent is in the range 60-70%; the chocolate bars get the most rating")



# In[6]:


#normalize the Rating Column and printing results.

Scaler = MinMaxScaler()



df['Rating'] = Scaler.fit_transform(df[['Rating']])
display(df)


# In[7]:


#encoding two categorical columns with ordinal encoder

selected_columns = df[['Company','CompanyLocation']]

df[['Company', 'CompanyLocation']] = OrdinalEncoder().fit_transform(df[['Company', 'CompanyLocation']])


# In[ ]:




