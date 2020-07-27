#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


import pandas as pd


# In[ ]:





# In[7]:


df=pd.read_csv("C:/Users/Lenovo/Desktop/01_District_wise_crimes_committed_IPC_2001_2012.csv")


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


df.head()


# In[11]:


df.isnull().values.any()



# In[13]:


df.plot(x='YEAR', y='RAPE')


# In[8]:


x='YEAR'
y='BURGLARY'
df.plot(x,y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


y=df.YEAR


# In[11]:


x=df.drop(['DISTRICT', 'STATE/UT', 'YEAR'], axis=1)


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[13]:


x_train.shape


# In[14]:


x_test.shape


# In[15]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[16]:


import numpy as np 
import matplotlib.pyplot as plt


# In[17]:


n = np.size(x)
m_x, m_y = np.mean(x), np.mean(y)


# In[18]:


SS_xy = np.sum(y*x) - n*m_y*m_x 



# In[19]:


SS_xx = np.sum(x*x) - n*m_x*m_x


# In[20]:


b_1 = SS_xy / SS_xx


# In[21]:


b_0 = m_y - b_1*m_x


# In[22]:


max_x = np.max(x) + 100
min_x = np.min(x) - 100


# In[23]:


plt.plot(x, y, color='#EF4423', label='Regression Line')


# In[24]:


reg=linear_model.LinearRegression()


# In[25]:


reg.fit(x_train, y_train)


# In[26]:


y_pred=reg.predict(x_test)


# In[1]:


acc=reg.score(x_test,y_test)
acc


# In[30]:


plt.plot(x_test['MURDER'], y_pred, color='#EF4423', label='Regression Line')


# In[ ]:




