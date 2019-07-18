#!/usr/bin/env python
# coding: utf-8

# In[27]:


# import our packages
import pandas as pd
# command that allows the notebook to list visualizations inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# connect to our data
myExploratoryData = pd.read_csv("exploratory-py.csv")


# In[41]:


# see a summary of our data
myExploratoryData


# In[24]:


get_ipython().system('pip install seaborn')


# In[26]:


import seaborn as sns


# In[38]:


# visualize our data
ax = sns.kdeplot(myExploratoryData.cpa)


# In[39]:


# visualize our data with additional detail
sns.distplot(myExploratoryData.cpa)


# In[42]:


# pivot the data
myETLData =  myExploratoryData.pivot("keyword","impressions","cpa")


# In[43]:


# see a summary of our pivot
myETLData


# In[45]:


# visualize our data
sns.heatmap(myETLData)

