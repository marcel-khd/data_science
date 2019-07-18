#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install statsmodels')


# In[ ]:


# Bring our packages in 
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Connect to our data
myRegressionData = pd.read_csv('regression-py.csv')


# In[ ]:


# View a snapshot of our data
myRegressionData.head(3)


# In[7]:


# Plot the data
myRegressionData.plot(kind='scatter',x='broadcast',y='sales')


# In[8]:


# Calculate r-squared
slope, intercept, r_value, p_value, std_err = stats.linregress(myRegressionData.broadcast,myRegressionData.sales)


# In[9]:


# Print the r-squared value
print("r-squared:",r_value**2)


# In[11]:


# Model OLS to generate coefficients
myLinearModel = smf.ols(formula='sales ~ broadcast', data = myRegressionData).fit()


# In[12]:


# Output our coefficient
myLinearModel.params


# In[ ]:




