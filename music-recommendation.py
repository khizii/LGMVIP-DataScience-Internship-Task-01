#!/usr/bin/env python
# coding: utf-8

# # Music Recommendation System

# # Importing libraries

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import sweetviz as sv
import seaborn as sns
import plotly
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[39]:


data=pd.read_csv('C:/Users/admin/Desktop/track_records.csv')


# In[40]:


data


# In[41]:


data.head()


# In[42]:


data.tail()


# In[43]:


data.info


# In[44]:


print(data.shape)
print(data.columns)


# In[45]:


my_report = sv.analyze(data)
my_report.show_html()


# In[46]:


numeric_data = data.select_dtypes(include=[np.number])
categorical_data = data.select_dtypes(exclude=[np.number])


# In[47]:


print('numerical columns are:\n{}'.format(numeric_data.columns))
print('Categorical columns are:\n{}'.format(categorical_data.columns))


# In[48]:


numeric_data.describe()


# In[49]:


import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Box(y=numeric_data['acousticness'],name='Acousticness'))
fig.add_trace(go.Box(y=numeric_data['energy'],name='Energy'))
fig.add_trace(go.Box(y=numeric_data['danceability'],name='Danceability'))
fig.add_trace(go.Box(y=numeric_data['liveness'],name='liveness'))
fig.show()


# In[50]:


fig = go.Figure(go.Box(y=numeric_data['popularity'],name='Popularity'))
fig.show()


# In[51]:


fig = go.Figure(go.Scatter(y=numeric_data['popularity'],name='Popularity'))
fig.show()


# In[52]:


fig = go.Figure(go.Histogram(x=numeric_data['year']))
fig.show()


# In[53]:


categorical_data.columns


# In[54]:


print(categorical_data['artists'].nunique())


# In[55]:


n_artists=categorical_data['artists'].value_counts().head(10)
print(n_artists)


# In[56]:


fig = go.Figure(go.Bar(y=n_artists,name='Artists VS Number of songs'))
fig.show()


# In[57]:


corrMatrix=numeric_data.corr()
sns.heatmap(corrMatrix)
plt.show()