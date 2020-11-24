#!/usr/bin/env python
# coding: utf-8

# In[30]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[1]:


import pandas as pd
df=pd.read_csv('mall.csv')


# In[3]:


x=df[['Annual Income (k$)','Spending Score (1-100)']]


# In[4]:


x.head()


# In[6]:


from sklearn.cluster import DBSCAN
db=DBSCAN(eps=3,min_samples=4)


# In[14]:


model=db.fit(x)
import numpy as np


# In[18]:


import numpy as np
labels=model.labels_
sample_cores=np.zeros_like(labels,dtype=bool)


# In[35]:


sample_cores[db.core_sample_indices_]=True


# In[33]:



plt.scatter(x.iloc[:,0],x.iloc[:,1],c=labels)


# In[36]:


n_clusters=len(set(labels))- (1 if -1 in labels else 0)


# In[37]:


n_clusters


# In[ ]:




