#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('fifa.csv')


# In[3]:


df


# In[8]:


kdf=df[['overall','age']]


# In[9]:


pd.set_option('display.max_info_columns',100)


# In[10]:


pd.set_option('display.max_rows',100)


# In[51]:


from sklearn.cluster import KMeans


# In[52]:



k_rang=range(1,10)
set_k=[]
for k in k_rang:
    knn=KMeans(n_clusters=k)
    knn.fit(kdf)
    set_k.append(knn.inertia_)
    
from sklearn.cluster import KMeans
knn=KMeans(n_clusters=4)


# In[53]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[54]:


df1 = kdf[kdf.cluster==0]
df2 = kdf[kdf.cluster==1]
df3 = kdf[kdf.cluster==2]
df4 = kdf[kdf.cluster==3]


# In[55]:


cluster=knn.fit_predict(kdf)
kdf['cluster']=cluster
get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


plt.scatter(df1.overall,df1.age,color='red')
plt.scatter(df2.overall,df2.age,color='green')
plt.scatter(df3.overall,df3.age,color='blue')
plt.scatter(df4.overall,df4.age,color='yellow')



plt.scatter(knn.cluster_centers_[:,0],knn.cluster_centers_[:,1],color='purple',marker='*',label='centroid')


# In[ ]:





# In[ ]:




