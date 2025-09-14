#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().system('pip3 freeze | grep scikit-learn')


# In[1]:


get_ipython().system('python3 -V')


# In[2]:


import pickle
import pandas as pd


# In[3]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[4]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


# In[5]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[14]:


df.head()


# In[6]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[ ]:


import numpy as np

print("Standard deviation of predicted durations:", np.std(y_pred))


# In[19]:


import pyarrow
year = 2023
month = 3
output_file = f"predictions_{year:04d}_{month:02d}.parquet"

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_predictions = pd.DataFrame({
    "ride_id": df['ride_id'],
    "predicted_duration": y_pred
})

df_predictions.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

