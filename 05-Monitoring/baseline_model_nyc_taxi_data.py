#!/usr/bin/env python
# coding: utf-8

# # Baseline model for batch monitoring example

# In[38]:


import requests
import datetime
import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

from joblib import load, dump
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


# In[2]:


files = [('green_tripdata_2022-02.parquet', './data'), ('green_tripdata_2022-01.parquet', './data')]

print("Download files:")
for file, path in files:
    url=f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file}"
    resp=requests.get(url, stream=True)
    save_path=f"{path}/{file}"
    with open(save_path, "wb") as handle:
        for data in tqdm(resp.iter_content(),
                        desc=f"{file}",
                        postfix=f"save to {save_path}",
                        total=int(resp.headers["Content-Length"])):
            handle.write(data)


# In[3]:


jan_data = pd.read_parquet('data/green_tripdata_2022-01.parquet')


# In[4]:


jan_data.describe()


# In[5]:


jan_data.shape


# In[6]:


# create target
jan_data["duration_min"] = jan_data.lpep_dropoff_datetime - jan_data.lpep_pickup_datetime
jan_data.duration_min = jan_data.duration_min.apply(lambda td : float(td.total_seconds())/60)


# In[7]:


# filter out outliers
jan_data = jan_data[(jan_data.duration_min >= 0) & (jan_data.duration_min <= 60)]
jan_data = jan_data[(jan_data.passenger_count > 0) & (jan_data.passenger_count <= 8)]


# In[28]:


jan_data.duration_min.hist()


# In[9]:


# data labeling
target = "duration_min"
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]


# In[10]:


jan_data.shape


# In[11]:


train_data = jan_data[:30000]
val_data = jan_data[30000:]


# In[12]:


model = LinearRegression()


# In[13]:


model.fit(train_data[num_features + cat_features], train_data[target])


# In[14]:


train_preds = model.predict(train_data[num_features + cat_features])
train_data['prediction'] = train_preds


# In[15]:


val_preds = model.predict(val_data[num_features + cat_features])
val_data['prediction'] = val_preds


# In[16]:


print(mean_absolute_error(train_data.duration_min, train_data.prediction))
print(mean_absolute_error(val_data.duration_min, val_data.prediction))


# # Dump model and reference data

# In[17]:


with open('models/lin_reg.bin', 'wb') as f_out:
    dump(model, f_out)


# In[18]:


val_data.to_parquet('data/reference.parquet')


# # Evidently Report

# In[19]:


column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features
)


# In[20]:


report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
]
)


# In[21]:


report.run(reference_data=train_data, current_data=val_data, column_mapping=column_mapping)


# In[22]:


report.show(mode='inline')


# In[23]:


result = report.as_dict()


# In[24]:


result


# In[25]:


#prediction drift
result['metrics'][0]['result']['drift_score']


# In[26]:


#number of drifted columns
result['metrics'][1]['result']['number_of_drifted_columns']


# In[27]:


#share of missing values
result['metrics'][2]['result']['current']['share_of_missing_values']


# # Evidently Dashboard

# In[32]:


from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import DashboardPanelCounter, DashboardPanelPlot, CounterAgg, PanelValue, PlotType, ReportFilter
from evidently.renderers.html_widgets import WidgetSize


# In[30]:


ws = Workspace("workspace")


# In[31]:


project = ws.create_project("NYC Taxi Data Quality Project")
project.description = "My project descriotion"
project.save()


# In[33]:


regular_report = Report(
    metrics=[
        DataQualityPreset()
    ],
    timestamp=datetime.datetime(2022,1,28)
)

regular_report.run(reference_data=None,
                  current_data=val_data.loc[val_data.lpep_pickup_datetime.between('2022-01-28', '2022-01-29', inclusive="left")],
                  column_mapping=column_mapping)

regular_report


# In[34]:


ws.add_report(project.id, regular_report)


# In[35]:


#configure the dashboard
project.dashboard.add_panel(
    DashboardPanelCounter(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        agg=CounterAgg.NONE,
        title="NYC taxi data dashboard"
    )
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Inference Count",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_rows",
                legend="count"
            ),
        ],
        plot_type=PlotType.BAR,
        size=WidgetSize.HALF,
    ),
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Number of Missing Values",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_missing_values",
                legend="count"
            ),
        ],
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
    ),
)

project.save()


# In[36]:


regular_report = Report(
    metrics=[
        DataQualityPreset()
    ],
    timestamp=datetime.datetime(2022,1,29)
)

regular_report.run(reference_data=None,
                  current_data=val_data.loc[val_data.lpep_pickup_datetime.between('2022-01-29', '2022-01-30', inclusive="left")],
                  column_mapping=column_mapping)

regular_report


# In[37]:


ws.add_report(project.id, regular_report)


# In[ ]:




