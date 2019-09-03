# SQL-Server Anomaly Detection Service

SQL Server Anomaly Detection Service is an open source software library for anomaly detection service utilizing state-of-the-art machine learning techniques. It is developed by DATA lab at Texas A&amp;M University. The goal of SQLServerAD is to provide anomaly detection services which meets the demands for users in different fields, w/wo data science or machine learning background. SQLServerAD gives the ability to execute machine learning algorithms in-database without moving data outside SQL Server or over the network.

SQL-Server Anomaly Detection Service is featured for:

- **Full Stack Service** which supports operations and maintenances from light-weight SQL based database to back-end machine learning algorithms;

- **State-of-the-art Anomaly Detection Approaches** including **Statistical/Machine Learning/Deep Learning** models with unified APIs and detailed documentation;

- **Powerful Data Analysis Mechanism** which supports both **static and time-series data** analysis with flexible time-slice(sliding-window) segmentation.  
 
#### API Demo:


```sh
from utils.import_algorithm import algorithm_selection
from utils.utilities import output_performance,connect_server,query_data

# connect to the database
conn,cursor=connect_server(host, user, password)

# query data from specific time range
data = query_data(database_name,table_name,start_time,end_time)

# train the anomaly detection algorithm
clf = algorithm_selection(algorithm_name)
clf.fit(X_train)

# get outlier result and scores
prediction_result = clf.predict(X_test)
outlierness_score = clf.decision_function(test)

#visualize the prediction_result
visualize_distribution(X_test,prediction_result,outlierness_score)

```
![](https://github.com/yli96/PyOutlierDetectionSys/blob/master/output/img/Result.png){:height="50%" width="50%"}


## Installation

To install the package, please use the [`pip`](https://pip.pypa.io/en/stable/installing/) installation as follows:

```sh
pip install sqlserveradservice
pip install git+https://github.com/yli96/PyOutlierDetectionSys.git
```
**Note:** SQL-Server Anomaly Detection Service is only compatible with **Python 3.6** and above.

#### Required Dependencies



- pandas>=0.25.0
- taos==1.4.15
- tensorflow==2.0.0b1
- numpy>=1.16.4
- seaborn>=0.9.0
- torch>=1.1.0
- luminol==0.4
- tqdm>=4.35.0
- matplotlib>=3.1.1
- scikit_learn>=0.21.3


## Gallery


## API

