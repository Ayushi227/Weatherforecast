import urllib.request
import calplot
import json
import requests
import pandas as pd
from collections import namedtuple
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt

api_endpoint = "http://history.openweathermap.org/data/2.5/history/city"

city="pune"
start = "1588320000"
end = "1617177600"
timespan = "1259231&type=hour&start="
num = 1588320000

apikey = "48128adf2151a7b0d01be89c19424ff2"


features = ["date", "temp", "temp_min", "temp_max", "feels_like", "pressure", "humidity"]
DailySummary = namedtuple("DailySummary", features)


def extract_weather_data():
    records=[]
    num = 1588320000
    start = "1588320000"
    for i in range(1588320000, 1617177600, 86400):
        url = api_endpoint + "?id=" + timespan + start + "&end=" + start + "&appid=" + apikey + "&units=metric"
        request = url
        response = requests.get(request)
        if response.status_code == 200:
            date = int(response.json()["list"][0]["dt"])
            local_time = datetime.fromtimestamp(date)
            data_date = local_time.strftime("%d-%m-%Y")
            data_temp = response.json()["list"][0]["main"]["temp"]
            data_temp_min = response.json()["list"][0]["main"]["temp_min"]
            data_temp_max = response.json()["list"][0]["main"]["temp_max"]
            data_feels_like = response.json()["list"][0]["main"]["feels_like"]
            data_pressure = response.json()["list"][0]["main"]["pressure"]
            data_humidity = response.json()["list"][0]["main"]["humidity"]
            records.append(DailySummary(
                data_date,
                data_temp,
                data_temp_min,
                data_temp_max,
                data_feels_like,
                data_pressure,
                data_humidity))

        num = num + 86400
        start = str(num)
    return records


records = extract_weather_data()
print(records)

print("City: " + city.upper())

df = pd.DataFrame(records, columns=features).set_index('date')

tmp = df[['temp', 'temp_min', 'temp_max']].head(10)
tmp

# 1 day prior
N = 1

# target measurement of temperature
feature = 'temp'

# total number of rows
rows = tmp.shape[0]

# a list representing Nth prior measurements of feature
# notice that the front of the list needs to be padded with N
# None values to maintain the consistent rows length for each N
nth_prior_measurements = [None]*N + [tmp[feature][i-N] for i in range(N, rows)]

# make a new column name of feature_N and add to DataFrame
col_name = "{}_{}".format(feature, N)
tmp[col_name] = nth_prior_measurements
tmp


def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements


for feature in features:
     if feature != 'date':
        for N in range(1, 4):
            derive_nth_day_feature(df, feature, N)
print(df.columns)

to_remove = [feature 
             for feature in features 
             if feature not in ['temp', 'temp_min', 'temp_max']]

# make a list of columns to keep
to_keep = [col for col in df.columns if col not in to_remove]

# select only the columns in to_keep and assign to df
df = df[to_keep]
df.columns

df.info()

df = df.apply(pd.to_numeric, errors='coerce')
df.info()

# Call describe on df and transpose it due to the large number of columns
spread = df.describe().T

# precalculate interquartile range for ease of use in next calculation
IQR = spread['75%'] - spread['25%']

# create an outliers column which is either 3 IQRs below the first quartile or
# 3 IQRs above the third quartile
spread['outliers'] = (spread['min'] < (spread['25%']-(3*IQR))) | (spread['max'] > (spread['75%']+3*IQR))
# just display the features containing extreme outliers
spread.loc[spread.outliers]
spread

# %matplotlib inline
plt.rcParams['figure.figsize'] = [14, 8]
df.humidity_1.hist()
plt.title('Distribution of humidity_1')
plt.xlabel('humidity_1')
plt.show()

df.pressure_1.hist()
plt.title('Distribution of pressure_1')
plt.xlabel('pressure_1')
plt.show()

# iterate over the feelslike columns
for feels_like_col in ['feels_like_1', 'feels_like_2', 'feels_like_3']:
    # create a boolean array of values representing nans
    missing_vals = pd.isnull(df[feels_like_col])
    df[feels_like_col][missing_vals] = 0

df = df.dropna()
