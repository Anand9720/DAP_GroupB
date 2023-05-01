import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import json
import re
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff


import matplotlib.pyplot as plt


with open('amazon_prime_titles_output.json', 'r') as file:
    data = json.load(file)

import csv
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://jhaanand9720:Parks321@cluster0.mhxavb3.mongodb.net/test')
db = client['DAP3_DataBase']
collection2 = db['Niraj_data']
# Open CSV file for reading


with open("C:\\Users\\WorkStation\\Desktop\\amazon_prime_titles_output.json") as file1:
    file_data1 = json.load(file1)

collection2.insert_many(file_data1)

d1 = db.Niraj_data.find({})
a=[]
for x in d1:
    a.append(x)

df = pd.DataFrame(a)

df

df = df.replace({'': np.nan, ' ': np.nan})

df

df_3 = df

df_3

df_3.shape

df_3.columns

df_3.info

df_3.describe()


df_3.isnull().sum().sort_values(ascending=False)

# As observed there are many missing values. We try to fill the missing values using various methods

df_3["director"].value_counts()

df_3.describe(include = object)

df_3['director'] = df_3['director'].replace(np.nan , "Unavailable")

df_3['cast'] = df_3['cast'].replace(np.nan , "Unavailable")

df_3['rating'] = df_3['rating'].fillna(df_3['rating'].mode()[0])

import random
country_names = ['Canada','UnitedStates','UnitedKingdom','india','France','Italy','Germany','Japan','Spain',]
df_3['country']=df_3['country'].replace(np.nan,random.choice(country_names))

df_3['date_added']= df_3['date_added'].ffill()

df_3.loc[13:20]

df_3['date_added']=pd.to_datetime(df_3['date_added'])

df_3.info()

# Split the duration column into duration_movies and duration_seasons

df_3[['duration_movies', 'duration_seasons']] = df_3['duration'].str.extract(r'(\d+) min|(\d+) Season')

 #Convert the data type of duration_movies and duration_seasons to numeric
df_3['duration_movies'] = pd.to_numeric(df_3['duration_movies'], errors='coerce')
df_3['duration_seasons'] = pd.to_numeric(df_3['duration_seasons'], errors='coerce')

print(df_3.head())

df_3.loc[df_3['release_year'].min() == df_3['release_year']]


recent_release = df_3.loc[df_3['release_year'].max() == df_3['release_year']]


recent_release.head()

sns_plot = sns.pairplot(df_3)
df_3.type.value_counts()

df_3.rating.value_counts()

sns.barplot(x = df_3.rating.value_counts(), y = df_3.rating.value_counts().index,data = df_3, orient = "h")
plt.savefig("Rating_bar_plot.png")
plt.close()

plt.figure(figsize = (12,10))
ax = sns.countplot(x="release_year", data = df_3, order = df_3.release_year.value_counts().index[0:15])
plt.savefig("Yearly_count.png")
plt.close()

# exclude 'Unavailable' from the analysis
df_no_unavailable = df_3[df_3['director'] != 'Unavailable']

# count the number of movies each director has directed
director_counts = df_no_unavailable['director'].value_counts()

# get the top 10 directors by number of movies
top_directors = director_counts.head(10)

# plot a bar chart to visualize the results
plt.figure(figsize=(20,10))
sns.barplot(x=top_directors.index, y=top_directors.values)
plt.title('Top 10 Directors by Number of Movies')
plt.xlabel('Director')
plt.ylabel('Number of Movies')
plt.savefig("Top_10_Director_By_Number_of_Movies.png")
plt.close()

# Convert the release_year column to datetime object
df_3['release_year'] = pd.to_datetime(df_3['release_year'], format='%Y')

# Group the data by year and count the number of titles released each year
df_year_count = df_3.groupby(df_3['release_year'].dt.year)['title'].count().reset_index()

# Plot the line chart
plt.figure(figsize=(12, 6))
sns.lineplot(x='release_year', y='title', data=df_year_count)
plt.xlabel('Year')
plt.ylabel('Number of titles')
plt.title('Number of titles released each year')
plt.savefig("Number_of_titles_released_each_year.png")
plt.close()
client.close()
