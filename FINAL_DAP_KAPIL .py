import json
import re
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from termcolor import colored
from plotly.offline import init_notebook_mode, iplot

import warnings

warnings.filterwarnings("ignore")


with open('netflix_output.json', 'r') as f:
    data = json.load(f)

data1 = data

import csv
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://jhaanand9720:Parks321@cluster0.mhxavb3.mongodb.net/test')
db = client['DAP3_DataBase']
collection1 = db['Kapil_data']
# Open CSV file for reading
with open('netflix_output.json') as file:
    file_data = json.load(file)

collection1.insert_many(file_data)

d1 = db.Kapil_data.find({})

a = []
for result in d1:
    a.append(result)

df_1 = pd.DataFrame(a)

print(df_1)
df_1=df_1.replace({'':np.nan,' ':np.nan})

df_1.drop(['director'],axis = 1,inplace = True)
df_1.head()

# Replace missing values in the 'cast' column with "No Cast"
df_1["cast"].replace(np.nan,"No Cast", inplace = True)

# Replace missing values in the 'country' column with "Unknown"
df_1["country"].replace(np.nan,"Unknown", inplace = True)

# Drop rows with missing values in the 'rating' and 'duration' columns
df_1.dropna(subset = ["rating","duration"], axis = 0, inplace = True)

# Drop rows with missing values in the 'date_added' column
df_1.dropna(subset = ["date_added"], axis = 0, inplace = True)

# checking null values
df_1.isnull().sum()

df_1.info()

from datetime import datetime

## these go on the numbers below
tl_dates = [
    "1997\nFounded",
    "1998\nMail Service",
    "2003\nGoes Public",
    "2007\nStreaming service",
    "2016\nIndia launch",
    "2021\nNetflix & Chill"
]

tl_x = [1, 2, 4, 5.3, 8, 9]

## these go on the numbers
tl_sub_x = [1.5, 3, 5, 6.5, 7]

tl_sub_times = [
    "1998", "2000", "2006", "2010", "2012"
]

tl_text = [
    "Netflix.com launched",
    "Starts\nPersonal\nRecommendations", "Billionth DVD Delivery", "Canadian\nLaunch", "UK Launch\n"]

# Set figure & Axes
fig, ax = plt.subplots(figsize=(15, 4), constrained_layout=True)
ax.set_ylim(-2, 1.75)
ax.set_xlim(0, 10)

# Timeline : line
ax.axhline(0, xmin=0.1, xmax=0.9, c='#4a4a4a', zorder=1)

# Timeline : Date Points
ax.scatter(tl_x, np.zeros(len(tl_x)), s=120, c='#4a4a4a', zorder=2)
ax.scatter(tl_x, np.zeros(len(tl_x)), s=30, c='#fafafa', zorder=3)
# Timeline : Time Points
ax.scatter(tl_sub_x, np.zeros(len(tl_sub_x)), s=50, c='#4a4a4a', zorder=4)

# Date Text
for x, date in zip(tl_x, tl_dates):
    ax.text(x, -0.55, date, ha='center',
            fontfamily='serif', fontweight='bold',
            color='#4a4a4a', fontsize=12)

# Stemplot : vertical line
levels = np.zeros(len(tl_sub_x))
levels[::2] = 0.3
levels[1::2] = -0.3
markerline, stemline, baseline = ax.stem(tl_sub_x, levels, use_line_collection=True)
plt.setp(baseline, zorder=0)
plt.setp(markerline, marker=',', color='#4a4a4a')
plt.setp(stemline, color='#4a4a4a')

# Text
for idx, x, time, txt in zip(range(1, len(tl_sub_x) + 1), tl_sub_x, tl_sub_times, tl_text):
    ax.text(x, 1.3 * (idx % 2) - 0.5, time, ha='center',
            fontfamily='serif', fontweight='bold',
            color='#4a4a4a' if idx != len(tl_sub_x) else '#b20710', fontsize=11)

    ax.text(x, 1.3 * (idx % 2) - 0.6, txt, va='top', ha='center',
            fontfamily='serif', color='#4a4a4a' if idx != len(tl_sub_x) else '#b20710')

# Spine
for spine in ["left", "top", "right", "bottom"]:
    ax.spines[spine].set_visible(False)

# Ticks
ax.set_xticks([])
ax.set_yticks([])

# Title
ax.set_title("Netflix through the years", fontweight="bold", fontfamily='serif', fontsize=16, color='#4a4a4a')
ax.text(2.4, 1.57, "From DVD rentals to a global audience of over 150m people - is it time for Netflix to Chill?",
        fontfamily='serif', fontsize=12, color='#4a4a4a')

plt.savefig("Netflix_through_the_years.png")
plt.close()


plt.figure(figsize=(10,5))
plt.pie(df_1['type'].value_counts().sort_values(),labels=df_1['type'].value_counts().index,explode=[0.05,0],
        autopct='%1.2f%%',colors=['Red','grey'])


plt.savefig("piechart.png")
plt.close()



#plt.show()

client.close()








