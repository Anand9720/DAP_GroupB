#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# Initialize plotly notebook mode
# init_notebook_mode(connected=True)


with open('netflix_output.json', 'r') as f:
    data = json.load(f)

data1=data



import csv
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://jhaanand9720:Parks321@cluster0.mhxavb3.mongodb.net/test')
db = client['DAP3_DataBase']
collection1 = db['Kapil_data']
# Open CSV file for reading
with open('/Users/kapiltyagi/Downloads/netflix_output.json') as file:
    file_data = json.load(file)
    
collection1.insert_many(file_data)



d1 = db.Kapil_data.find({})

a = []
for result in d1:
    a.append(result)

df_1= pd.DataFrame(a)

df_1


# In[2]:


df_1=df_1.replace({'':np.nan,' ':np.nan})


# In[3]:


df_1


# In[4]:


# Handling missing values
#I choose to drop the 'director'  columns completely as they have high volume of missing values and dropping thiscolumns will not effect my visualization.


# In[5]:


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


# In[6]:


df_1.info()


# In[7]:


# Convert the 'date_added' column to datetime format
df_1["date_added"] = pd.to_datetime(df_1['date_added'])

# Extract the month and year from the 'date_added' column and create new columns for them
df_1['month_added']=df_1['date_added'].dt.month
df_1['month_name_added']=df_1['date_added'].dt.month_name()
df_1['year_added'] = df_1['date_added'].dt.year

# Drop the 'date_added' column since we no longer need it
df_1.drop('date_added',axis=1,inplace=True)


# In[8]:


df_1.head()


# In[ ]:





# In[9]:



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

tl_x = [1, 2, 4, 5.3, 8,9]

## these go on the numbers
tl_sub_x = [1.5,3,5,6.5,7]


tl_sub_times = [
    "1998","2000","2006","2010","2012"
]

tl_text = [
    "Netflix.com launched",
    "Starts\nPersonal\nRecommendations","Billionth DVD Delivery","Canadian\nLaunch","UK Launch\n"]



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
ax.scatter(tl_sub_x, np.zeros(len(tl_sub_x)), s=50, c='#4a4a4a',zorder=4)

# Date Text
for x, date in zip(tl_x, tl_dates):
    ax.text(x, -0.55, date, ha='center', 
            fontfamily='serif', fontweight='bold',
            color='#4a4a4a',fontsize=12)
    

# Stemplot : vertical line
levels = np.zeros(len(tl_sub_x))    
levels[::2] = 0.3
levels[1::2] = -0.3
markerline, stemline, baseline = ax.stem(tl_sub_x, levels, use_line_collection=True)    
plt.setp(baseline, zorder=0)
plt.setp(markerline, marker=',', color='#4a4a4a')
plt.setp(stemline, color='#4a4a4a')

# Text
for idx, x, time, txt in zip(range(1, len(tl_sub_x)+1), tl_sub_x, tl_sub_times, tl_text):
    ax.text(x, 1.3*(idx%2)-0.5, time, ha='center', 
            fontfamily='serif', fontweight='bold',
            color='#4a4a4a' if idx!=len(tl_sub_x) else '#b20710', fontsize=11)
    
    ax.text(x, 1.3*(idx%2)-0.6, txt, va='top', ha='center', 
        fontfamily='serif',color='#4a4a4a' if idx!=len(tl_sub_x) else '#b20710')



# Spine
for spine in ["left", "top", "right", "bottom"]:
    ax.spines[spine].set_visible(False)

# Ticks    
ax.set_xticks([]) 
ax.set_yticks([]) 

# Title
ax.set_title("Netflix through the years", fontweight="bold", fontfamily='serif', fontsize=16, color='#4a4a4a')
ax.text(2.4,1.57,"From DVD rentals to a global audience of over 150m people - is it time for Netflix to Chill?", fontfamily='serif', fontsize=12, color='#4a4a4a')

plt.show()


# In[10]:


plt.figure(figsize=(10,5))
plt.pie(df_1['type'].value_counts().sort_values(),labels=df_1['type'].value_counts().index,explode=[0.05,0],
        autopct='%1.2f%%',colors=['Red','grey'])
plt.savefig("Movie_and_Tv_show.png")
plt.close()
#plt.show()


# In[11]:


cast_shows = df_1[df_1.cast != "No cast"].set_index('title').cast.str.split(',', expand = True).stack().reset_index(level = 1,drop = True)
plt.figure(figsize = (13,7))
plt.title('Top 10 Movie actors based on the no. of tiltes')
sns.countplot(y = cast_shows, order= cast_shows.value_counts().index[:10])

plt.savefig("Top 10 Movie actors based on the no. of tiltes.png")
plt.close()

#plt.show()


# In[12]:


from collections import Counter
country_data = df_1['country']
country_count = pd.Series(dict(Counter(','.join(country_data).replace(' ,',',').replace(
    ', ',',').split(',')))).sort_values(ascending=False)


# In[13]:


top20country = country_count.head(20)


# In[14]:


from matplotlib import gridspec

fig = plt.figure(figsize=(20, 7))
gs = gridspec.GridSpec(nrows=1, ncols=2, height_ratios=[6], width_ratios=[10, 5])

ax = plt.subplot(gs[0])
sns.barplot(top20country.index, top20country, ax=ax, palette="RdGy")
ax.set_xticklabels(top20country.index, rotation='90')
ax.set_title('Top 20 countries with most contents', fontsize=15, fontweight='bold')

ax2 = plt.subplot(gs[1])
ax2.pie(top20country, labels=top20country.index, shadow=True, startangle=0, colors=sns.color_palette("RdGy", n_colors=20),
       autopct='%1.2f%%')
ax2.axis('equal') 
plt.savefig("Top 20 countries with most contents.png")
plt.close()
#plt.show()


# We can see that US, India, United Kingdom,Unknown, Canada and France contribute 76% of the top20 countries.

# In[15]:


country = df_1.set_index('title').country.str.split(', ', expand=True).stack().reset_index(level=1, drop=True);
country = country[country != 'unknown']

iplot([go.Choropleth(
locationmode='country names',
locations=country,
z=country.value_counts())])


# # Content added over the years
# 
# 

# In[16]:


df_1_tv = df_1[df_1["type"] == "TV Show"]
df_1_movies = df_1[df_1["type"] == "Movie"]


# In[17]:


df_1_content = df_1['year_added'].value_counts().reset_index().rename(columns = {
    'year_added' : 'count', 'index' : 'year_added'}).sort_values('year_added')
df_1_content['percent'] = df_1_content['count'].apply(lambda x : 100*x/sum(df_1_content['count']))


df_1_tv1 = df_1_tv['year_added'].value_counts().reset_index().rename(columns = {
    'year_added' : 'count', 'index' : 'year_added'}).sort_values('year_added')
df_1_tv1['percent'] = df_1_tv1['count'].apply(lambda x : 100*x/sum(df_1_tv1['count']))


df_1_movies1 = df_1_movies['year_added'].value_counts().reset_index().rename(columns = {
    'year_added' : 'count', 'index' : 'year_added'}).sort_values('year_added')
df_1_movies1['percent'] = df_1_movies1['count'].apply(lambda x : 100*x/sum(df_1_movies1['count']))

t1 = go.Scatter(x=df_1_movies1['year_added'], y=df_1_movies1["count"], name="Movies", marker=dict(color="#a678de"))
t2 = go.Scatter(x=df_1_tv1['year_added'], y=df_1_tv1["count"], name="TV Shows", marker=dict(color="#6ad49b"))
t3 = go.Scatter(x=df_1_content['year_added'], y=df_1_content["count"], name="Total Contents", marker=dict(color="brown"))

data = [t1, t2, t3]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()


# The growth in number of movies on netflix is much higher than that of TV shows About 1200 new movies were added in both 2018 and 2019 The growth in content started from 2013

# # Content added over the month

# In[18]:


df_1_content = df_1[['month_added','month_name_added']].value_counts().reset_index().rename(columns = {
    0 : 'count'}).sort_values('month_added').drop('month_added',axis=1)
df_1_content['percent'] = df_1_content['count'].apply(lambda x : 100*x/sum(df_1_content['count']))


df_1_tv2 = df_1_tv[['month_added','month_name_added']].value_counts().reset_index().rename(columns = {
    0 : 'count'}).sort_values('month_added').drop('month_added',axis=1)
df_1_tv2['percent'] = df_1_tv2['count'].apply(lambda x : 100*x/sum(df_1_tv2['count']))


df_1_movies2 = df_1_movies[['month_added','month_name_added']].value_counts().reset_index().rename(columns = {
    0 : 'count'}).sort_values('month_added').drop('month_added',axis=1)
df_1_movies2['percent'] = df_1_movies2['count'].apply(lambda x : 100*x/sum(df_1_movies2['count']))

t1 = go.Scatter(x=df_1_movies2['month_name_added'], y=df_1_movies2["count"], name="Movies", marker=dict(color="#a678de"))
t2 = go.Scatter(x=df_1_tv2['month_name_added'], y=df_1_tv2["count"], name="TV Shows", marker=dict(color="#6ad49b"))
t3 = go.Scatter(x=df_1_content['month_name_added'], y=df_1_content["count"], name="Total Contents", marker=dict(color="Brown"))


data = [t1, t2, t3]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()


# * The growth in contents are higher in the first three months and the last three months of the year.
# * Least number of contents are added in the month of February.

# # Genre Relationship

# For Movies

# In[19]:


from sklearn.preprocessing import MultiLabelBinarizer 

def relation_heatmap(df_1, title):
    df_1['genre'] = df_1['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 
    Types = []
    for i in df_1['genre']: Types += i
    Types = set(Types)
    print("There are {} types in the Netflix {} Dataset".format(len(Types),title))    
    test = df_1['genre']
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)
    corr = res.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(10, 7))
    pl = sns.heatmap(corr, mask=mask, cmap= "coolwarm", vmax=.5, vmin=-.5, center=0, square=True, linewidths=.7,
                     cbar_kws={"shrink": 0.6})
    plt.savefig("relation_heatmap.png")
    plt.close()
    #plt.show()


# In[20]:


relation_heatmap(df_1_movies, 'Movie')


# In[21]:


relation_heatmap(df_1_tv, 'TV Show')


# * TV shows are more clearly correlated than movies.
# * The negative relationship between kid's TV and International Tv Shows is remarkable.
# * There is a strong positive corelation between Science & Natural and Docuseries.

# In[22]:



top_genres = df_1.loc[df_1['listed_in'].isin(df_1.groupby('listed_in').sum().sort_values(by='listed_in', ascending=False).reset_index()['listed_in'][:10])].groupby('listed_in').mean().sort_values(by='listed_in', ascending=False).reset_index()['listed_in']
plt.figure(figsize= (30, 10))

sns.countplot(df_1.loc[df_1['listed_in'].isin(top_genres)]['release_year'], 
              hue= df_1.loc[df_1['listed_in'].isin(top_genres)]['listed_in'])

plt.title('Released Genre per Year', size= 25)
plt.xlabel(None)
plt.xticks(size= 16)
plt.savefig("Released Genre per Year.png")
plt.close()
plt.show()


# # Rating Distribuiton

# In[23]:


df_1.rating.value_counts()


# In[24]:


sns.barplot(x = df_1.rating.value_counts(), y = df_1.rating.value_counts().index,data = df_1, orient = "h")
plt.savefig("No. of each rating.png")
plt.close()
#plt.show()


# In[25]:


rating_order_movie =  ['TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG', 'PG-13', 'TV-14', 'R', 'NC-17', 'TV-MA']
rating_order_tv =  ['TV-Y', 'TV-G', 'TV-Y7', 'TV-Y7-FV', 'TV-PG', 'TV-14', 'R', 'TV-MA']


# In[26]:


movie_rating = df_1_movies['rating'].value_counts()[rating_order_movie]
tv_rating = df_1_tv['rating'].value_counts()[rating_order_tv].fillna(0)


# In[27]:


def rating_barplot(data, title, height, h_lim=None):
    fig, ax = plt.subplots(1,1, figsize=(15, 7))
    if h_lim :
        ax.set_ylim(0, h_lim)
    ax.bar(data.index, data,  color="#d0d0d0", width=0.6, edgecolor='black')

    color =  ['green',  'blue',  'orange',  'red']
    span_range = [[0, 2], [3,  6], [7, 8], [9, 11]]

    for idx, sub_title in enumerate(['Little Kids', 'Older Kids', 'Teens', 'Mature']):
        ax.annotate(sub_title,
                    xy=(sum(span_range[idx])/2 ,height),
                    xytext=(0,0), textcoords='offset points',
                    va="center", ha="center",
                    color="w", fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round4', pad=0.4, color=color[idx], alpha=0.6))
        ax.axvspan(span_range[idx][0]-0.4,span_range[idx][1]+0.4,  color=color[idx], alpha=0.1)

    ax.set_title(f'Distribution of {title} Rating', fontsize=20, fontweight='bold', position=(0.5, 1.0+0.03))
    plt.savefig(f'Distribution of {title} Rating.png')
    plt.close()
    
    #plt.show()


# # For Movies

# In[28]:


rating_barplot(movie_rating,'Movie', 1200)


# # for TV

# In[29]:


rating_barplot(tv_rating,'TV Show' , 700, 800)


# From the plots we can say that the TV Show are more for young children.

# # Distribuiton of movie duration

# In[30]:


from scipy.stats import norm

plt.figure(figsize=(15,7))
sns.distplot(df_1_movies['duration'].str.extract('(\d+)'),fit=norm,kde=False,color=['red'])
plt.title('Distplot with Normal distribution for Movies',fontweight="bold")
plt.savefig('Distplot with Normal distribution for Movies.png')
plt.close()
#plt.show()


# Form the above plot we can say that majority of the movies have duration ranging from 85 min to 120 min.

# # Distribution of Tv show

# In[31]:


plt.figure(figsize=(15,7))
ax = sns.countplot(df_1_tv['duration'],order = df_1_tv['duration'].value_counts().index,palette="RdGy")
plt.title('Countplot for Seasons in TV_Shows',fontweight="bold")
plt.xticks(rotation=90)
for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, (p.get_height() * 1.005)))

plt.figure(figsize=(15,7))
ax = sns.barplot(x=((df_1_tv['duration'].value_counts()/df_1_tv.shape[0])*100).index,
                 y=round(((df_1_tv['duration'].value_counts()/df_1_tv.shape[0])*100),2).values,
                 palette="RdGy")
plt.title('Percentage of Seasons in TV_Shows',fontweight="bold")
plt.xticks(rotation=90)
for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, (p.get_height() * 1.005)))
plt.savefig('Countplot for Seasons in TV_Shows.png')
plt.close()
#plt.show()


# From the above plots we can say that 90% of the TV_Shows end by at most Season 3.

# # Top 10 Genre in movie & Tv shows

# In[32]:


plt.figure(figsize=(15,5))
sns.barplot(x = df_1_movies["listed_in"].value_counts().head(10).index,
            y = df_1_movies["listed_in"].value_counts().head(10).values,palette="RdGy")
plt.xticks(rotation=80)
plt.title("Top10 Genre in Movies",fontweight="bold")

plt.savefig('Top 10 genre in Movies.png')
plt.close()
#plt.show()


# In[33]:


plt.figure(figsize=(15,5))
sns.barplot(x = df_1_tv["listed_in"].value_counts().head(10).index,
            y = df_1_tv["listed_in"].value_counts().head(10).values,palette="RdGy")
plt.xticks(rotation=80)
plt.title("Top10 Genre in TV Shows",fontweight="bold")
plt.savefig('Top 10 genre in TV_Shows.png')
plt.close()
#plt.show()


# # Top 20 artists present on Netflix

# In[34]:


df_1['cast_name'] = df_1['cast'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 
cast_count = []
for i in df_1['cast_name']: cast_count += i
    
cast_dict = dict((i, cast_count.count(i)) for i in cast_count)

df_1_cast_count = pd.DataFrame(cast_dict.values(),cast_dict.keys()).reset_index().sort_values(0,ascending=False).rename(
    columns = {'index' : 'cast_name', 0 : 'count'}).iloc[1:21]


# In[35]:


plt.figure(figsize=(15,5))
sns.barplot(x='cast_name',y='count',data=df_1_cast_count,palette="RdGy")
plt.title("Top20 Artist on Netflix",fontweight="bold")
plt.xticks(rotation=90)
plt.savefig('Top20 Artist on Netflix.png')
plt.close()
#plt.show()


# Top five artist's are from India

# In[ ]:


client.close()


# In[ ]:





# In[ ]:




