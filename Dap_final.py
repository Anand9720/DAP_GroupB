#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

from requests import get
from bs4 import BeautifulSoup
from warnings import warn
from time import sleep
from random import randint
import numpy as np
import pandas as pd
import seaborn as sns
import requests
from pymongo import MongoClient
import json
import matplotlib.pyplot as plt


# In[2]:


headers = {'Accept-Language': 'en-US,en;q=0.8'} 

titles = []
years = []
ratings = []
genres = []
runtimes = []
imdb_ratings = []
imdb_ratings_standardized = []
director = []
star=[]
revenue=[]
votes =[]


# In[3]:


pages = np.arange(1, 6000, 50)


# In[4]:


for page in pages:
    response = get("https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&sort=user_rating,desc&"
                  + "start="
                  + str(page)
                  + "&ref_=adv_nxt", headers=headers)
  
    sleep(randint(8,15))
    if response.status_code != 200:
       warn('Request: {}; Status code: {}'.format(requests, response.status_code))
    page_html = BeautifulSoup(response.text, 'html.parser')
      
    movie_containers = page_html.find_all('div', class_ = 'lister-item mode-advanced')
    
    for container in movie_containers:

        #title
        title = container.h3.a.text
        titles.append(title)

        if container.h3.find('span', class_= 'lister-item-year text-muted unbold') is not None:
                
            
               #year released
            year = container.h3.find('span', class_= 'lister-item-year text-muted unbold').text # remove the parentheses around the year and make it an integer
            years.append(year)

        else:
            years.append(None) # each of the additional if clauses are to handle type None data, replacing it with an empty string so the arrays are of the same length at the end of the scraping

        if container.p.find('span', class_ = 'certificate') is not None:
            
             #rating
            rating = container.p.find('span', class_= 'certificate').text
            ratings.append(rating)

        else:
            ratings.append("")

        if container.p.find('span', class_ = 'genre') is not None:
            
                #genre
            genre = container.p.find('span', class_ = 'genre').text #.replace("\n", "").rstrip().split(',') # remove the whitespace character, strip, and split to create an array of genres
            genres.append(genre)
          
        else:
            genres.append("")

        if container.p.find('span', class_ = 'runtime') is not None:

             #runtime
            time = int(container.p.find('span', class_ = 'runtime').text.replace(" min", "")) # remove the minute word from the runtime and make it an integer
            runtimes.append(time)

        else:
            runtimes.append(None)

        if float(container.strong.text) is not None:

             #IMDB ratings
            imdb = float(container.strong.text) # non-standardized variable
            imdb_ratings.append(imdb)

        else:
            imdb_ratings.append(None)
            
        a=container.find_all('p')[2].text
        b=a.split('|')[0].split(':')[1]
        director.append(b)
        star.append(a.split('|')[1].split(':')[1])
    
        b=container.find('p', {"class":"sort-num_votes-visible"}).text.strip()
    
        votes.append(b.split('|')[0].split(":")[1])
        if len(b.split('|'))>1:
            revenue.append(b.split('|')[1].split(":")[1])
        else:
            revenue.append(None)
        


# In[5]:


len(votes)


# # saving data to Mongo db 
# 

# In[6]:


data_set1={
    'Titles':titles,
    'Years': years,
    'Genres': genres,
    'Revenue':revenue,
    'Ratings': ratings,
    'Imdb_ratings': imdb_ratings,
    'Votes':votes,
    'Star': star,
    'Director':director,
    'Runtime':runtimes
}


# In[7]:


import csv
from pymongo import MongoClient
client = MongoClient('mongodb+srv://jhaanand9720:Parks321@cluster0.mhxavb3.mongodb.net/test')
db = client['DAP3_DataBase']
collection3=db['Anand_data']
collection3.insert_one(data_set1)


# In[8]:


client.list_database_names()


# In[9]:


d2=client.DAP3_DataBase


# In[10]:


d2


# In[11]:


d2.list_collection_names()


# In[12]:


d1=db.Anand_data.find_one({})


# In[13]:


list1=d1['Titles']
list2=d1['Years']
list3=d1['Genres']
list4=d1['Revenue']
list5=d1['Imdb_ratings']
list6=d1['Votes']
list7=d1['Director']
list8=d1['Star']
list10=d1['Runtime']
list9=d1['Ratings']


# In[14]:


df=pd.DataFrame(list(zip(list1,list2,list3,list4,list6,list7,list8,list9,list10,list5)),columns=['Titles','Year',"Genre",'Revenue(Millions)','Votes','Director','Star','Certificate','Runtime','Rating'])


# In[15]:


df.head()


# In[16]:


df.index = np.arange(1, len(df) + 1)


# # Explodatory Data Analysis

# In[17]:


df["Year"]= df['Year'].str.replace(r'\(|\)', '', regex=True)


# In[18]:


df['Votes'] = df['Votes'].str.replace(r'\n', '')
df['Votes'] = df['Votes'].str.replace(r',', '', regex=True)


# In[19]:


df['Revenue(Millions)'] = df['Revenue(Millions)'].str.replace(r'\n', '', regex=True)


# In[20]:


df['Star'] = df['Star'].str.replace(r'\n', '', regex=True)
df['Director'] = df['Director'].str.replace(r'\n', '', regex=True)
df['Revenue(Millions)'] = df['Revenue(Millions)'].str.replace(r'\$|M', '', regex=True)


# In[21]:


df['Genre'] = df['Genre'].str.replace(r'\n', '', regex=True)


# In[22]:


df


# In[23]:


print(df.dtypes)


# In[24]:


df["Year"]= df['Year'].str.replace(r'^\D*', '', regex=True)


# In[25]:


df['Year']=df['Year'].astype(int)


# In[26]:


df['Votes']=df['Votes'].astype(int)


# In[27]:


df['Revenue(Millions)']=df['Revenue(Millions)'].astype(float)


# In[28]:


df.to_csv('Imdb_dataset_+1.csv',index=False)


# In[ ]:





# In[29]:


print("Number of  Rows", df.shape[0])
print("Number of Columns", df.shape[1])


# In[30]:


df.info()


# In[ ]:





# In[ ]:


# save cleaned data into Postgresql


# In[31]:


from sqlalchemy import create_engine
import psycopg2


# In[32]:


pgconn=psycopg2.connect(
    host='localhost',
    user='postgres',
    port=5432,
    password='1234',
    database='postgres')


# In[33]:


pgcursor=pgconn.cursor()


# In[36]:


pgcursor.execute('CREATE DATABASE DAP_project_db')


# In[35]:


pgconn.commit()


# In[37]:


pgconn=psycopg2.connect(
    host='localhost',
    user='postgres',
    port=5432,
    password='1234',
    database='projec2_db')


# In[38]:


pgcursor=pgconn.cursor()


# In[39]:


from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
pgconn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)


# In[40]:


from sqlalchemy import create_engine


# In[41]:


engine=create_engine('postgresql+psycopg2://postgres:1234@localhost/projec_db')


# In[42]:


df.to_sql('my_table',con=engine,if_exists='replace',index=False)


# # EXTRACTING DATA FROM POSTGRESQL

# In[43]:


df1=pd.read_sql('my_table',engine)


# In[44]:


df1


# ### Find shape of our Dataset(Number of Rows and Columns)

# In[45]:


print("Number of Rows :",df1.shape[0])
print("Number of Columns :", df1.shape[1])


# ### Getting Information about our dataset like Total Number Rows, Total Number of columns, Datatypes of each Columns and Memory Requirement

# In[46]:


df1.info()


# ### Check Missing Values in the dataset?
# 

# In[47]:


print("Any missing value?", df1.isnull().values.any())


# In[48]:


df1.isnull().sum()


# In[49]:


sns.heatmap(df1.isnull())


# ### Drop all the missing values

# In[50]:


df1.dropna(axis=0,inplace=True)


# In[51]:


df1.isnull().sum()


# ### check for Duplicate data

# In[52]:


du_data=df1.duplicated().any()


# In[53]:


print(" is there any duplicated values in data? : ",du_data)  # there is no duplicated values


# ### Get over all statistic of dataset

# In[54]:


df1.describe()


# ### Display Title of Movie having Runtime >= 180 Minutes

# In[55]:


df1.columns


# In[56]:


df1[df1['Runtime']>=180]['Titles']


# ### In which year There was the highest Average voting?

# In[57]:


a=df1.groupby('Year')['Votes'].mean().sort_values(ascending=False)[:10]


# In[58]:


a


#   ### in which year we have highest revenue

# In[59]:


b=df1.groupby('Year')['Votes'].mean().sort_values(ascending=False)[:10]


# In[60]:


b


# ### Find the average Rating for Top 10 Director

# In[61]:


df1.groupby('Director')['Rating'].mean().sort_values(ascending=False)[:10]


# ### Number of Movies Per Year

# In[62]:


df1['Year'].value_counts()[:10]


# In[63]:


df1[df1['Revenue(Millions)'].max()==df1['Revenue(Millions)']]['Titles']


# ### Display Top 10 Highest rated Movie Titles and Director

# In[64]:


df1.columns


# In[65]:


top_rating= df1.nlargest(10,'Rating')[['Titles','Rating','Director']].set_index('Titles')


# In[66]:


top_rating


# In[67]:


pop= df1.sort_values('Rating', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['Titles'].head(6),pop['Rating'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("rating")
plt.title(" Title")


# In[68]:


df1['Rating'].plot(logy=True, kind='hist')


# In[69]:


df1['Revenue(Millions)'].plot(logy=True, kind='hist')


# In[70]:


plt.figure(figsize=(12,5))
sns.barplot(x='Genre', y='Revenue(Millions)', data=df1.iloc[1:11])
plt.xticks(rotation=90)
plt.show()


# ### Do Rating and vote average share a tangible relationship? In other words, is there a strong positive correlation between these two quanitties? Let us visualise their relationship in the form of a scatterplot.

# In[71]:


sns.jointplot(x='Votes', y='Rating', data=df1)


# ### Number of movies per Decade

# In[72]:




# create dataframe
df2 = pd.DataFrame(df1['Year'])

# define a function to get decade from year
def get_decade(year):
    return str(year)[2] + '0s'

# create new column based on decade
df2['decade'] = df1['Year'].apply(get_decade)

# display dataframe
print(df2)


# In[73]:


print(df2["decade"].unique())


# In[74]:


plt.figure(figsize=(12,6))
plt.title("Number of Movies released in a particular Decade.")
sns.countplot(x=df2["decade"])


# In[ ]:


### 


# In[75]:


plt.figure(figsize=(12,6))
plt.title("Average Gross by the Month for Blockbuster Movies")
sns.barplot(x=df2['decade'], y=df1['Revenue(Millions)'])


# In[76]:


year_count = df1.groupby('Year')['Titles'].count()
plt.figure(figsize=(18,5))
year_count.plot()


# ### Highest Grossing Films of All Time

# In[77]:


from IPython.display import Image, HTML
gross_top = df1[[ 'Titles', 'Revenue(Millions)', 'Year']].sort_values('Revenue(Millions)', ascending=False).head(10)
pd.set_option('display.max_colwidth', 100)
HTML(gross_top.to_html(escape=False))


# In[78]:


plt.figure(figsize=(18,5))
year_revenue = df1[(df1['Revenue(Millions)'].notnull()) & (df1['Year'] != 'NaT')].groupby('Year')['Revenue(Millions)'].max()
plt.plot(year_revenue.index, year_revenue)
plt.xticks(np.arange(1920, 2024, 10.0))
plt.show()


# In[79]:


sns.set(font_scale=1)
corr = df1.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    plt.figure(figsize=(9,9))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True)


# In[80]:


pop_gen = pd.DataFrame(df1['Genre'].value_counts()).reset_index()
pop_gen.columns = ['Genre', 'Movies_count']
pop_gen.head(10)


# ### which genere combination has highest Movie count

# In[81]:


plt.figure(figsize=(18,8))
sns.barplot(x='Genre', y='Movies_count', data=pop_gen.head(15))
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# In[82]:


plt.title('Directors with the Highest Total Revenue')
df1.groupby('Director')['Revenue(Millions)'].sum().sort_values(ascending=False).head(10).plot(kind='bar', colormap='copper_r')
plt.show()


# In[83]:


df1.Director.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))
plt.title('TOP 10 DIRECTORS OF MOVIES')
plt.show()


# In[ ]:




