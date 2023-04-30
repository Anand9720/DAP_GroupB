#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from requests import get
from bs4 import BeautifulSoup
from time import sleep
from random import randint
import numpy as np

headers = {'Accept-Language': 'en-US,en;q=0.8'}

# here we have created empty list for the columns name
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
pages = np.arange(1, 6000, 50)
try:
    for page in pages:
        response = get("https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&sort=user_rating,desc&"
                      + "start="
                      + str(page)
                      + "&ref_=adv_nxt", headers=headers)

        #sleep(randint(8,15))
        response.raise_for_status()

        page_html = BeautifulSoup(response.text, 'html.parser')
        movie_containers = page_html.find_all('div', class_ = 'lister-item mode-advanced')

        for container in movie_containers:
            try:
                #title
                title = container.h3.a.text
                titles.append(title)

                if container.h3.find('span', class_= 'lister-item-year text-muted unbold') is not None:
                    #year released
                    year = container.h3.find('span', class_= 'lister-item-year text-muted unbold').text
                    years.append(year)
                else:
                    years.append(None)

                if container.p.find('span', class_ = 'certificate') is not None:
                    #rating
                    rating = container.p.find('span', class_= 'certificate').text
                    ratings.append(rating)
                else:
                    ratings.append("")

                if container.p.find('span', class_ = 'genre') is not None:
                    #genre
                    genre = container.p.find('span', class_= 'genre').text
                    genres.append(genre)
                else:
                    genres.append("")

                if container.p.find('span', class_ = 'runtime') is not None:
                    #runtime
                    time = int(container.p.find('span', class_= 'runtime').text.replace(" min", ""))
                    runtimes.append(time)
                else:
                    runtimes.append(None)

                if float(container.strong.text) is not None:
                    #IMDB ratings
                    imdb = float(container.strong.text)
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
            except Exception as e:
                print("Error occurred while scraping data from container: ", e)
except Exception as e:
    print("Error occurred while scraping data from page: ", e)


# In[2]:


print(len(votes))


# In[3]:


import csv
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
try:
    client = MongoClient('mongodb+srv://jhaanand9720:Parks321@cluster0.mhxavb3.mongodb.net/test')
    db = client['DAP3_DataBase']
    collection3=db['Anand_data']
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

# Insert data into MongoDB
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
try:
    collection3.insert_one(data_set1)
    print("Data inserted into MongoDB")
except Exception as e:
    print(f"Error inserting data into MongoDB: {e}")

# Retrieve data from MongoDB
try:
    d1 = db.Anand_data.find_one({})
    list1 = d1['Titles']
    list2 = d1['Years']
    list3 = d1['Genres']
    list4 = d1['Revenue']
    list5 = d1['Imdb_ratings']
    list6 = d1['Votes']
    list7 = d1['Director']
    list8 = d1['Star']
    list10 = d1['Runtime']
    list9 = d1['Ratings']

    df=pd.DataFrame(list(zip(list1,list2,list3,list4,list6,list7,list8,list9,list10,list5)),columns=['Titles','Year',"Genre",'Revenue(Millions)','Votes','Director','Star','Certificate','Runtime','Rating'])
    print(df.head())
except Exception as e:
    print(f"Error retrieving data from MongoDB: {e}")


# In[ ]:





# In[4]:


# Replace newline and comma characters in the 'Votes' column
df['Votes'] = df['Votes'].str.replace(r'\n', '')
df['Votes'] = df['Votes'].str.replace(r',', '', regex=True)

# Remove parentheses characters from the 'Year' column
df["Year"]= df['Year'].str.replace(r'\(|\)', '', regex=True)

# Remove newline characters from the 'Revenue(Millions)' column
df['Revenue(Millions)'] = df['Revenue(Millions)'].str.replace(r'\n', '', regex=True)

# Remove newline characters from the 'Star' and 'Director' columns
df['Star'] = df['Star'].str.replace(r'\n', '', regex=True)
df['Director'] = df['Director'].str.replace(r'\n', '', regex=True)

# Remove dollar sign and 'M' character from the 'Revenue(Millions)' column
df['Revenue(Millions)'] = df['Revenue(Millions)'].str.replace(r'\$|M', '', regex=True)

# Remove newline characters from the 'Genre' column
df['Genre'] = df['Genre'].str.replace(r'\n', '', regex=True)

# Display the modified DataFrame's data types
print(df.dtypes)

# Remove non-digit characters from the beginning of the 'Year' column
df["Year"]= df['Year'].str.replace(r'^\D*', '', regex=True)

# Convert the 'Year' column to integer data type
df['Year']=df['Year'].astype(int)

# Convert the 'Votes' column to integer data type
df['Votes']=df['Votes'].astype(int)

# Convert the 'Revenue(Millions)' column to float data type
df['Revenue(Millions)']=df['Revenue(Millions)'].astype(float)

# Write the modified DataFrame to a new CSV file
df.to_csv('Imdb_dataset_+1.csv',index=False)

# Display the number of rows and columns in the modified DataFrame
print("Number of  Rows", df.shape[0])
print("Number of Columns", df.shape[1])

# Display information about the modified DataFrame's data types and memory usage
df.info()


# In[16]:


# import necessary libraries
from sqlalchemy import create_engine
import psycopg2

try:
    # connect to the PostgreSQL server
    pgconn = psycopg2.connect(
        host='localhost',
        user='postgres',
        port=5432,
        password='1234',
        database='postgres'
    )

    # set isolation level to autocommit
    #pgconn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    # create a cursor object to execute PostgreSQL commands
    pgcursor = pgconn.cursor()

    # check if the database exists
    pgcursor.execute('SELECT 1 FROM pg_database WHERE datname=\'dap_project_db\'')
    exists = pgcursor.fetchone()

    # create the database if it does not exist
    if not exists:
        pgcursor.execute('CREATE DATABASE dap_project_db')
        pgconn.commit()

    # connect to the newly created database
    pgconn = psycopg2.connect(
        host='localhost',
        user='postgres',
        port=5432,
        password='1234',
        database='dap_project_db'
    )
    pgcursor = pgconn.cursor()

    # create a SQLAlchemy engine object to interact with the database
    engine = create_engine('postgresql+psycopg2://postgres:1234@localhost/projec_db')

    # write dataframe to the database table
    df.to_sql('my_table', con=engine, if_exists='replace', index=False)

    # read the table back from the database into a new dataframe
    df1 = pd.read_sql('my_table', engine)
    print("Number of Rows :", df1.shape[0])
    print("Number of Columns :", df1.shape[1])

except psycopg2.Error as e:
    # handle PostgreSQL errors
    print(f"An error occurred: {e}")

except Exception as e:
    # handle other unexpected errors
    print(f"An unexpected error occurred: {e}")

finally:
    # close the cursor and database connections
    if pgcursor is not None:
        pgcursor.close()
    if pgconn is not None:
        pgconn.close()


# In[17]:


import pandas as pd
import seaborn as sns

def analyze_data(df):
    # Print basic info about the DataFrame
    print(df.info())
    
    # Check if there are any missing values and print result
    print("Any missing value?", df.isnull().values.any())

    # Print the count of missing values per column
    print(df.isnull().sum())

    # Plot a heatmap to visualize the missing values
    sns.heatmap(df.isnull())

    # Drop rows with missing values
    df.dropna(axis=0, inplace=True)

    # Print the count of missing values after dropping rows
    print(df.isnull().sum())

    # Check if there are any duplicated values and print result
    du_data = df.duplicated().any()
    print("Is there any duplicated values in data?: ", du_data) 

    # Print summary statistics about the DataFrame
    print(df.describe())


# In[8]:


analyze_data(df1)


# In[18]:


print(df1.isnull().sum())


# In[10]:


print(df1.describe())


# In[73]:


# Drop non-numeric columns
df1_numeric = df1.select_dtypes(include=[np.number])

# Calculate correlation matrix
corr = df1_numeric.corr()

# Plot heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
# In[19]:


import matplotlib.pyplot as plt

def plot_top_revenue_pie_chart(df):
    # Get the top 10 revenue values
    revenue_data = df.groupby('Year')['Revenue(Millions)'].mean().sort_values(ascending=False)[:10]

    # Create a pie chart
    plt.pie(revenue_data, labels=revenue_data.index, autopct='%1.1f%%', startangle=90)
    plt.title('Top 10 Yearly Mean Revenues')
    plt.axis('equal')

    # Show the chart
    plt.show()

plot_top_revenue_pie_chart(df1)


# In[75]:


import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_yearly_revenues(df, n=10):
    # Group by Year and calculate mean Revenue, and sort in descending order
    yearly_revenues = df.groupby('Year')['Revenue(Millions)'].mean().sort_values(ascending=False)[:n]

    # Create a horizontal bar plot
    plt.figure(figsize=(10,6))
    colors = sns.color_palette("rocket_r", len(yearly_revenues))
    sns.barplot(x=yearly_revenues.index, y=yearly_revenues.values, palette=colors)

    # Set axis labels and title
    plt.xlabel('Year')
    plt.ylabel('Revenue (Millions)')
    plt.title(f'Top {n} Years with Highest Average Revenue')
    plt.xticks(rotation=90)
    # Show the plot
    plt.show()
plot_top_yearly_revenues(df1, 20)


# In[25]:


def plot_top_yearly_movies(df, n=20):
    # Group by Year and calculate mean Revenue, and sort in descending order
    yearly_movie_count = df1['Year'].value_counts()[:n]
    # Create a horizontal bar plot
    plt.figure(figsize=(10,6))
    colors = sns.color_palette("rocket_r", len(yearly_movie_count))
    sns.barplot(x=yearly_movie_count.index, y=yearly_movie_count.values, palette=colors)

    # Set axis labels and title
    plt.xlabel('Year')
    plt.ylabel('Movie_count')
    plt.title(f'Top {n} Years with Highest Movie released')
    plt.xticks(rotation=90)
    # Show the plot
    plt.show()
    
plot_top_yearly_movies(df1)


# In[32]:


top_rating=pd.DataFrame(df1.nlargest(10,'Rating')[['Titles','Rating','Director']].set_index('Titles'))
top_rating


# In[33]:


df1['Rating'].plot(logy=True, kind='hist')


# In[34]:


df1['Revenue(Millions)'].plot(logy=True, kind='hist')


# In[76]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_genres(df, column, n=15):
    pop_gen = pd.DataFrame(df[column].value_counts()).reset_index()
    pop_gen.columns = [column, 'Movies_count']
    plt.figure(figsize=(18,8))
    sns.barplot(x=column, y='Movies_count', data=pop_gen.head(n))
    plt.xticks(rotation=90)
    plt.show()
    
# example usage
plot_top_genres(df1, 'Genre')


# In[77]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_genre_revenue(df, genre_col, revenue_col):
    top_genres = df.groupby(genre_col)[revenue_col].mean().sort_values(ascending=False)[:10]
    top_genres_df = pd.DataFrame(top_genres).reset_index()
    plt.figure(figsize=(12,5))
    sns.barplot(x=genre_col, y=revenue_col, data=top_genres_df)
    plt.xticks(rotation=90)
    plt.xlabel('Genre')
    plt.ylabel('Revenue (Millions)')
    plt.title('Top 10 Genres by Revenue')
    plt.show()
plot_genre_revenue(df1, 'Genre', 'Revenue(Millions)')


# In[38]:


# create dataframe
df2 = pd.DataFrame(df1['Year'])

# define a function to get decade from year
def get_decade(year):
    return (year // 10) * 10

# create new column based on decade
df2['decade'] = df1['Year'].apply(get_decade)

# display dataframe
print(df2)


# In[39]:


print(df2["decade"].unique())


# In[40]:


plt.figure(figsize=(12,6))
plt.title("Number of Movies released in a particular Decade.")
sns.countplot(x=df2["decade"])


# In[41]:


plt.figure(figsize=(12,6))
plt.title("Average Gross by the Month for Blockbuster Movies")
sns.barplot(x=df2['decade'], y=df1['Revenue(Millions)'])


# In[44]:


year_count = df1.groupby('Year')['Titles'].count()
plt.figure(figsize=(18,5))
year_count.plot()


# In[55]:


get_ipython().system('pip install itertools')


# In[79]:


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import  numpy as np
from itertools import count
import pandas as pd

df3 = pd.DataFrame(year_count.index)
df3['Movie_Count']=year_count.values
x=[]
y=[]
fig,ax=plt.subplots()
ax.plot(year_count.index,year_count.values)
counter=count(0,1)
def update(i):
    idx=next(counter)
    x.append(df3.iloc[idx,0])
    y.append(df3.iloc[idx,1])
    plt.cla()
    ax.plot(x,y)

ani=FuncAnimation(fig=fig,func=update,interval=9000000000)
plt.show()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import  numpy as np
from itertools import count
import pandas as pd

df3 = pd.DataFrame(year_count.index)
df3['Movie_Count']=year_count.values


# In[78]:


def plot_top_directors(df):
    plt.title('Directors with the Highest Total Revenue')
    df.groupby('Director')['Revenue(Millions)'].sum().sort_values(ascending=False).head(10).plot(kind='bar', colormap='copper_r')
    plt.show()
    
plot_top_directors(df1)


# In[ ]:




