#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

    
    
analyze_data(df1)

