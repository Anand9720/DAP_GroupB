# DAP-Project
Project Name
In this Project we are doing Data Analysis on Entertainment Industries, where we are taking 3 datasets from different sites. 
First dataset we fetched by web Scraping from the IMDB website where we fetched 17 pages.
Second dataset is of Netflix which was in json format.
Third dataset is of Amazon Prime Movie data which was in json format.

After Loading Dataset we store data in Cloud version of MongDB i.e in MongoDb Atlas.
Then we fetched and did EDA (Exploratory Data analysis) and load data in POstgresSql and after that we perform Data Visualization on it.

Table of Contents
•	Features
•	Getting Started
•	Installation
•	Usage
•	Contributing
•	License
Features
We have,
•	Done WebScraping
•	EDA using regular expressions.
•	Used both NoSQL and SQL database like Mongodb Atlas and Postgresql
•	We have done visualization and found some meaningful insights.

Getting Started
Should have,
•	Python Notebook, 
•	account on Mongodb Atlas, 
•	and PgAdmin Installed
•	basic Knowledge of Python

Installation:
Install all Following Libraries before running the code:
•	import json

 For webscraping
•	import requests
•	from requests import get
•	from bs4 import BeautifulSoup

 To perform EDA and graphs
•	from warnings import warn
•	from time import sleep
•	from random import randint
•	import numpy as np
•	import pandas as pd
•	import seaborn as sns

For visualization
•	import matplotlib.pyplot as plt

 For mongodb
        
•	import csv
•	from pymongo import MongoClient

For PostgreSql
•	from sqlalchemy import create_engine
•	import psycopg2
** Note: keep password of your server 1234 **
Usage
Run all the cell in Jupiter Notebook.

Contribution:
Contact us on email: jhaanand9720@gmail.com, tyagik7@gmail.com, x22133275@student.ncirl.ie, niraj.nidan@gmail.com

License
GNU General Public License v3.0


