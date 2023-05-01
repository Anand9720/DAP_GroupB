#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


d1=pd.read_csv("Imdb_dataset_+1.csv")


# In[3]:


d1.info()


# In[4]:


d2=pd.read_csv("Netflix cleaner data.csv")


# In[5]:


d2.info()


# In[6]:


d1.rename(columns={'Titles': 'title'}, inplace=True)


# In[7]:


key_column = 'title'
d1[key_column] = d1[key_column].astype(str)
d2[key_column] = d2[key_column].astype(str)


# In[8]:


d1 = d1.sort_values(key_column)
d2 = d2.sort_values(key_column)


# In[9]:



merged_data = pd.merge(d1, d2, on=key_column)


# In[10]:


merged_data.info()


# In[11]:


import pandas as pd

# Load the JSON data into a pandas dataframe
with open('amazon_prime_titles_output.json', 'r') as f:
    data = pd.read_json(f)

# Convert the dataframe to CSV format
data.to_csv('data11.csv', index=False)


# In[12]:


d3=pd.read_csv("data11.csv")


# In[13]:


d3.info()


# In[14]:


key_column = 'title'
d1[key_column] = d1[key_column].astype(str)
d3[key_column] = d3[key_column].astype(str)


# In[15]:


merged_data2 = pd.merge(d1, d3, on=key_column)


# In[16]:


merged_data2.info()


# In[17]:


merged_data


# In[43]:


top_rating_by_netflix= merged_data.nlargest(10,'Rating')[['title','Rating','Director']].set_index('title')


# In[44]:


top_rating_by_netflix


# In[40]:


Highest_revenue_by_netflix= merged_data.nlargest(10,'Revenue(Millions)')[['title','Revenue(Millions)','Director']].set_index('title')


# In[41]:


Highest_revenue_by_netflix


# In[42]:


def plot_top_yearly_revenues_by_netflix(df, n=10):
    # Group by Year and calculate mean Revenue, and sort in descending order
    yearly_revenues = merged_data.groupby('Year')['Revenue(Millions)'].mean().sort_values(ascending=False)[:n]

    # Create a horizontal bar plot
    plt.figure(figsize=(10,6))
    colors = sns.color_palette("rocket_r", len(yearly_revenues))
    sns.barplot(x=yearly_revenues.index, y=yearly_revenues.values, palette=colors)

    # Set axis labels and title
    plt.xlabel('Year')
    plt.ylabel('Revenue (Millions)')
    plt.title(f'Top {n} Years with Highest Average Revenue by netflix')
    plt.xticks(rotation=90)
    # Show the plot
    plt.savefig("plot_top_yearly_revenues_by_netflix.png")
    plt.close()
    #plt.show()
plot_top_yearly_revenues_by_netflix(merged_data, 20)


# In[45]:


top_rating_by_amazon= merged_data2.nlargest(10,'Rating')[['title','Rating','Director']].set_index('title')


# In[46]:


top_rating_by_amazon


# In[47]:


Highest_revenue_by_amazon= merged_data2.nlargest(10,'Revenue(Millions)')[['title','Revenue(Millions)','Director']].set_index('title')


# In[48]:


Highest_revenue_by_amazon


# In[49]:


def plot_top_yearly_revenues_by_amazon(df, n=10):
    # Group by Year and calculate mean Revenue, and sort in descending order
    yearly_revenues = merged_data2.groupby('Year')['Revenue(Millions)'].mean().sort_values(ascending=False)[:n]

    # Create a horizontal bar plot
    plt.figure(figsize=(10,6))
    colors = sns.color_palette("rocket_r", len(yearly_revenues))
    sns.barplot(x=yearly_revenues.index, y=yearly_revenues.values, palette=colors)

    # Set axis labels and title
    plt.xlabel('Year')
    plt.ylabel('Revenue (Millions)')
    plt.title(f'Top {n} Years with Highest Average Revenue by amazon')
    plt.xticks(rotation=90)
    # Show the plot
    plt.savefig("plot_top_yearly_revenues_by_amazon.png")
    plt.close()
    #plt.show()
plot_top_yearly_revenues_by_amazon(merged_data2, 20)


# In[ ]:




