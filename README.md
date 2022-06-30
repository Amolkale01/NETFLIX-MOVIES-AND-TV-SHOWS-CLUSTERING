
NETFLIX MOVIES AND TV SHOWS CLUSTERING.ipynb
NETFLIX MOVIES AND TV SHOWS CLUSTERING.ipynb_
Problem Statement
This dataset consists of tv shows and movies available on Netflix as of 2019. The dataset is collected from Flixable which is a third-party Netflix search engine.

In 2018, they released an interesting report which shows that the number of TV shows on Netflix has nearly tripled since 2010. The streaming service’s number of movies has decreased by more than 2,000 titles since 2010, while its number of TV shows has nearly tripled. It will be interesting to explore what all other insights can be obtained from the same dataset.

Integrating this dataset with other external datasets such as IMDB ratings, rotten tomatoes can also provide many interesting findings.

In this project, you are required to do
Exploratory Data Analysis

Understanding what type content is available in different countries

Is Netflix has increasingly focusing on TV rather than movies in recent years.

Clustering similar content by matching text-based features

Attribute Information
show_id : Unique ID for every Movie / Tv Show

type : Identifier - A Movie or TV Show

title : Title of the Movie / Tv Show

director : Director of the Movie

cast : Actors involved in the movie / show

country : Country where the movie / show was produced

date_added : Date it was added on Netflix

release_year : Actual Releaseyear of the movie / show

rating : TV Rating of the movie / show

duration : Total Duration - in minutes or number of seasons

listed_in : Genere

description: The Summary description

Importing libraries
[ ]
# Importing basic libraries
import pandas as pd
import numpy as np

# Importing libraries for visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')
Mounting drive and Loading Dataset
[ ]
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
[ ]
df=pd.read_csv('/content/drive/MyDrive/Capston Unsupervised Machine Learning/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv')
Looking and understanding about all the aspects of dataset.
[ ]
df.head()

[ ]
df.tail()

[ ]
#shape and size of dataset
df.shape
(7787, 12)
[ ]
#looking for any duplicates in dataset
len(df[df.duplicated()])
0
Looking for null values in dataset
[ ]
#sum of all null values for each features
df.isnull().sum()
show_id            0
type               0
title              0
director        2389
cast             718
country          507
date_added        10
release_year       0
rating             7
duration           0
listed_in          0
description        0
dtype: int64
Several features in dataset have null values.
So to handel those null values in better way, lets look for percentage of null values in dataset.
[ ]
#percentage of all null values for each features
df.isnull().sum()/df.shape[0]*100
show_id          0.000000
type             0.000000
title            0.000000
director        30.679337
cast             9.220496
country          6.510851
date_added       0.128419
release_year     0.000000
rating           0.089893
duration         0.000000
listed_in        0.000000
description      0.000000
dtype: float64
[ ]
df.drop('director', axis=1, inplace=True)
df['country'].fillna(df['country'].mode()[0], inplace=True)
df['rating'].fillna(df['rating'].mode()[0], inplace=True)
df.dropna(subset=['date_added'], inplace=True)
df['cast'].fillna('missing', inplace=True)
[ ]
#final check for null value
df.isnull().sum()/df.shape[0]*100
show_id         0.0
type            0.0
title           0.0
cast            0.0
country         0.0
date_added      0.0
release_year    0.0
rating          0.0
duration        0.0
listed_in       0.0
description     0.0
dtype: float64
All the null values handeled and dataset is ready for next challenges.
Looking for Data Types and Data Formats
[ ]
# Information about all the features of dataset
df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 7777 entries, 0 to 7786
Data columns (total 11 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   show_id       7777 non-null   object
 1   type          7777 non-null   object
 2   title         7777 non-null   object
 3   cast          7777 non-null   object
 4   country       7777 non-null   object
 5   date_added    7777 non-null   object
 6   release_year  7777 non-null   int64 
 7   rating        7777 non-null   object
 8   duration      7777 non-null   object
 9   listed_in     7777 non-null   object
 10  description   7777 non-null   object
dtypes: int64(1), object(10)
memory usage: 729.1+ KB
Date_added feature have object datatype.

Converting to datetime datatype from object datatype.

[ ]
df['date_added'] = pd.to_datetime(df['date_added'])
Duration feature have object datatype. Converting to int datatype from object datatype.

[ ]
df['duration']
0       4 Seasons
1          93 min
2          78 min
3          80 min
4         123 min
          ...    
7782       99 min
7783      111 min
7784       44 min
7785     1 Season
7786       90 min
Name: duration, Length: 7777, dtype: object
Duration are in combination of int values and string.
Removing string part so as to get int datatype.
[ ]
#spliting each values by space and selecting int part at zeroth index
df['duration'] = df['duration'].apply(lambda x : x.split(" ")[0])
[ ]
#check for updated duration values
df['duration']
0         4
1        93
2        78
3        80
4       123
       ... 
7782     99
7783    111
7784     44
7785      1
7786     90
Name: duration, Length: 7777, dtype: object
[ ]
df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 7777 entries, 0 to 7786
Data columns (total 11 columns):
 #   Column        Non-Null Count  Dtype         
---  ------        --------------  -----         
 0   show_id       7777 non-null   object        
 1   type          7777 non-null   object        
 2   title         7777 non-null   object        
 3   cast          7777 non-null   object        
 4   country       7777 non-null   object        
 5   date_added    7777 non-null   datetime64[ns]
 6   release_year  7777 non-null   int64         
 7   rating        7777 non-null   object        
 8   duration      7777 non-null   object        
 9   listed_in     7777 non-null   object        
 10  description   7777 non-null   object        
dtypes: datetime64[ns](1), int64(1), object(9)
memory usage: 729.1+ KB
Now all the features and thier values are in required datatypes and formats.
[ ]
# setting limits for display of rows and columns for better understanding
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
Creating some new features from existing features to understand data better.
[ ]
# add new features from date feature
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month
df['day_added'] = df['date_added'].dt.day
[ ]
#Check for final dataset shape
df.shape
(7777, 14)
EDA
Movies Vs. TV Shows
This dataset contains data about movies and TV shows which were added on netflix.
Lets look at dominance between Movies and TV Shows.
[ ]
# Pie chart showing percentage of toal movies and TV shows.
# Choose this facecolor so as to give style of netflix
fig, ax = plt.subplots(figsize=(5,5),facecolor="#363336")
ax.patch.set_facecolor('#363336')
explode = (0, 0.1)
ax.pie(df['type'].value_counts(), explode=explode, autopct='%.2f%%', labels= ['Movie', 'TV Show'],shadow=True,
       startangle=90,textprops={'color':"black", 'fontsize': 20}, colors =['red','#F5E9F5'])


Movies uploaded on Netflix are more than twice the TV Shows uploaded.
This dose not implies that movies are more indulging that of TV Shows.
Beacuase TV shows may have several seasons which consits of number of episodes.
Duration of TV shows are much more that of movies.
Number of Movies and TV Shows added on netflix.
On Year Basis
[ ]
fig, ax = plt.subplots(figsize=(15,6),facecolor="#363336")
ax.patch.set_facecolor('#363336')
sns.countplot(x='year_added', hue='type',lw=5, color='red', data=df, ax=ax)
ax.tick_params(axis='x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Years", color='#F5E9F5', fontsize=20)
ax.set_ylabel("Counts",  color='#F5E9F5', fontsize=20)
ax.set_title("Yearwise Movies & TV Shows", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)

TV shows are incresing continuosly.
Movies were incresing continuosly but after 2019 there is fall.
On Month Basis
[ ]
fig, ax = plt.subplots(figsize=(15,6),facecolor="#363336")
ax.patch.set_facecolor('#363336')
sns.countplot(x='month_added', hue='type',lw=5, color='red', data=df, ax=ax)
ax.tick_params(axis='x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Months", color='#F5E9F5', fontsize=20)
ax.set_ylabel("Counts",  color='#F5E9F5', fontsize=20)
ax.set_title("Monthwise Movies & TV Shows", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)

From Octomber to January, maximum number of movies and TV shows were added.
Possible reason for that is, during this period of time events such as Christmas, New Year and several holidays takes place.
On Day Basis
[ ]
fig, ax = plt.subplots(figsize=(15,6),facecolor="#363336")
ax.patch.set_facecolor('#363336')
sns.countplot(x='day_added', hue='type',lw=5, color='red', data=df, ax=ax)
ax.tick_params(axis='x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Days", color='#F5E9F5', fontsize=20)
ax.set_ylabel("Counts",  color='#F5E9F5', fontsize=20)
ax.set_title("Daywise Movies & TV Shows", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)


Maximum number of movies and TV shows were either on start of the month or mid of the month.
Wordwide Presence of Netflix
Popularity Netflix is all over the world.
Lets look for its highest presence over countries.
Top 10 Countries having maximum Movies and TV Shows
[ ]
fig, ax = plt.subplots(figsize=(12,5),facecolor="#363336")
ax.patch.set_facecolor('#363336')
(df['country'].value_counts().sort_values()/df.shape[0]*100)[-10:].plot(kind='barh', ax=ax,color ='red',alpha=0.8)
ax.tick_params(axis= 'x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Percentage", color='#F5E9F5', fontsize=20)
ax.set_title("Top 10 Countries", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

Unites State tops the in list of maximum number of movies and TV shows.
Followed by India, UK and Japan.
Ratings on Movies and TV Shows
For Movies:

G: Kids
PG: Older Kids (7+)
PG-13: Teens (13+)
NC-17, NR, R, Unrated: Adults (18+)
For TV Shows:

TV-G, TV-Y: Kids
TV-Y7/FV/PG: Older Kids (7+)
TV-14: Young Adults (16+)
TV-MA: Adults (18+)
[ ]
fig, ax = plt.subplots(figsize=(15,6),facecolor="#363336")
ax.patch.set_facecolor('#363336')
sns.countplot(x='rating', hue='type',lw=5, color='red', data=df, ax=ax)
ax.tick_params(axis='x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Ratings", color='#F5E9F5', fontsize=20)
ax.set_ylabel("Counts",  color='#F5E9F5', fontsize=20)
ax.set_title("Ratings: Movies & TV Shows", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)

Maximum of the movies as well as TV shows are for matures only.
Top 10 Cast Involved either in Movies or TV Shows
[ ]
# just take look at values of cast
df['cast'][1]

[ ]
# list of cast by making split by comma
df['cast'] = df['cast'].apply(lambda x :  x.split(', '))
[ ]
# making list which contains all the entries from rows
cast_list = []
for i in df['cast']:
  cast_list += i

# unique cast    
unique_cast = set(cast_list)

# create dictionary to save cast and their counts
cast_dict = dict((i, cast_list.count(i)) for i in unique_cast)

# create dataframe from above dictionary
cast_df = pd.DataFrame.from_dict(cast_dict, orient='index',
                       columns=['Counts']).sort_values('Counts',ascending=True).reset_index().rename(columns = {'index' : 'cast'})
[ ]
# plot of top 10 cast involved either movies or tv shows
fig, ax = plt.subplots(figsize=(10,5),facecolor="#363336")
ax.patch.set_facecolor('#363336')
ax.barh(y=cast_df['cast'][-11:-1], width = cast_df['Counts'][-11:-1], height=0.4, color = 'red',alpha=0.8)
ax.tick_params(axis= 'x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Number of movies", color='#F5E9F5', fontsize=20)
ax.set_title("Top 10 Cast", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)

Anupam Kher have maximum number of movies or TV shows.
Seperate Dataframes for Movies and TV Shows
[ ]
# All the movies and TV shows in different dataframe
movies_df = df[df['type']=='Movie']
tv_df = df[df['type']=='TV Show']
Running Time of Movies
[ ]
fig, ax = plt.subplots(figsize=(20,7),facecolor="#363336")
ax.patch.set_facecolor('#363336')
sns.distplot(movies_df['duration'], hist=True, bins=60,color='red', ax=ax)
ax.tick_params(axis= 'x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Duration", color='#F5E9F5', fontsize=20)
ax.set_ylabel("Density", color='#F5E9F5', fontsize=20)
ax.set_title("Running Time of Movies", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
Majority of movies have running time of between 50 to 150 min.
Seasons of TV Shows
[ ]
fig, ax = plt.subplots(figsize=(8,5),facecolor="#363336")
ax.patch.set_facecolor('#363336')
(tv_df['duration'].value_counts().sort_values()/tv_df.shape[0]*100)[-10:].plot(kind='barh', ax=ax,color ='red',alpha=0.5)
ax.tick_params(axis='x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Percentage", color='#F5E9F5', fontsize=20)
ax.set_ylabel("Number of Seasons", color='#F5E9F5', fontsize=20)
ax.set_title("TV Shows Seasons", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)

Almost 68% of TV shows consist of single season only.
Top Genres in Movies & TV Shows
[ ]
# list of genres by making split by comma
movies_df['listed_in'] = movies_df['listed_in'].apply(lambda x :  x.split(','))
tv_df['listed_in'] = tv_df['listed_in'].apply(lambda x :  x.split(','))
[ ]
# creating list having all the genres in dataset
movie_genre_list = []
for i in movies_df['listed_in']:
  movie_genre_list += i

tv_genre_list = []
for i in tv_df['listed_in']:
  tv_genre_list += i
[ ]
# to make dictionary out of list of genres
from collections import Counter
[ ]
# creating dataframe for genres in movies and TV shows
movie_genre_list = Counter(movie_genre_list)
tv_genre_list = Counter(tv_genre_list)

movie_genre_df = pd.DataFrame(movie_genre_list.items())
TV_genre_df = pd.DataFrame(tv_genre_list.items())

movie_genre_df.columns = ['genre','movie_count']
TV_genre_df.columns = ['genre','tv_count']

movie_genre_df = movie_genre_df.sort_values(by= 'movie_count').reset_index(drop=True)
TV_genre_df = TV_genre_df.sort_values(by= 'tv_count').reset_index(drop=True)
[ ]
# first look of movie genres dataframe
movie_genre_df
[ ]
# first look of TV shows genres dataframe
TV_genre_df
Some rows seems to be exactly same but due to initial space attached they are different.
Need to remove those spaces, so as to get total count of each genres.
After removing spaces, aggregation using pandas pivot table in order to get sum of counts of each genres.
[ ]
# to remove spaces around the word
movie_genre_df['genre']=movie_genre_df['genre'].str.strip()
TV_genre_df['genre']=TV_genre_df['genre'].str.strip()
[ ]
# Aggregating dataframe so as to get final counts of each genres
movie_genre_df = pd.pivot_table(movie_genre_df, index=['genre'],values=['movie_count'], aggfunc='sum').reset_index().sort_values('movie_count')
TV_genre_df = pd.pivot_table(TV_genre_df, index=['genre'],values=['tv_count'], aggfunc='sum').reset_index().sort_values('tv_count')
[ ]
# Final movie genres datafrmae look like
movie_genre_df
[ ]
# Final TV shows genres datafrmae look like
TV_genre_df
Now do not have repetative rows in genres.
Top 5 Genres with Maximum Number of Movies
[ ]
# bar plot showing top 5 genres in movies
fig, ax = plt.subplots(figsize=(10,5),facecolor="#363336")
ax.patch.set_facecolor('#363336')
ax.barh(y=movie_genre_df['genre'][-5:], width = movie_genre_df['movie_count'][-5:]/movie_genre_df['movie_count'].sum()*100, height=0.5, color = 'red',alpha=0.8)
ax.tick_params(axis= 'x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Percentage", color='#F5E9F5', fontsize=20)
ax.set_ylabel("Genres", color='#F5E9F5', fontsize=20)
ax.set_title("Top 5 Genres in Movies", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)

[ ]
# bar plot showing top 5 genres in TV shows
fig, ax = plt.subplots(figsize=(10,5),facecolor="#363336")
ax.patch.set_facecolor('#363336')
ax.barh(y=TV_genre_df['genre'][-5:], width = TV_genre_df['tv_count'][-5:]/TV_genre_df['tv_count'].sum()*100,
        height=0.5, color = 'red',alpha=0.8)
ax.tick_params(axis= 'x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Percentage", color='#F5E9F5', fontsize=20)
ax.set_ylabel("Genres", color='#F5E9F5', fontsize=20)
ax.set_title("Top 5 Genres in TV Shows", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)

Top 3 genres are exactly same for movies and TV shows.
Dramas genres hit all over the world.
Originally Uploaded on Netflix
Some movies and TV shows were actually released in the past and they were added later on Netflix.
But some movies and TV shows were released on Netflix itself. Named those as Netflix Originals.
Originals : For which released year and added year is same.
Originals in Movies
Creating new feature as originals having values Yes and No.
[ ]
movies_df['originals'] = np.where(movies_df['release_year']==movies_df['year_added'], 'Yes', 'No')
[ ]
# pie plot showing percentage of originals and others in movies
fig, ax = plt.subplots(figsize=(5,5),facecolor="#363336")
ax.patch.set_facecolor('#363336')
explode = (0, 0.1)
ax.pie(movies_df['originals'].value_counts(), explode=explode, autopct='%.2f%%', labels= ['Others', 'Originals'],
       shadow=True, startangle=90,textprops={'color':"black", 'fontsize': 20}, colors =['red','#F5E9F5'])

30% movies released on Netflix.
70% movies added on Netflix were released earlier by different mode.
May be after buying rights of old released movies and then adding all the movies on Netflix.
Data Preprocessing
[ ]
df.head()

[ ]
# this is how values in cast feature looks like
df['cast'][0]

[ ]
# cast contains list of strings; so joining those strings in a string
df['cast'] = df['cast'].apply(lambda x: ','.join(map(str, x)))
[ ]
# this is how values in cast feature looks now
df['cast'][0]

[ ]
# cast name contains space between initials, if it pass directly to clustering algoritm, it consider them as two different name
# initial can be same but different last name
# so we need to remove space between initial and last name
df['cast'] = df['cast'].apply(lambda x: x.replace(' ','').split(','))

[ ]
# above same operation applicable on country name, to remove complexity of two or more words in country name
df['country'] = df['country'].apply(lambda x: x.replace(' ','').split(','))
[ ]
# above same operation done on listed_in
df['listed_in'] = df['listed_in'].apply(lambda x: x.replace(' ','').split(','))

[ ]
# after above all the changes, those features are in list format, so making list of description feature
df['description'] = df['description'].apply(lambda x: x.split(' '))

[ ]
# creating new feature for clustering
df['text']=df['cast']+df['country']+df['listed_in']+df['description']
[ ]
# converting text feature to string from list
df['text']= df['text'].apply(lambda x: " ".join(x))
[ ]
# making all the words in text feature to lowercase
df['text']= df['text'].apply(lambda x: x.lower())
Removing Punctuations
[ ]
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)
[ ]
# applying above function on text feature
df['text']= df['text'].apply(remove_punctuation)
[ ]
# this is how value in text looks like after removing punctuations
df['text'][0]

Removing StopWords

[ ]
# using nltk library to download stopwords
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
sw=stopwords.words('english')
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[ ]
#Defining stopwords 
def stopwords(text):
    '''a function for removing the stopword'''
    text = [word for word in text.split() if word not in sw]
    # joining the list of words with space separator
    return " ".join(text)
[ ]
# applying above function on text feature
df['text']=df['text'].apply(stopwords)
[ ]
# this is how value in text looks like after removing stopwords
df['text'][0]

Stemming

[ ]
from nltk.stem.snowball import SnowballStemmer
[ ]
# Create an object of stemming function
stemmer = SnowballStemmer("english")

def stemming(text):    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)
[ ]
# applying above stemming function on text feature
df['text']=df['text'].apply(stemming)
[ ]
# this is how value in text looks like after applying stemming function
df['text'][0]

[ ]
# Final datafrmae will look like this after preprocessing
df.head()

YES!!! Data Preprocessing is done now. Data is as per our requirement to carry out clustering.

K-Means Clustering
Here we have textual data. Unfortunately clustering algoritms can not understand textual data. So, we use vectorization techniuque to convert textual data to numerical vectors.

[ ]
# importing TfidVectorizer from sklearn library
from sklearn.feature_extraction.text import TfidfVectorizer

# sublinear_df is set to True to use a logarithmic form for frequency
# min_df is the minimum numbers of documents a word must be present in to be kept
# norm is set to l2, to ensure all our feature vectors have a euclidian norm of 1
# ngram_range is set to (1, 2) to indicate that we want to consider both unigrams and bigrams
# stop_words is set to "english" to remove all common pronouns ("a", "the", ...) to reduce the number of noisy features

vectorizer = TfidfVectorizer(sublinear_tf= True, min_df=10, norm='l2', ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(df["text"])

# to look at the values of numerical vectors X
pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()).head()

[ ]
# X is sparse matrix
X
<7777x2817 sparse matrix of type '<class 'numpy.float64'>'
	with 129267 stored elements in Compressed Sparse Row format>
[ ]
#Performing the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

score = []
# for loop to append kmeans inertia values
for k in range(1,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    score.append(kmeans.inertia_)

#Plot linegraph
plt.figure(figsize=(10,6))

plt.plot(range(1,10),score,"-o")
plt.grid(True)
plt.xlabel("Number of Clusters(k)",fontsize=12)
plt.ylabel("score",fontsize=12)
plt.title("Elbow Method For Optimal k",fontsize=14)
plt.xticks(range(1,10))
plt.tight_layout()

[ ]
#Finding the optimal number of cluster using silhoutte method
# will be taking number of clusters from 2 to 10
from sklearn.metrics import silhouette_score
for n in range(2,10):
    clusterer = KMeans(n_clusters=n)
    preds = clusterer.fit_predict(X)
    centers = clusterer.cluster_centers_

    score = silhouette_score(X, preds)
    print("For n_clusters = {}, silhouette score is {}".format(n, score))
For n_clusters = 2, silhouette score is 0.007247013210482062
For n_clusters = 3, silhouette score is 0.009323314366074056
For n_clusters = 4, silhouette score is 0.010626519785887868
For n_clusters = 5, silhouette score is 0.010492040711001578
For n_clusters = 6, silhouette score is 0.012225862627264857
For n_clusters = 7, silhouette score is 0.013578617646512872
For n_clusters = 8, silhouette score is 0.014722238789450488
For n_clusters = 9, silhouette score is 0.015475879315089496
Maximum value of Silhouette score is for k equals to 9.
[ ]
# converting X to array
y=X.toarray()
[ ]
y
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
From Elbow curve and Silhoueete Score we can conclude that optimal number of clusters will be near about 25.
Lets plot the Silhouette plot to fix the number of cluster.
[ ]
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm

range_n_clusters = [2,3,4,5,6,7,8,9]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(y) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(y)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(y, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(y, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) /n_clusters)
    ax2.scatter(y[:, 0], y[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

[ ]
# choosing optimal value of k for clustering
k = 9
[ ]
#Clustering the dataset with the optimal number of clusters
model = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=600, tol=0.000001, random_state=0)
model.fit(X)
KMeans(max_iter=600, n_clusters=9, random_state=0, tol=1e-06)
[ ]
#Predict the clusters and evaluate the silhouette score
clusters = model.predict(X)
score = silhouette_score(X, clusters)
print("Silhouette score is {}".format(score))
Silhouette score is 0.01578908369818921
[ ]
#Adding a seperate column for the cluster
df["cluster"] = clusters
[ ]
df.head()

[ ]
df['cluster'].value_counts()
7    1551
4    1145
2    1014
1     958
8     782
5     760
0     713
6     518
3     336
Name: cluster, dtype: int64
[ ]
ax.patch.set_facecolor('#363336')
fig, ax = plt.subplots(figsize=(15,6),facecolor="#363336")
sns.countplot(x='cluster', hue='type',lw=5, color='red', data=df, ax=ax)
ax.tick_params(axis='x', colors='#F5E9F5',labelsize=15) 
ax.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
ax.set_xlabel("Clusters", color='#F5E9F5', fontsize=20)
ax.set_ylabel("Counts",  color='#F5E9F5', fontsize=20)
ax.set_title("Clusterwise Movies & TV Shows", color='#F5E9F5', fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)

Most of clusters shows either movies or TV shows.
Some of the clusters shows both in them.
[ ]
df[df['cluster']==1]

Recommender System
[ ]
# take a look to decide which features to use for recommender system
df.tail(2)

[ ]
# title and text features used for recommender system
final_df= df[['title', 'text']].reset_index()
[ ]
# reseting new index and droping old index
final_df.drop('index', axis=1, inplace=True)
[ ]
#look at final dataframe
final_df.head()

[ ]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=4000,stop_words='english')
[ ]
vector = cv.fit_transform(final_df['text']).toarray()
vector.shape
(7777, 4000)
[ ]
# Cosine similarity is a metric that measures the cosine of the angle between two vectors projected in a multi-dimensional space.
from sklearn.metrics.pairwise import cosine_similarity   
[ ]
similarity = cosine_similarity(vector)
similarity 
array([[1.        , 0.        , 0.06085806, ..., 0.        , 0.08164966,
        0.        ],
       [0.        , 1.        , 0.05143445, ..., 0.15018785, 0.        ,
        0.        ],
       [0.06085806, 0.05143445, 1.        , ..., 0.05407381, 0.        ,
        0.        ],
       ...,
       [0.        , 0.15018785, 0.05407381, ..., 1.        , 0.        ,
        0.37851665],
       [0.08164966, 0.        , 0.        , ..., 0.        , 1.        ,
        0.0745356 ],
       [0.        , 0.        , 0.        , ..., 0.37851665, 0.0745356 ,
        1.        ]])
[ ]
# how to fetch index of movie by its name
final_df[final_df['title'] == "Zumbo's Just Desserts"].index[0]
7775
[ ]
# function to get 10 movies which are at very near distance from given movie
def recommend(movie):
     # first fetch the index of given movie
    index = final_df[final_df['title'] == movie].index[0]
    # then list down the closest movies index
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    # selecting top 10 movies except first one; first one will be same movie as given on
    for i in distances[1:11]:
        print(final_df.iloc[i[0]].title)
[ ]
#example 1
recommend("Zumbo's Just Desserts")
Ink Master
Fit for Fashion
The Apartment
Instant Hotel
My Hotter Half
Nailed It! Mexico
The Chefs' Line
Nailed It! France
Skin Wars
Yummy Mummies
[ ]
# example 2
recommend("50 First Dates")
Big Daddy
Mr. Deeds
The Knight Before Christmas
Four Christmases
Hubie Halloween
Employee of the Month
Heartbreakers
When We First Met
As Good as It Gets
Click
In example 2: All the movies are of romantic genres.

Conclusions
Movies uploaded on Netflix are more than twice the TV Shows uploaded.
TV shows and movies are incresing continuosly but in 2019 there is drop in number of movies.
From Octomber to January, maximum number of movies and TV shows were added.
Maximum number of movies and TV shows were either on start of the month or mid of the month.
United State tops in the list of maximum number of movies and TV shows followed by India, UK and Japan.
Maximum of the movies as well as TV shows are for matures only.
Anupam Kher top from the list of casts having maximum number of movies and TV shows.
Majority of movies have running time of between 50 to 150 min.
Almost 68% of TV shows consist of single season only.
Top 3 genres are exactly same for movies and TV shows.
Dramas genres hit all over the world.
30% movies and 50% TV shows are Netflix Originals.
Clustering done by K-Means Clustering, found optimal number of clusters equal to 9 with highest Silhoeutte Score.
Recommender system using cosine similarirty performs well on data.
Creating a copy...
