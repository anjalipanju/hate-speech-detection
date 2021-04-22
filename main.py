# Importing the initial libraries 
import pandas as pd
import numpy as np
import os
import re


# In[3]:


# Loading the dataset
tweets = pd.read_csv('https://github.com/anjalipanju/hate-speech-detection/blob/main/project_dataset_new.xlsx')
tweets.head()
# tweets.info()


# In[4]:


# Looking at the total number of labels
# A highly unbalanced dataset
# tweets.label.value_counts()
tweets.label.value_counts(normalize=True)


# In[5]:


tweets.tweet.sample().values[0]


# In[6]:


# Getting the tweets into a list 
# The tweets contains @user id handles, hashtags, url links, etc
tweet_list = tweets.tweet.values
# len(tweet_list)
tweet_list[:5]


# In[7]:


# Cleaning the tweet list - Step by Step
# 1. Normalize the casing
# 2. Using regular expressions, remove user handles. These begin with '@’
# 3. Using regular expressions, remove URLs
# 4. Using TweetTokenizer from NLTK, tokenize the tweets into individual terms
# 5. Remove stop words.
# 6. Remove redundant terms like ‘amp’, ‘rt’, etc
# 7. Remove ‘#’ symbols from the tweet while retaining the term
import re


# In[8]:


# Normalizing the casing to lower
lower_tweets = [twt.lower() for twt in tweet_list]
lower_tweets[:5]


# In[9]:


# Removing @
# re.sub("@\w+","", "@chocolate is the best! http://rahimbaig.com/ai")
no_user = [re.sub("@\w+","", twt) for twt in lower_tweets]
no_user[:5]


# In[10]:


# Removing url links
# re.sub("\w+://\S+","", "@chocolate is the best! http://rahimbaig.com/ai")
no_url = [re.sub("\w+://\S+","", twt) for twt in no_user]
no_url[:5]


# In[11]:


# Tokenization
from nltk.tokenize import TweetTokenizer
token = TweetTokenizer()
# print(token.tokenize(no_url[0]))
final_token = [token.tokenize(sent) for sent in no_url]
print(final_token[0])


# In[12]:


from nltk.corpus import stopwords
from string import punctuation

stop_nltk = stopwords.words("english")
stop_punct = list(punctuation)
stop_punct.extend(['...','``',"''",".."])
stop_context = ['rt', 'amp']
stop_final = stop_nltk + stop_punct + stop_context


# In[13]:


# Creating a function for removing terms with lenght = 1

def Remover(sent):
    return [re.sub("#","",term) for term in sent if ((term not in stop_final) & (len(term)>1))]

Remover(final_token[0])


# In[14]:


# Final set of tweets
clean_tweets = [Remover(tweet) for tweet in final_token]
clean_tweets[:5]


# #### Check out the top terms in the tweets

# In[15]:


# Looking for the top terms
# Creating an emply list and putting the top values in it
from collections import Counter
top_terms = []
for tweet in clean_tweets:top_terms.extend(tweet)


# In[16]:


toppr = Counter(top_terms)
toppr.most_common(10)


# In[17]:


# Preparing the cleaned data for modeling
# Converting tokens into strings
clean_tweets[0]


# In[18]:


clean_tweets = [" ".join(tweet) for tweet in clean_tweets]
clean_tweets[0]


# In[19]:


# Splitting the data 70/30
from sklearn.model_selection import train_test_split
x = clean_tweets
y = tweets.label.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 1)
