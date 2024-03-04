#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')
get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install textblob')


# In[ ]:


#Load the dataset -> feature extraction ->data visualization
# -> data cleaning -> train test split
# -> model building -> model training > model evaluation
# -> model saving -> streamlit application deploy


# In[4]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd


# In[6]:


df = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_sentiment.csv", header = None, index_col=0)
df.head()


# In[8]:


df = df[[2,3]].reset_index(drop=True)


# In[9]:


df.columns = ['sentiment','text']
df.head(3)


# In[10]:


df.info()


# In[18]:


df.isnull().sum()
df.dropna(inplace=True)
sum(df['text'].apply(len)>5), sum(df['text'].apply(len)<=5)


# In[23]:


print(df.shape)
df = df[df['text'].apply(len)>5]
print(df.shape)


# In[25]:


df['sentiment'].value_counts()


# In[28]:


#Preprocessing  preprocess_kgptalkie
get_ipython().system('pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall')


# In[30]:


import preprocess_kgptalkie as ps

df.columns


# In[33]:


df = ps.get_basic_features(df)
df.columns


# In[34]:


df.head()


# In[42]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,6))
num_cols = df.select_dtypes(include='number').columns

for index,col in enumerate(num_cols):
    plt.subplot(2,4, index+1)
    sns.kdeplot(data=df, x=col, hue="sentiment", fill=False)
plt.tight_layout()
plt.show()


# In[43]:


df['sentiment'].value_counts().plot(kind='pie',autopct='%1.0f%%')


# In[47]:


#word cloud visualization
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS


# In[49]:


stopwords = set(STOPWORDS)


# In[53]:


wordcloud = WordCloud(background_color='white', stopwords=stopwords,
                     max_words=300,max_font_size=40,scale=5).generate(str(df['text']))

plt.imshow(wordcloud)


# In[58]:


plt.figure(figsize=(40,20))

for index, sent in enumerate(df['sentiment'].unique()):
    plt.subplot(2,2, index+1)
    
    data = df[df['sentiment'] == sent]['text']
    wordcloud = WordCloud(background_color='white',stopwords=stopwords,max_words=300,
                         max_font_size=40, scale=5).generate(str(data))
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    plt.title(sent, fontsize=40)


# In[ ]:


# Data cleaning


# In[59]:


df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: ps.remove_urls(x))
df['text'] = df['text'].apply(lambda x: ps.remove_html_tags(x))
df['text'] = df['text'].apply(lambda x: ps.remove_rt(x))
df['text'] = df['text'].apply(lambda x: ps.remove_special_chars(x))


# In[60]:


#train test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df['text'],
                                                df['sentiment'],
                                                test_size=0.2,
                                                random_state=0)


# In[73]:


X_train.shape,X_test.shape


# In[72]:


#Model Building


# In[80]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

clf = Pipeline([('tfid', TfidfVectorizer()),
               ('rfc',RandomForestClassifier(n_jobs=-1))])

clf.fit(X_train,y_train)


# In[81]:


#evaluation
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)


# In[82]:


print("Classification Report:\n",classification_report(y_test,y_pred))


# In[83]:


import pickle

pickle.dump(clf,open("twitter_sentiment.pkl","wb"))


# In[84]:


clf.predict(['let me not upset you'])


# In[85]:


clf.predict(['glad to meet u'])


# In[87]:


clf.predict(['how the hell are we in to halloween month already'])


# In[ ]:




