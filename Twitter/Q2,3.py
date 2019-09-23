# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:36:22 2018

@author: Amruth
"""

import pandas as pd
import numpy as np
import re, json, datetime, pytz
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from textblob import TextBlob
from sklearn.metrics import confusion_matrix,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm

def get_data(text,return_day_count=False,return_hour_count=False):
    with open(text+'.txt', 'r',encoding="utf8") as f:
        retweet_count,month,day,hour,followers,place,title,time = [],[],[],[],[],[],[], []
        pst_tz = pytz.timezone('US/Pacific') 
        for line in f:
            tweet = json.loads(line)
            retweet_count.append(tweet['metrics']['citations']['total']) 
            month.append(datetime.datetime.fromtimestamp(tweet['citation_date'], pst_tz).month)
            day.append(datetime.datetime.fromtimestamp(tweet['citation_date'], pst_tz).day)
            hour.append(datetime.datetime.fromtimestamp(tweet['citation_date'], pst_tz).hour)
            followers.append(tweet['author']['followers'])
            time.append(datetime.datetime.fromtimestamp(tweet['citation_date'], pst_tz))
            place.append(tweet['tweet']['user']['location'])
            title.append(tweet['title'])
    
    d = {'retweet_count':retweet_count,'month':month,'day':day,'hour':hour,
    'followers':followers,'time':time,'place':place,'title':title}
    df = pd.DataFrame(d)
    day_count = np.array(month)*31 + np.array(day) - 44
    hour_count = (day_count-1)*24 + np.array(hour)
    if return_day_count:
        df['day_count'] = day_count
    if return_hour_count:
        df['hour_count'] = hour_count
    return df   

#################################################################################
#Problem 1.1
#################################################################################

def get_stats(file):
    df = get_data(file)
    print('Average number of tweets per hour is',df.groupby(['month','day','hour']).count().mean())
    print('Average number of followers is',np.mean(df['followers']))   
    print('Average number of retweets is', np.mean(df['retweet_count']))

get_stats('tweets_#superbowl')

def hist_plot(df):
    tweet_hour = df.groupby(['month','day','hour']).count()['retweet_count'].tolist()
    plt.bar(range(len(tweet_hour)),tweet_hour)
    plt.xlabel('Hour')
    plt.ylabel('Number of tweets in hour')
    plt.title('Histogram of number of tweets per hour')
    plt.show()

hist_plot(nfl)
hist_plot(superbowl)

#################################################################################
#Problem 2
#################################################################################                  
superbowl = get_data('tweets_#superbowl')

#massachusetts = ['massachusetts', 'boston','worcester', 'springfield', 'lowell', 'cambridge',
#                'plymouth','quincy','bedford', 'ma']
#washington = ['washington','seattle','spokane','tacoma','vancouver','bellevue','everett',
#              'kent','yakima','renton','kirkland','marysville','olympia','wa','wash']
              
massachusetts = ['massachusetts', 'boston', 'ma']
washington = ['washington','seattle','kirkland','wa','wash']

superbowl['place_id']=None
any_in = lambda a, b:not set(a).isdisjoint(b)
for i in range(len(superbowl)):
    if i%5000==0:
        print(i)
    place_list = re.sub('[^a-zA-Z]', ' ', superbowl.loc[i,'place']).lower().split(' ')
    if any_in(place_list, massachusetts):
        superbowl.loc[i, 'place_id'] = 1
    elif any_in(place_list, washington):
        superbowl.loc[i, 'place_id'] = 0

#To save the dataframe and load it
#superbowl.to_pickle('superbowl')
#superbowl = pd.read_pickle('superbowl')
wash_mass = superbowl[superbowl['place_id']>=0]
wash_mass = wash_mass.reset_index()

#Cleaning the tweet text
stop_words = text.ENGLISH_STOP_WORDS
corpus = []
ps = PorterStemmer()
for i in range(len(wash_mass)):
    text = re.sub('[^a-zA-Z]', ' ', wash_mass.loc[i,'title']).lower().split()
    text = [ps.stem(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    corpus.append(text)
    
#Splitting into training and test sets    
y = wash_mass['place_id'].tolist()
train_corpus,test_corpus,y_train,y_test = train_test_split(corpus,y,test_size=0.2,random_state=0)
tfidf = TfidfVectorizer(min_df=5)
tfidf_matrix_train = tfidf.fit_transform(train_corpus).toarray()
tfidf_matrix_test = tfidf.transform(test_corpus).toarray()

#Using LSI for dimensionality reduction
svd = TruncatedSVD(n_components = 50, random_state = 0)
X_train = svd.fit_transform(tfidf_matrix_train)
X_test = svd.transform(tfidf_matrix_test)

def classification(classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print('Accuracy is',(cm[0,0] + cm[1,1])/cm.sum())
    print('Precision is',cm[1,1] / (cm[1,1] + cm[0,1]))
    print('Recall is',cm[1,1] / (cm[1,1] + cm[1,0]))
    print(confusion_matrix(y_test, y_pred))
    y_pred_prob = classifier.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr)
    plt.title('ROC curve for the Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0,1.1)
    plt.show()

#Logistic Regression
classification(LogisticRegression(random_state = 0))
#Soft margin SVM
#classification(SVC(kernel = 'linear', random_state = 0, C=0.001, probability = True))   
#Naive Bayes Classifier
classification(GaussianNB())
#Random Forest Classifier 
classification(RandomForestClassifier(n_estimators=10))
#Neural network classifier
classification(MLPClassifier())
#KNN Classifier
classification(KNeighborsClassifier(n_neighbors=8))

#################################################################################
#Problem 3 (Sentiment Analysis)
################################################################################# 
def clean_tweets(tweets):
    clean_tweets = []
    for tweet in tweets:
        clean_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).lower().split())
        clean_tweets.append(clean_tweet)
    return clean_tweets
    
def get_sentiment(tweets):
    sentiments = []
    positive,negative,neutral =0,0,0
    for tweet in tweets:
        blob = TextBlob(tweet)
        sentiments.append(blob.sentiment.polarity)
        if blob.sentiment.polarity > 0:
            positive+=1
        elif blob.sentiment.polarity == 0:
            neutral+=1
        else:
            negative+=1
    print('Number of positive tweets =',positive)
    print('Number of negative tweets =',negative)
    print('Number of neutral tweets =',neutral)
    enc_sentiment = [1 if i>0 else 0 if i==0 else -1 for i in sentiments]
    return sentiments, enc_sentiment

def plot_sentiment(df,title,by='hour_count'):
    grouped_data = pd.DataFrame(df.groupby([by,'sentiments']).count())
    num_time = int(np.max(df[by]))
    positive,negative,neutral =np.zeros(num_time),np.zeros(num_time),np.zeros(num_time)
       
    for index in grouped_data.index.values:
        if index[1] == 1:
            positive[index[0]-1] = grouped_data.iloc[index]['title']
        elif index[1] == -1:
            negative[index[0]-1] = grouped_data.iloc[index]['title']
        else:
            neutral[index[0]-1] = grouped_data.iloc[index]['title']
    plt.plot(range(1,num_time+1),positive,label='Positive tweets')
    plt.plot(range(1,num_time+1),negative,label='Negative tweets')
    plt.plot(range(1,num_time+1),neutral,label='Neutral tweets')
    if by=='day_count':
        plt.xlabel('Day count')
    else:
        plt.xlabel('Hour count')
    plt.ylabel('Number of tweets')
    plt.title('Variations of different types of tweets for '+title)
    plt.legend(loc=9)
    plt.show()
    
tweet_data = get_data('tweets_#superbowl',return_hour_count=True,return_day_count=True)  
tweet_titles = tweet_data['title'].tolist()
cleaned_tweets = clean_tweets(tweet_titles)
sentiments, enc_sentiments = get_sentiment(cleaned_tweets)   
tweet_data['sentiments'] = enc_sentiments
plot_sentiment(tweet_data,title='#gosuperbowl')

#################################################################################
#Problem 1.2
################################################################################# 

gopatriots = get_data('tweets_#gopatriots',return_hour_count=True)
grp = gopatriots.groupby(pd.Grouper(key='time', freq='60min'))

num_hours = len(grp)
num_features = 5
X = np.zeros((num_hours,num_features))
for i,(key,val) in enumerate(grp):
    X[i,:] = [val.followers.count(),val.retweet_count.sum(),val.followers.sum(),val.followers.max(),(i+1)%24]
X = np.nan_to_num(X)    
y = X[1:,0]
X = X[:-1,:]

results = sm.OLS(y, X).fit()
print(results.summary())

    
#################################################################################
#Problem 3 (Wordcloud)
#################################################################################
def full_clean(tweets):
    from sklearn.feature_extraction import text
    stop_words = text.ENGLISH_STOP_WORDS
    corpus = []
    for tweet in tweets:
        clean_tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).lower().split()
        text = [word for word in clean_tweet if not word in stop_words]
        text = ' '.join(text)
        corpus.append(text)
    return corpus
    
from collections import Counter
from wordcloud import WordCloud
def get_wordcloud(text):
    df = get_data(text)
    tweets = df['title'].tolist()
    cleaned_tweets = full_clean(tweets)
    full_text = ' '.join(cleaned_tweets)
    cleaned_tweets = full_text.split()
    #wordcloud = WordCloud(max_font_size=40).generate(full_text)
    wordcloud = WordCloud(max_font_size=40).generate_from_frequencies(Counter(cleaned_tweets))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

get_wordcloud('tweets_#gopatriots')    


#################################################################################
#Problem 3 (Summary)
#################################################################################

def get_media_data(text):
    with open(text+'.txt', 'r',encoding="utf8") as f:
        name, media,day,hour = [],[],[],[]
        pst_tz = pytz.timezone('US/Pacific') 
        for line in f:
            tweet = json.loads(line)
            name.append(tweet['tweet']['user']['screen_name'])
            day.append(datetime.datetime.fromtimestamp(tweet['citation_date'], pst_tz).day)
            hour.append(datetime.datetime.fromtimestamp(tweet['citation_date'], pst_tz).hour)
            media.append(tweet['tweet']['entities']['urls'])
    d = {'name':name,'media':media,'day':day,'hour':hour}
    df = pd.DataFrame(d)
    return df   

def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

superbowl_media = get_media_data('tweets_#superbowl')
#superbowl_media.to_pickle('superbowl_media')
#superbowl_media = pd.read_pickle('superbowl_media')

target_day = superbowl_media[superbowl_media['day']==1]
target_time = target_day[target_day['hour']==19]
target = target_time[target_time['name']=='YahooSports']
url = target.iloc[0]['media'][0]['expanded_url']

import requests
from readability import Document
response = requests.get(url)
doc = Document(response.text)
print(remove_tags(doc.summary()))
t = open('article.txt','w')
t.write(remove_tags(doc.summary()))
t.close()

from sumy.parsers.plaintext import PlaintextParser 
from sumy.nlp.tokenizers import Tokenizer 
from sumy.summarizers.lex_rank import LexRankSummarizer

file = "article.txt"
parser = PlaintextParser.from_file(file, Tokenizer("english"))
summarizer = LexRankSummarizer()
summary = summarizer(parser.document, 5) 
print(doc.title())
for sentence in summary:
    print(sentence)
    
#################################################################################
#Problem 3 ()
#################################################################################


#superbowl = pd.read_pickle('superbowl')
superbowl = superbowl[['place','title','place_id']]
any_in = lambda a, b:not set(a).isdisjoint(b)
place_list = []
for i in range(len(superbowl)):
    if i%5000==0:
        print(i)
    place_list.append(re.sub('[^a-zA-Z]', ' ', superbowl.loc[i,'place']).lower().split(' '))

superbowl['place_list'] = place_list    
        
wash_mass = superbowl[superbowl['place_id']>=0]
wash_mass = wash_mass.reset_index()
y_train = wash_mass['place_id'].tolist()

#Cleaning the tweet text
stop_words = text.ENGLISH_STOP_WORDS
train_corpus,test_corpus = [],[]
ps = PorterStemmer()
for i in range(len(wash_mass)):
    text = re.sub('[^a-zA-Z]', ' ', wash_mass.loc[i,'title']).lower().split()
    text = [ps.stem(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    train_corpus.append(text)
    
#Forming training tfidf matrix    
tfidf = TfidfVectorizer(min_df=5)
tfidf_matrix_train = tfidf.fit_transform(train_corpus).toarray()

#Using LSI for dimensionality reduction
svd = TruncatedSVD(n_components = 50, random_state = 0)
X_train = svd.fit_transform(tfidf_matrix_train)

classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)

def city_support(city): 
    for i in range(len(superbowl)):
        if i%5000==0:
            print(i) 
        if any_in(place_list[i], city):
            superbowl.loc[i, 'place_id'] = -1  
    target_city = superbowl[superbowl['place_id']<0]
    target_city = target_city.reset_index() 
    print('Starting to clean tweets')       
    for i in range(len(target_city)):
        text = re.sub('[^a-zA-Z]', ' ', target_city.loc[i,'title']).lower().split()
        text = [ps.stem(word) for word in text if not word in stop_words]
        text = ' '.join(text)
        test_corpus.append(text)    
    print('Tweets have been cleaned')
    tfidf_matrix_test = tfidf.transform(test_corpus).toarray()
    X_test = svd.transform(tfidf_matrix_test)    
    y_pred = classifier.predict(X_test)
    print('Support for patriots is',np.mean(y_pred))

LA = ['angeles','la','hollywood']
city_support(LA)