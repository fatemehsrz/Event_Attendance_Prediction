from sys import argv
from sys import exit

import re
import csv
import nltk
import numpy as np

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import nltk.metrics
import collections

from keras import backend as K

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics

from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec

from numpy import array
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

import collections
from collections import defaultdict


#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def getStopWordList(stopWordFile):
    stopwords = []
    stopwords.append("AT_USER")
    stopwords.append("URL")

    with open(stopWordFile, 'r') as f:
        reader = csv.reader(f)
        for w in reader:

            stopwords.append(w[0])

    return stopwords


def getFeatureVector(tweet, stopWordFile):
    features = []

    stop_words = getStopWordList(stopWordFile)

    words = tweet.split()
    for w in words:

        w = replaceTwoOrMore(w)

        #strip digits
        w = w.strip('0123456789')

        #strip punctuation
        w = w.strip('\'"!?,.')

        if (w == ""):
            continue
        elif(w in stop_words):
            
            continue
        else:
            features.append(w.lower())

    return features


def processRow(tweet):
    
    tweet = tweet
  
    #Lower case
    tweet.lower()
    #convert any url to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert any @Username to "AT_USER"
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub('[\n]+', ' ', tweet)
    
    #Remove not alphanumeric symbols white spaces
    #tweet = re.sub(r'[^\w]', ' ', tweet)
    
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    
    #Remove :( or :)
    #tweet = tweet.replace(':)','')
    #tweet = tweet.replace(':(','')
    
    #trim
    #tweet = tweet.strip('\'"')

    return tweet



def read_tweets(tweet_file):

    features = []
    tweets = []
    user_ids=[]
    labels=[]

    with open(tweet_file,'r') as csv_file:
        
        csv_reader = csv.reader(csv_file)
        
        count=0
        
        for row in csv_reader:
            
            user_ids.append(row[0])
            labels.append(int(row[1]))
            
          
            clean_tweet =processRow(row[2])
             
            if len(clean_tweet) < 2:
                print ("Malformed Data")
                continue
                
            features = getFeatureVector(clean_tweet, stop_words_file_name)

            #tweets.append(( [f.strip("\'") for f in features], row[1], row[0]))
            
            tweets.append([f.strip("\'") for f in features])
            
            #user_tweet[row[0]]= [f.strip("\'") for f in features]
            
            count+=1
            
    return tweets, user_ids, labels


tweet_file= './data/vfest_tweets2.csv'
stop_words_file_name='./data/stopwords.txt'
tweets, user_ids, labels = read_tweets(tweet_file)

print(len(labels), len(user_ids), len(labels))



docs=[]
docs_all=[]


print('==========================================================================================')

for i in range(len(tweets)):
    tweet_str = ' '.join(tweets[i])
    docs.append(tweet_str) 
    docs_all.extend(tweet_str)
    

vocab=len(set(docs_all))

print('vocabulary_size:',vocab)

tokenizer = Tokenizer(num_words=vocab )
tokenizer.fit_on_texts(docs)

X = tokenizer.texts_to_matrix(docs, mode='count') # 'count'mode='freq'  mode='count'

print('tweet matrix shape:',X.shape)
#print('bag of work by counting:', X[0])


## concatenation ##########################################################

emb_size=128

vec_list= defaultdict()
vec_float=[]


for size in [4,8, 32, 64, 128, 256]: 

    embfile = open( "./emb_vectors/vfest_%d.harp"%size, 'r') 
    
    for line in embfile:
        a=line.strip('\n').split(' ')
        
        user_id=a[0]
        
        vec_str= a[1:-1]
        
        vec_float=[]
        for j in vec_str:
            
            vec_float.append(float(j))
             
        #print(len(vec_str), len(vec_float))    
        
        vec_list[str(user_id)]=vec_float
        
            
    ######################################################
    
    X_con=[]
    y=[]
    
    for i in range(X.shape[0]):
        
        id= str(user_ids[i])
                                         
        con= np.concatenate((X[i], vec_list[id]), axis=0)
        
        X_con.append(con)
            
        y.append(int(labels[i]))
    
      
    train_size= int(0.8*len(X_con))
    
    print(train_size)
    
    
    
    X_con=np.asarray(X_con)
    y= np.asarray(y)
    
    print('feature matrix with network emb',X_con.shape)
    
    X_train=X_con [0:train_size]
    y_train = y[0:train_size]
    
    X_test= X_con[train_size:]
    y_test=y[train_size:]
    
    print( X_train.shape, y_train.shape,  X_test.shape, y_test.shape )
    
    n_dim = X_test.shape[1]
    print('feature size:',n_dim)
    
    # define network
    
    accuracies=[]
    precisions=[]
    recalls=[]
    fscores=[]
     
    for i in range(5):
        
        K.clear_session()
        
        model = Sequential()
        model.add(Dense(n_dim, input_shape=(n_dim,), activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(int(n_dim*0.5), activation='relu'))  #200
        model.add(Dropout(0.4))
        
    
        model.add(Dense(30, activation='relu'))  #60
        #model.add(Dropout(0.1))
         
        model.add(Dense(1, activation='sigmoid'))
        
        # compile network
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(X_train, y_train, epochs=150, verbose=2)
        # evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=2)
        
        preds = model.predict(X_test)
        
        y_pred=[]
        
        for i in preds:
            if i<0.5:
                #print('0')
                y_pred.append(0)
            else:
                #print('1') 
                y_pred.append(1)   
        
        
        
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import f1_score
        
        
        precision= precision_score(y_test, y_pred)
        recall= recall_score(y_test, y_pred)
        acc2= accuracy_score(y_test, y_pred)
        f_score= f1_score(y_test, y_pred)
    
        print( acc2, precision, recall)
           
        #print('Test Accuracy: %.3f' % acc)
        
        accuracies.append(acc2)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(f_score)
        
    result_file= open('./emb_vectors/result_emb%d.txt'%size, 'w')  
      
    for i in range(len(accuracies)):
        #print('%.3f' %i)
        
        print('accuracy %.3f'%accuracies[i], ' precisions %.3f' % precisions[i], ' recalls %.3f'%recalls[i], ' fscores %.3f' %fscores[i])
    
        result_file.write( str(' accuracy %.3f'%accuracies[i])+ str(' precisions %.3f' % precisions[i])+ str(' recalls %.3f'%recalls[i])+ str(' fscores %.3f' %fscores[i]) ) 
        result_file.write('\n')

    result_file.close()
