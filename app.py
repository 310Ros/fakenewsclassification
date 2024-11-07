# libraries
import re
import json
import string
import pickle
import wordninja
import math, nltk
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn import feature_extraction
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from flask import Flask, request, render_template
# Warning
import warnings
warnings.filterwarnings('ignore')
nltk.download('all')


#load model and tfidf vect
tfidf_vect = pickle.load(open('tfidf_vect.pkl', 'rb'))
stc_model = pickle.load(open('./saved_model/stc_model.pkl', 'rb'))


application = Flask(__name__)

app=application



#text cleaning
def html_references(tweets):
    texts = tweets
    # remove url - references to websites
    url_remove  = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    texts  = re.sub(url_remove, '', texts)
    # remove common html entity references in utf-8 as '&lt;', '&gt;', '&amp;'
    entities_remove = r'&amp;|&gt;|&lt'
    texts = re.sub(entities_remove, "", texts)
    # split into words by white space
    words = texts.split()
    #convert to lower case
    words = [word.lower() for word in words]
    return " ".join(words)

def decontraction(tweet):
    # specific
    tweet = re.sub(r"won\'t", " will not", tweet)
    tweet = re.sub(r"won\'t've", " will not have", tweet)
    tweet = re.sub(r"can\'t", " can not", tweet)
    tweet = re.sub(r"don\'t", " do not", tweet)

    tweet = re.sub(r"can\'t've", " can not have", tweet)
    tweet = re.sub(r"ma\'am", " madam", tweet)
    tweet = re.sub(r"let\'s", " let us", tweet)
    tweet = re.sub(r"ain\'t", " am not", tweet)
    tweet = re.sub(r"shan\'t", " shall not", tweet)
    tweet = re.sub(r"sha\n't", " shall not", tweet)
    tweet = re.sub(r"o\'clock", " of the clock", tweet)
    tweet = re.sub(r"y\'all", " you all", tweet)
    # general
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"n\'t've", " not have", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'d've", " would have", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'ll've", " will have", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    return tweet

def filter_punctuations_etc(tweets):
    words = tweets.split()
    # prepare regex for char filtering
    re_punc = re.compile( '[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    words = [re_punc.sub('', w) for w in words]
    # filter out non-printable characters
    re_print = re.compile( '[^%s]' % re.escape(string.printable))
    words = [re_print.sub(' ', w) for w in words]
    return " ".join(words)

def separate_alphanumeric(tweets):
    words = tweets
    # separate alphanumeric
    words = re.findall(r"[^\W\d_]+|\d+", words)
    return " ".join(words)

def cont_rep_char(text):
    tchr = text.group(0)

    if len(tchr) > 1:
        return tchr[0:2] # take max of 2 consecutive letters

def unique_char(rep, tweets):
    substitute = re.sub(r'(\w)\1+', rep, tweets)
    return substitute


def split_attached_words(tweet):
    words = wordninja.split(tweet)
    return" ".join(words)

def stopwords_shortwords(tweet):
    # filter out stop words
    words = tweet.split()
    stop_words = set(stopwords.words( 'english' ))
    words = [w for w in words if not w in stop_words]
    # filter out short tokens
    for word in words:
        if word.isalpha():
            words = [word for word in words if len(word) > 1 ]
        else:
            words = [word for word in words]
    return" ".join(words)



#flask routing

@app.route('/')
def index(): 
    return render_template('index.html')


@app.route('/home')
def home(): 
    return render_template('index.html')


@app.route('/service')
def service(): 
    return render_template('service.html')


@app.route('/fakenewsclf',methods=['POST'])
def fakenewsclf():

    inputtext = request.form['inputtxt']
    data = [inputtext]
    data_clean = pd.DataFrame({'text': data})
    data_clean['clean_text'] = data_clean['text'].apply(lambda x : html_references(x))
    data_clean['clean_text'] = data_clean['clean_text'].apply(lambda x : decontraction(x))
    data_clean['clean_text'] = data_clean['clean_text'].apply(lambda x : filter_punctuations_etc(x))
    data_clean['clean_text'] = data_clean['clean_text'].apply(lambda x : separate_alphanumeric(x))
    data_clean['clean_text'] = data_clean['clean_text'].apply(lambda x : unique_char(cont_rep_char, x))
    data_clean['clean_text'] = data_clean['clean_text'].apply(lambda x : split_attached_words(x))
    data_clean['clean_text'] = data_clean['clean_text'].apply(lambda x : stopwords_shortwords(x))
    inputtext = data_clean['clean_text'].tolist()
    inputtxt = inputtext[0]
    inputtext = tfidf_vect.transform(inputtext).toarray()
    my_pred = stc_model.predict(inputtext)
    my_pred = my_pred
    print(my_pred)
    if my_pred[0] == 1:
        my_pred = 'NEWS CLASSIFY AS : ' + 'FAKE'
        return render_template('service.html', inputvalue=inputtxt, pred_class=my_pred)
    else:
        my_pred = 'NEWS CLASSIFY AS : ' + 'REAL'
        return render_template('service.html', inputvalue=inputtxt, pred_class=my_pred)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



