import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pickle import load
from nltk.tokenize import word_tokenize
import pandas as pd
import spacy
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
nltk.download('stopwords')



def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def cleaning(text):
    corpus = []
    text = decontracted(text)
    text = text.lower()                              #lowering the text
    text = re.sub(r'@\S+','',text)                   #Removed @mentions
    text = re.sub(r'#\S+','',text)                   #Remove the hyper link
    text = re.sub(r'RT\S+','',text)                  #Removing ReTweets
    text = re.sub(r'https?\S+','',text)              #Remove the hyper link
    text = re.sub('[^a-z]',' ',text)              #Remove the character other than alphabet
    text = text.split()
    text=[ps.stem(word) for word in text if word not in stopwords.words('english')]
    text=' '.join(text)
    corpus.append(text)
    return corpus

def predict(input_msg):
    vectorizer = load(open('pickle/countvectorizer.pkl','rb'))
    
    classifier = load(open('pickle/model.pkl','rb'))
    
    clean_text = cleaning(input_msg)
    
    clean_text_encoded = vectorizer.transform(clean_text)
    
    future_text = clean_text_encoded.toarray()
    
    prediction = classifier.predict(future_text)
    
    return prediction

def main():
    st.image("images.jpg", use_column_width=True)
    st.header("Enter any tweet to check the sentiment positive or negative")
    input_msg = st.text_input("")
    prediction = predict(input_msg)

    if(input_msg):
        st.subheader('Prediction')
        if prediction == 1:
            st.write('Positive')
        else:
            st.write('Negative')

if __name__ == '__main__' :
    main()