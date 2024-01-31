#importing Libraries 
import streamlit as st
import pandas as pd
import numpy as np
from functions import remove_stopwords, remove_punctuation, lemmatize_text
import joblib
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import TreebankWordTokenizer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import urllib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression



st.title("Spam Detection Application")
user_input = st.text_area("**Enter your email \u2193 :**", height=5)
st.write("**Press \u2193 to pedict**")
button_clicked = st.button("Predict", key="predict_button", kwargs={"style": "background-color: red; color: red"})

if button_clicked:
    # Removing stop words
    rmv_stopwords = remove_stopwords(user_input)
    # Removing Punctuation
    rmv_punct = remove_punctuation(rmv_stopwords)
    # Lemmatization
    lemma = lemmatize_text(rmv_punct)
    
    # Your further processing code...
    tf = joblib.load('tf_vector.joblib')  # Loading the vectorizer
    X = tf.transform([lemma]).toarray()
    l_encoder = joblib.load('label_encoder.joblib')  # Loading the encoder
    lr_model = joblib.load('lr_Model.joblib')  # loading the model
    y = lr_model.predict(X)
    y_original = l_encoder.inverse_transform(y)

    if y_original == 'spam':
        st.error("Spam email.")
    else:
        st.success("Legitimate email.")

st.subheader("Dont have an email? Try this emails samples out.")
st.write("Spam email example")
st.write("Congratulations! You have been selected as a winner of our exclusive prize giveaway! You have a limited time to claim your prize, so act now. 🎉 Your Prize:   -$1,000 cash reward   -All-expenses-paid vacation  - Free latest gadgetsTo claim your prize, Act fast  before this exclusive offer expires! Don't miss out on this once-in-a-lifetime opportunity.")
st.write("Normal email example")
st.write("I hope this email finds you well. Just a quick reminder that we have a team meeting scheduled for tomorrow at 10 AM in the conference room. We'll be discussing the upcoming project and addressing any questions or concerns. Looking forward to your participation.")
        
    