import streamlit as st
import pandas as pd
import numpy as np
from functions import remove_stopwords, remove_punctuation, lemmatize_text
import joblib
import nltk
from nltk import TreebankWordTokenizer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
import urllib
import nltk





def page1():
    #st.title("Spam Detection")
   
    
    st.title("Spam Dectection Project Overview")
    
    
    st.image("e-mail-3597088_1280.jpg")

    st.write("Email is one of the quickest means of communication widely utilized by both companies and individuals on a daily basis. Despite its convenience, there are drawbacks associated with using emails, with one of the major issues being 'SPAM.' Spam emails are unsolicited mails sent to a large number of users, serving various purposes such as advertising, phishing, spreading malware, and engaging in other malicious activities. The presence of spam can significantly impact user experience, leading to dissatisfaction. To enhance user experience and mitigate the negative effects of spam, companies that manage email services have implemented filters. These filters work to identify and segregate spam, ensuring that users do not interact with emails that may compromise their computers or expose them to scams. This proactive approach helps safeguard users from potential harm and maintains the integrity of the communication platform.")
    st.image("letter-7102986_1280.jpg")
    st.write("The project's primary objective is to construct a model capable of accurately predicting whether an email is classified as spam or not. The dataset utilized for this project was obtained from the Kaggle platform. The user interface includes a textbox where users can input their email, click the predict button and the model will subsequently provide a prediction indicating whether the email is categorized as spam or not.")
    st.write("Enjoy the app! 😊") #if you want to add some colour

def page2():
    st.title("Spam Detection Application")
    
    #user_input = st.text_input(label='', value='Hi, welcome to my app')
    user_input = st.text_area("**Enter your mail \u2193 :**", height=5)
    st.write("")
    st.write("**Verify your mail \u2193**")
    st.write(user_input)
    #st.pd.DataFrame('user_input', columns='Your email')
    st.write("**Press \u2193 to pedict**")
    if st.button("Predict"):
        #Removing stop words
        rmv_stopwords=remove_stopwords(user_input)
        #Removing Punctuation
        rmv_punct=remove_punctuation(rmv_stopwords)
        #lemmatization
        lemma=lemmatize_text(rmv_punct)
        #st.write(lemma)
        tf = joblib.load('tf_vector.joblib') #Loading the vectorizer
        X=tf.transform([lemma]).toarray()
        l_encoder = joblib.load('label_encoder.joblib') #Loading the encoder
        lr_model = joblib.load('lr_model.joblib') #loading the model
        y=lr_model.predict(X)
        y_original = l_encoder.inverse_transform(y)
        #st.write(y_original)
        if y_original == 'spam':
            st.error("Spam email.")
        else:
            st.success("legitimate email.")
        
    st.subheader("Dont have a mail? Try this mails out.")
    st.write("Spam email example")
    st.write("Congratulations! You have been selected as a winner of our exclusive prize giveaway! You have a limited time to claim your prize, so act now. 🎉 Your Prize:   -$1,000 cash reward   -All-expenses-paid vacation  - Free latest gadgetsTo claim your prize, Act fast  before this exclusive offer expires! Don't miss out on this once-in-a-lifetime opportunity.")
    st.write("Normal email example")
    st.write("I hope this email finds you well. Just a quick reminder that we have a team meeting scheduled for tomorrow at 10 AM in the conference room. We'll be discussing the upcoming project and addressing any questions or concerns. Looking forward to your participation.")
        
    # st.header('Wordcloud of  top twenty prevalent **words** on spam mail **training data**')
    # st.image("image_streamlit.png")
        

selected_page = st.sidebar.radio("Select a page", ["Project Overview", "Application"])

# Display the selected page
if selected_page == "Project Overview":
    page1()

elif selected_page == "Application":
    page2()
    