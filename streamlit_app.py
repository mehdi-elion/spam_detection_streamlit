import joblib
import re
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st


# create a header
st.write("# Spam Detection Engine")

# adding text input
message_text = st.text_input("Enter a message for spam evaluation")


# loading the model
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) 
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

model = joblib.load('spam_classifier.joblib')
classes = list(model.classes_)


# Generating and Displaying Predictions
def classify_message(model, message):
    label = model.predict([message])[0]
    probs = model.predict_proba([message])
    return {
        'label': label, 
        'probability': probs[0][classes.index(label)]
    }


if message_text != '':
  result = classify_message(model, message_text)
  st.write(result)