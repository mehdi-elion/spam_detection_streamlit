import re

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# create a header
st.write("# Spam Detection Engine")

# adding text input
message_text = st.text_input("Enter a message for spam evaluation")


# loading the model
def preprocessor(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    return text


model = joblib.load("spam_classifier.joblib")
class_names = list(model.classes_)


# Generating and Displaying Predictions
def classify_message(model, message):
    label = model.predict([message])[0]
    probs = model.predict_proba([message])
    return {"label": label, "probability": probs[0][class_names.index(label)]}


if message_text != "":
    result = classify_message(model, message_text)
    st.write(result)


# explaining predictions with lime
explain_pred = st.button("Explain Predictions")

if explain_pred:
    with st.spinner("Generating explanations"):
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(
            message_text, model.predict_proba, num_features=10
        )
        components.html(exp.as_html(), height=800)
