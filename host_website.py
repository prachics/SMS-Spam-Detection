import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import sklearn
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load the saved TF-IDF vectorizer and the trained model
tfidf = pickle.load(open('/home/chinmay/Desktop/prachi/ML_hackathon/vectorizer.pkl', 'rb'))
model = pickle.load(open('/home/chinmay/Desktop/prachi/ML_hackathon/model.pkl', 'rb'))

# Streamlit app title
st.title("SMS Spam Classifier")

# User input
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the input
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict using the model
    result = model.predict(vector_input)[0]
    
    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
