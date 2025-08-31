import nltk
import os

nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.append(nltk_data_dir)

# Download required packages if missing
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)


import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


ps = PorterStemmer()

def transform_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Tokenize
    text = nltk.word_tokenize(text)
    # 3. Remove non-alphanumeric
    y = [i for i in text if i.isalnum()]
    # 4. Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    # 5. Stemming
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load vectorizer & model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# App UI
st.title('Email/SMS Spam Classifier')
input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]

        # Force "Not Spam" output if desired
       # 4. Display
        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
