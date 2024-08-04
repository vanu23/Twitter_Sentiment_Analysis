import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load

# Load the sentiment analysis model
model = load_model('sentiment_model.h5')

# Load the tokenizer
tokenizer = load('tokenizer.joblib')

# Define the maximum sequence length (you should use the same value used during training)
max_length = 200 # Adjust this value based on your training data

# Create a function to predict sentiment

# streamlit run twitter_sentiment_app.py
def predict_sentiment(text):

    # Preprocess the input text: tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=200, padding='post')

    # Predict sentiment
    predictions = model.predict(padded_sequences)

    # Decode the predictions (assuming 0: negative, 1: neutral, 2: positive)
    print(predictions)
    predicted_labels = np.argmax(predictions, axis=1)

    # Map numeric labels back to sentiment labels
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_sentiments = [label_mapping[label] for label in predicted_labels]

    return predicted_sentiments[0]

# Set up Streamlit app
st.title("Sentiment Analysis App")

# Text input for user
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input:
        # Call the prediction function
        predicted_sentiment = predict_sentiment(user_input)

        # Deter`mine the sentiment based on the prediction
        # Display the result
        st.write(f"Sentiment: {predicted_sentiment}")
else:
    st.warning("Please enter some text for analysis.")