import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Custom CSS for UI enhancements
st.markdown(
    """
    <style>
    .title {
        color: #FFFFFF;  /* White text */
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        background-color: #8B4513;  /* Dark brown background */
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stApp {
        background-color: #2E2E2E;  /* Dark gray background */
    }
    .stTextInput>div>div>input {
        background-color: #3E3E3E;  /* Darker gray for input */
        color: #FFFFFF;  /* White text */
    }
    .stSlider>div>div>div>div {
        background-color: #8B4513;  /* Dark brown for slider */
    }
    .stButton>button {
        background-color: #8B4513;  /* Dark brown for button */
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #A0522D;  /* Lighter brown on hover */
    }
    .stMarkdown {
        color: #FFFFFF;  /* White text for markdown */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load your trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("poetry_lstm_model.h5")

model = load_model()

# Load your dataset (this is where 'df' is defined)
@st.cache_data
def load_data():
    return pd.read_csv("Roman-Urdu-Poetry.csv", usecols=["Poetry"])

df = load_data()

# Load or initialize the tokenizer
@st.cache_resource
def load_tokenizer():
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["Poetry"].tolist())  # Fit tokenizer on your poetry data
    return tokenizer

tokenizer = load_tokenizer()

# Generate Poetry Function
def generate_poetry(seed_text, next_words=50, temperature=0.8):
    generated_text = seed_text.lower()
    for _ in range(next_words):
        tokenized_input = tokenizer.texts_to_sequences([generated_text])
        tokenized_input = pad_sequences(tokenized_input, maxlen=model.input_shape[1], padding='pre')

        predicted_probs = model.predict(tokenized_input, verbose=0)[0]

        # Apply temperature scaling
        predicted_probs = np.log(predicted_probs + 1e-8) / temperature
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))  # Normalize

        # Sample a word based on the probabilities
        predicted_word_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        predicted_word = tokenizer.index_word.get(predicted_word_index, "")

        if not predicted_word:
            break

        generated_text += " " + predicted_word

    return generated_text

# Streamlit UI
st.markdown('<h1 class="title">NeuralNazm - AI Meets Structured Poetry</h1>', unsafe_allow_html=True)

# Seed text input
seed_text = st.text_input("Enter seed text for poetry generation:")

# Temperature slider for creativity
temperature = st.slider("Select Temperature", 0.1, 2.0, 0.8)

# Button to trigger poetry generation
if st.button("Generate Poetry"):
    if seed_text:
        generated_poetry = generate_poetry(seed_text, temperature=temperature)
        st.write("📝 Generated Poetry:")
        st.write(generated_poetry)
    else:
        st.write("Please enter a seed text.")