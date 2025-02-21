import streamlit as st
import numpy as np
import tensorflow as tf
import base64
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to set background using Base64
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image (Ensure 'background.jpg' is in your project folder)
set_background("title_banner.jpg")

# Custom Styling
st.markdown("""
    <style>
        .title-container {
            font-size: 35px;
            font-weight: bold;
            text-align: center;
            color: black;
            padding: 20px;
            background: linear-gradient(135deg, #D7B19D, #BCAAA4, #8D6E63);
            border-radius: 12px;
            box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.3);
            animation: fadeIn 1.5s ease-in-out;
        }

        [data-testid=stSidebar] {
        background-color: #8B4513;  
         }
         
           [data-testid=stSidebar] .sidebar-title {
        color: #FFFFFF;
        font-size: 15px;
    }
     [data-testid=stSidebar] .sidebar-separator {
        border-bottom: 2px solid #FFFFFF;
        margin: 15px 0;
        width: 100%;
    }
        .stTextInput > div > div > input {
            background-color: #E0C3A3;
            color: black;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px;
            border: 1px solid #A1887F;
        }

        .stButton > button {
            background: linear-gradient(135deg, #A0522D, #D2691E);
            color: black;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 25px;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease-in-out;
            box-shadow: 2px 4px 6px rgba(0, 0, 0, 0.2);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #D2691E, #FF8C00);
            transform: scale(1.05);
        }

        .generated-text {
            background: linear-gradient(135deg, #D7B19D, #BCAAA4);
            padding: 10px;
            border-left: 5px solid #D2691E;
            border-radius: 8px;
            font-size: 18px;
            color: black;
            line-height: 1.8;
            box-shadow: 2px 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Fade-in Animation */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("poetry_lstm_model.h5")

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Roman-Urdu-Poetry.csv", usecols=["Poetry"])

df = load_data()

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["Poetry"].tolist())
    return tokenizer

tokenizer = load_tokenizer()

# Function to generate poetry
def generate_poetry(seed_text, next_words=50, temperature=0.8):
    generated_text = seed_text.lower()
    for _ in range(next_words):
        tokenized_input = tokenizer.texts_to_sequences([generated_text])
        tokenized_input = pad_sequences(tokenized_input, maxlen=model.input_shape[1], padding='pre')

        predicted_probs = model.predict(tokenized_input, verbose=0)[0]

        # Apply temperature scaling
        predicted_probs = np.log(predicted_probs + 1e-8) / temperature
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

        # Sample a word based on the probabilities
        predicted_word_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        predicted_word = tokenizer.index_word.get(predicted_word_index, "")

        if not predicted_word:
            break

        generated_text += " " + predicted_word

    return generated_text

# Streamlit UI
st.markdown("<div class='title-container'>📜 NeuralNazm - AI Meets Structured Poetry</div>", unsafe_allow_html=True)
generated_poetry=""
# Layout with columns
col1, col2 = st.columns([2, 2])
with col1:
    st.markdown("<h4 style='color: black;'>📜 Enter Seed Text</h4>", unsafe_allow_html=True)
    seed_text = st.text_input("", key="seed_text")

    if st.button("✨Generate Poetry"):
        if seed_text:
            generated_poetry = generate_poetry(seed_text, temperature=st.session_state["temperature"])
        else:
            st.error("⚠️ Please enter a seed text.")
    with col2:
        if seed_text:
            st.markdown("<h4 style='color: black;'>✨Generated Poetry</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='generated-text'>{generated_poetry}</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("āvāz de ke dekh lo shāyad vo mil hī jaae ")
st.sidebar.title("varna ye umr bhar kā safar rāegāñ to hai")
st.sidebar.markdown('<p class="sidebar-separator"></p>', unsafe_allow_html=True)

# Move Temperature Control to Sidebar
st.sidebar.markdown("<h4 style='color:black;text-align: center;'>🎨Custom Creativity</h4>", unsafe_allow_html=True)
st.session_state["temperature"] = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8)
st.sidebar.markdown("""<p>
                    Low → Predictable.<br>
                    High → Creative & chaotic.<br>
                    0.8 is a good default value.</p>""", unsafe_allow_html=True)