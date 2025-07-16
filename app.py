import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from PIL import Image

# Load word_to_index and index_to_word
with open("word_to_idx.pkl", 'rb') as file:
    word_to_index = pickle.load(file)

with open("idx_to_word2.pkl", 'rb') as file:
    index_to_word = pickle.load(file)

# Load the models
@st.cache_resource
def load_models():
    decoder_model = load_model('model_14.h5')
    base_model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
    encoder_model = Model(base_model.input, base_model.layers[-2].output)
    return decoder_model, encoder_model

decoder_model, encoder_model = load_models()

# Preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Encode image
def encode_image(img):
    img = preprocess_image(img)
    features = encoder_model.predict(img)
    return features

# Generate caption
def generate_caption(photo):
    in_text = "startseq"
    for i in range(38):
        sequence = [word_to_index.get(w, 0) for w in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=80, padding='post')
        yhat = decoder_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    caption = ' '.join(in_text.split()[1:-1])
    return caption

# Streamlit UI
st.title("🖼️ Image Caption Generator")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Encoding & generating caption..."):
        photo = encode_image(img).reshape((1, 2048))
        caption = generate_caption(photo)

    st.success("✅ Caption generated:")
    st.markdown(f"**📌 {caption}**")
