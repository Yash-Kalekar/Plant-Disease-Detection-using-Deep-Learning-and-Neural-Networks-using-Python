import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Streamlit app title and description
st.title("Image Classification with Keras Model")
st.write("Upload an image to classify it using a pre-trained Keras model.")

# Load the Keras model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# File uploader for uploading images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image: resize and normalize
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert the image to a numpy array and normalize it
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create a batch of size 1 (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make the prediction using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the prediction and confidence score
    st.write(f"**Prediction:** {class_name[2:].strip()}")
    st.write(f"**Confidence Score:** {confidence_score:.2f}")
