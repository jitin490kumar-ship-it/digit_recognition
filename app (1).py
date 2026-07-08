
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Title
st.title("Digit Recognition using CNN (MNIST)")
st.write("Upload an image of a handwritten digit (28x28 grayscale).")

# Load trained model
model = tf.keras.models.load_model("mnist_cnn.h5")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")

    # Invert colors (white digit on black background)
    image = ImageOps.invert(image)

    # Resize to 28x28
    image = image.resize((28, 28))

    # Show uploaded image
    st.image(image, caption="Processed Image", width=150)

    # Convert image to numpy array
    img = np.array(image)

    # Normalize pixel values
    img = img / 255.0

    # Reshape for CNN
    img = img.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img)

    # Predicted digit
    digit = np.argmax(prediction)

    # Confidence
    confidence = np.max(prediction) * 100

    # Display results
    st.success(f"Predicted Digit: {digit}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show all probabilities
    st.subheader("Prediction Probabilities")

    for i in range(10):
        st.write(f"Digit {i}: {prediction[0][i]*100:.2f}%")
