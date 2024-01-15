import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('oralcancer')

@st.cache_data
def preprocess_image(image, target_size):
    # Convert BytesIO to Image
    img = Image.open(image).convert("RGB")
    img = np.array(img)
    
    # Resize the image
    img = cv2.resize(img, target_size)
    # Add more image preprocessing steps as needed
    return img

def predict_class(image, target_size):
    # Preprocess and make predictions
    image = preprocess_image(image, target_size)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    return predicted_class

def main():
    # Set a custom background color
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f8ff;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Streamlit app title with custom styling
    st.title("Oral Cancer Detection App")
    st.markdown(
        """
        <p style='text-align: center; color: #4682B4; font-size: 20px; font-weight: bold;'>
            Detect Oral Cancer from Dental Images
        </p>
        """,
        unsafe_allow_html=True
    )

    # File uploader for image selection
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        
        # Preprocess and make predictions
        target_size = (64, 64)
        predicted_class = predict_class(uploaded_file, target_size)

        # Display the predicted class label with custom styling
        class_labels = {
            0: 'Negative (Related to OSCC)',
            1: 'Positive (Related to NORMAL)'
        }
        predicted_label = class_labels[predicted_class]
        st.subheader("Prediction:")
        st.markdown(
            f"<p style='color: #4682B4; font-size: 18px; font-weight: bold;'>Predicted Class: {predicted_label}</p>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    # Run the Streamlit app
    main()
