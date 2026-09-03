import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Title
st.title("🌿 Plant Disease Detection System")
st.write("Upload a plant leaf image to detect the disease.")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/plant_disease_model.keras"
    )

model = load_model()

# Get class names
dataset_path = "datasets/PlantVillage"

class_names = sorted([
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
])

# Upload image
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    st.image(
        uploaded_file,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    # Prediction button
    if st.button("🔍 Detect Disease"):

        # Load and preprocess image
        image = tf.keras.utils.load_img(
            uploaded_file,
            target_size=(128, 128)
        )

        image_array = tf.keras.utils.img_to_array(image)

        image_array = tf.expand_dims(image_array, 0)

        # Prediction
        predictions = model.predict(image_array)

        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        confidence = 100 * np.max(predictions[0])

        # Clean disease name
        clean_disease_name = predicted_class.replace(
            "___", " - "
        ).replace("_", " ")

        # Display result
        st.success("Prediction Completed!")

        st.subheader("🌿 Prediction Result")

        st.write(
            f"**Disease:** {clean_disease_name}"
        )

        st.write(
            f"**Confidence:** {confidence:.2f}%"
        )

        # Progress bar
        st.progress(int(confidence))