import tensorflow as tf
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model(
    "model/plant_disease_model.keras"
)


# Get class names
dataset_path = "datasets/PlantVillage"

class_names = sorted([
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
])
print("Number of classes:", len(class_names))
print("Classes:", class_names)
print("Model output shape:", model.output_shape)

# Path of test image
img_path = "test_images/leaf2.jpg"

# Load image
img = tf.keras.utils.load_img(
    img_path,
    target_size=(128, 128)
)


img_array = tf.keras.utils.img_to_array(img)


img_array = tf.expand_dims(img_array, 0)


predictions = model.predict(img_array)


predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]

confidence = 100 * np.max(predictions[0])

# Make disease name more readable
clean_disease_name = predicted_class.replace("___", " - ").replace("_", " ")

print("\n==============================")
print("     PLANT DISEASE RESULT")
print("==============================")
print("Disease:", clean_disease_name)
print("Confidence:", round(confidence, 2), "%")
print("==============================")
