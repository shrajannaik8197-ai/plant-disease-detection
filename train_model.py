import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset path
dataset_path = "datasets/PlantVillage"

# Image settings
img_height = 128
img_width = 128
batch_size = 32

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get class names
class_names = train_dataset.class_names

print("Number of classes:", len(class_names))

# Improve dataset performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

# Build CNN model
model = models.Sequential([
    
    # Normalize pixel values
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # First CNN layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    # Second CNN layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    # Third CNN layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    # Convert data to one dimension
    layers.Flatten(),

    # Neural network layer
    layers.Dense(128, activation='relu'),

    # Dropout to prevent overfitting
    layers.Dropout(0.3),

    # Output layer
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Display model structure
model.summary()

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_dataset)

print("\nModel Evaluation Results:")
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Save the trained model
model.save("model/plant_disease_model.keras")

print("\nModel saved successfully!")