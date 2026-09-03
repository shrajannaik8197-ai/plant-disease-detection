import tensorflow as tf

dataset_path = "datasets/PlantVillage"

img_height = 128
img_width = 128
batch_size = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_dataset.class_names

print("Number of classes:", len(class_names))
print("Classes:")
print(class_names)

print("\nDataset loaded successfully!")