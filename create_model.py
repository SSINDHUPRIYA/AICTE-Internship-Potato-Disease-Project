import tensorflow as tf
import os

# Define model path
dataset_folder = "C:/Users/Admin/Documents/potato disease project/dataset"
model_path = os.path.join(dataset_folder, "trained_plant_disease_model.keras")

# Create a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile and save the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.save(model_path)

print(f"✅ Model successfully saved to: {model_path}")
