"""
Dog vs Cat Image Classifier
Description: Build a simple neural network to classify dog and cat images.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))

# Load the CIFAR-10 dataset
(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

# CIFAR-10 labels: cat = 3, dog = 5
cat_class = 3
dog_class = 5

# Keep only cats and dogs
train_mask = np.isin(y_train_original.flatten(), [cat_class, dog_class])
test_mask = np.isin(y_test_original.flatten(), [cat_class, dog_class])

x_train = x_train_original[train_mask]
y_train_original = y_train_original[train_mask].flatten()

x_test = x_test_original[test_mask]
y_test_original = y_test_original[test_mask].flatten()

# Encode labels: cat = 0, dog = 1
y_train = (y_train_original == dog_class).astype("float32")
y_test = (y_test_original == dog_class).astype("float32")

# Normalize pixel values from 0–255 to 0–1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nData loaded and prepared")
print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)
print("Training cats:", np.sum(y_train == 0))
print("Training dogs:", np.sum(y_train == 1))
print("Test cats:", np.sum(y_test == 0))
print("Test dogs:", np.sum(y_test == 1))

# Display a sample of training images
plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    label = "Dog" if y_train[i] == 1 else "Cat"
    plt.title(label)
    plt.axis("off")

plt.tight_layout()
plt.show()

# Build the neural network
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

# Configure how the model learns
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\nModel architecture:")
model.summary()

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# Evaluate it on the separate test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2%}")

# Plot training and validation metrics
epochs_range = range(1, len(history.history["accuracy"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
axes[0].plot(epochs_range, history.history["accuracy"], label="Training accuracy")
axes[0].plot(epochs_range, history.history["val_accuracy"], label="Validation accuracy")
axes[0].set_title("Training and Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(epochs_range, history.history["loss"], label="Training loss")
axes[1].plot(epochs_range, history.history["val_loss"], label="Validation loss")
axes[1].set_title("Training and Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved training plot to 'training_history.png'")

# Predict the first 10 test images
probabilities = model.predict(x_test[:10], verbose=0).flatten()
predicted_labels = (probabilities >= 0.5).astype(int)

plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])

    actual = "Dog" if y_test[i] == 1 else "Cat"
    predicted = "Dog" if predicted_labels[i] == 1 else "Cat"
    confidence = probabilities[i] if predicted_labels[i] == 1 else 1 - probabilities[i]

    color = "green" if actual == predicted else "red"
    plt.title(
        f"Pred: {predicted} ({confidence:.0%})\nActual: {actual}",
        color=color,
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved sample predictions to 'sample_predictions.png'")

# Save the complete Keras model
model.save("dog_cat_classifier.keras")

# Save only the model weights
model.save_weights("dog_cat_classifier.weights.h5")

# Export a SavedModel for serving or deployment
model.export("dog_cat_savedmodel")

print("\nSaved model files:")
print("- dog_cat_classifier.keras")
print("- dog_cat_classifier.weights.h5")
print("- dog_cat_savedmodel/")

# Reload the complete Keras model and evaluate it
reloaded_model = keras.models.load_model("dog_cat_classifier.keras")
reloaded_loss, reloaded_accuracy = reloaded_model.evaluate(
    x_test,
    y_test,
    verbose=0,
)

print(f"\nReloaded model test loss: {reloaded_loss:.4f}")
print(f"Reloaded model test accuracy: {reloaded_accuracy:.2%}")