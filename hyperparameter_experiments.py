import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10
(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

# Keep only cats (3) and dogs (5)
train_mask = np.isin(y_train_original.flatten(), [3, 5])
test_mask = np.isin(y_test_original.flatten(), [3, 5])

x_train = x_train_original[train_mask].astype("float32") / 255.0
x_test = x_test_original[test_mask].astype("float32") / 255.0

# Encode cat = 0 and dog = 1
y_train_original = y_train_original.flatten()[train_mask]
y_test_original = y_test_original.flatten()[test_mask]

y_train = (y_train_original == 5).astype("float32")
y_test = (y_test_original == 5).astype("float32")


def build_model(first_layer_size, second_layer_size, learning_rate):
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Flatten(),
        layers.Dense(first_layer_size, activation="relu"),
        layers.Dense(second_layer_size, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# Three configurations to compare
experiments = [
    {
        "name": "Smaller network",
        "first_layer": 64,
        "second_layer": 32,
        "learning_rate": 0.001,
        "batch_size": 32,
    },
    {
        "name": "Larger network",
        "first_layer": 256,
        "second_layer": 128,
        "learning_rate": 0.001,
        "batch_size": 32,
    },
    {
        "name": "Lower learning rate",
        "first_layer": 128,
        "second_layer": 64,
        "learning_rate": 0.0005,
        "batch_size": 32,
    },
]

results = []

for experiment in experiments:
    print(f"\nRunning: {experiment['name']}")

    tf.keras.utils.set_random_seed(42)

    model = build_model(
        experiment["first_layer"],
        experiment["second_layer"],
        experiment["learning_rate"],
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=experiment["batch_size"],
        validation_split=0.2,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    result = {
        "name": experiment["name"],
        "first_layer": experiment["first_layer"],
        "second_layer": experiment["second_layer"],
        "learning_rate": experiment["learning_rate"],
        "batch_size": experiment["batch_size"],
        "final_validation_accuracy": history.history["val_accuracy"][-1],
        "test_accuracy": test_accuracy,
    }
    results.append(result)

    print(f"Final validation accuracy: {result['final_validation_accuracy']:.2%}")
    print(f"Test accuracy: {result['test_accuracy']:.2%}")

# Save the comparison results to a CSV file
with open("hyperparameter_results.csv", "w", newline="") as csv_file:
    fieldnames = results[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to 'hyperparameter_results.csv'")