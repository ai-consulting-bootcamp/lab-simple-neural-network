# Build Simple Neural Network

> **How you'll submit this lab**
>
> This repo is your lab. Fork it, do the work described below in your fork, then open a pull
> request back into this repository. An AI reviewer will check your PR against `rubric.md` and
> leave feedback directly on the PR. See `README.md` for the full workflow.

**Scenario**  
 In Part 1, you'll build your first neural network to classify images of dogs and cats using a simplified approach. In Part 2, you'll explore how pre-trained models can "see" and "understand" audio data using transformers - discovering how we teach machines to perceive the world around them.

**Learning Objectives**
- [ ] Build and train a simple neural network from scratch
- [ ] Understand the basic components of a neural network (layers, activation functions, loss)
- [ ] Train a model to classify images (dogs vs. cats)
- [ ] Export and save model weights
- [ ] Use pre-trained transformer models for audio classification
- [ ] Understand how models learn to "see" and "understand" different data types

**Estimated Time:** 180-210 minutes (Part 1: 120-150 min, Part 2: 60-60 min)

**Prerequisites:**
- [ ] Basic understanding of Python
- [ ] Familiarity with machine learning concepts (from previous labs)
- [ ] Understanding of train/test split
- [ ] Basic knowledge of neural networks (from lessons)

---

## Introduction

Welcome to your first neural network lab! This lab is divided into two exciting parts:

**Part 1 (Self-Guided):** You'll build a simple neural network to classify images of dogs and cats. You'll receive example code and a list of parameters you can experiment with. This is your chance to explore how neural networks work by tweaking different settings and observing the results.

**Part 2 (Exploration):** You'll work with pre-trained transformer models that can understand audio. This part will help you see how modern AI models are "taught" to perceive and understand different types of data - from images to sounds.

**What you'll build:**
- A neural network that can distinguish between dogs and cats
- Understanding of how neural networks learn
- Experience with pre-trained models and transformers
- Appreciation for how models "see" and "understand"

**Why this matters:**
Neural networks are the foundation of modern AI. Understanding how they work, even at a basic level, is crucial for any AI practitioner. Pre-trained models show us the power of transfer learning - using knowledge gained from one task to solve another.

**Success criteria:**
- [ ] Successfully train a neural network on dog/cat images
- [ ] Achieve reasonable accuracy (>60%) on the test set
- [ ] Export and save model weights
- [ ] Use pre-trained audio classification models
- [ ] Understand the concepts of model training and pre-trained models

---

## Part 1: Self-Guided - Dog vs. Cat Image Classifier

### Background

You'll build a neural network to classify images as either a dog or a cat. We'll use the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes. We'll filter it to use only dogs (class 5) and cats (class 3) to create a binary classification problem.

**Dataset:** CIFAR-10
- **Classes we'll use:** Cat (class 3) and Dog (class 5)
- **Image size:** 32x32 pixels, RGB color
- **Total images:** ~12,000 (6,000 cats + 6,000 dogs)

---

### Step 1: Setting Up the Environment

**Objective:** Install required libraries and set up your project.

**What to do:**
1. Install necessary packages
2. Create your project file
3. Import required libraries

**Required packages:**
```bash
pip install tensorflow numpy matplotlib
```

**Code template:**
```python
"""
Dog vs Cat Image Classifier
Author: [Your Name]
Description: Build a simple neural network to classify dog and cat images
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
print("GPU available:", tf.config.list_physical_devices('GPU'))
```

**Expected outcome:** You should see TensorFlow version information and whether a GPU is available.

**Checkpoint:** Verify that TensorFlow is installed correctly and you can import all required libraries.

---

### Step 2: Loading and Preparing the Data

**Objective:** Load CIFAR-10 dataset and filter to dogs and cats only.

**What to do:**
1. Load the CIFAR-10 dataset
2. Filter to only cats (class 3) and dogs (class 5)
3. Normalize the images
4. Split into training and testing sets
5. Visualize some sample images

**Code template:**
```python
# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

print(f"Full dataset shape - Training: {x_train_full.shape}, Test: {x_test_full.shape}")
print(f"Full dataset labels - Training: {y_train_full.shape}, Test: {y_test_full.shape}")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print("\nCIFAR-10 classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Filter to cats (3) and dogs (5)
cat_class = 3
dog_class = 5

# Training set
train_cat_mask = (y_train_full.flatten() == cat_class)
train_dog_mask = (y_train_full.flatten() == dog_class)
train_mask = train_cat_mask | train_dog_mask

x_train = x_train_full[train_mask]
y_train = y_train_full[train_mask]

# Convert labels: cat=0, dog=1
y_train = np.where(y_train == cat_class, 0, 1)

# Test set
test_cat_mask = (y_test_full.flatten() == cat_class)
test_dog_mask = (y_test_full.flatten() == dog_class)
test_mask = test_cat_mask | test_dog_mask

x_test = x_test_full[test_mask]
y_test = y_test_full[test_mask]

# Convert labels: cat=0, dog=1
y_test = np.where(y_test == dog_class, 1, 0)

print(f"\nFiltered dataset:")
print(f"  Training: {x_train.shape[0]} images")
print(f"  Test: {x_test.shape[0]} images")
print(f"  Image shape: {x_train.shape[1:]}")

# Check class distribution
print(f"\nTraining set - Cats: {(y_train == 0).sum()}, Dogs: {(y_train == 1).sum()}")
print(f"Test set - Cats: {(y_test == 0).sum()}, Dogs: {(y_test == 1).sum()}")

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten images for simple neural network (32x32x3 = 3072)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print(f"\nFlattened shape: {x_train_flat.shape}")

# Visualize some sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    row = i // 5
    col = i % 5
    axes[row, col].imshow(x_train[i])
    label = 'Cat' if y_train[i] == 0 else 'Dog'
    axes[row, col].set_title(f'{label}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
print("\nSaved sample images to 'sample_images.png'")
plt.show()
```

**Expected outcome:** You should have approximately 10,000 training images and 2,000 test images, with roughly equal numbers of cats and dogs.

**Checkpoint:** Verify that:
- Images are loaded and filtered correctly
- Labels are converted to 0 (cat) and 1 (dog)
- Images are normalized to [0, 1]
- Sample images display correctly

---

### Step 3: Building Your First Neural Network

**Objective:** Create a simple neural network architecture.

**What to do:**
1. Define the neural network architecture
2. Understand each component (layers, activation functions)
3. Compile the model with optimizer and loss function

**Code template:**
```python
# Build the neural network
print("\n" + "="*50)
print("BUILDING NEURAL NETWORK")
print("="*50)

# Simple feedforward neural network
model = keras.Sequential([
    # Input layer: 3072 features (32*32*3)
    layers.Dense(128, activation='relu', input_shape=(3072,), name='hidden_layer_1'),
    layers.Dense(64, activation='relu', name='hidden_layer_2'),
    # Output layer: 1 neuron for binary classification (dog or cat)
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

# Compile the model
model.compile(
    optimizer='adam',           # Optimization algorithm
    loss='binary_crossentropy', # Loss function for binary classification
    metrics=['accuracy']        # Metric to track during training
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Count parameters
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")
```

**Explanation:**
- **Dense layers:** Fully connected layers where each neuron connects to all neurons in the next layer
- **ReLU activation:** Rectified Linear Unit - introduces non-linearity, helps model learn complex patterns
- **Sigmoid activation:** Outputs values between 0 and 1, perfect for binary classification
- **Adam optimizer:** Adaptive learning rate optimizer, good default choice
- **Binary crossentropy:** Loss function for binary classification problems

**Expected outcome:** You should see a model summary showing the architecture and parameter count.

**Checkpoint:** Verify that your model is built correctly and you understand what each component does.

---

### Step 4: Training the Model

**Objective:** Train your neural network on the dog/cat images.

**What to do:**
1. Train the model with default parameters first
2. Monitor training progress
3. Observe how loss and accuracy change

**Code template:**
```python
# Train the model
print("\n" + "="*50)
print("TRAINING THE MODEL")
print("="*50)

# Training parameters
EPOCHS = 20
BATCH_SIZE = 32

print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}...")

# Train the model
history = model.fit(
    x_train_flat, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test_flat, y_test),
    verbose=1  # Show progress bar
)

print("\n✓ Training complete!")
```

**Expected outcome:** You should see training progress with loss and accuracy improving over epochs.

**Checkpoint:** Verify that training completes and you can see the final training and validation accuracy.

---

### Step 5: Evaluating the Model

**Objective:** Evaluate model performance on the test set.

**What to do:**
1. Evaluate on the test set
2. Make predictions on sample images
3. Visualize training history
4. Check model performance

**Code template:**
```python
# Evaluate the model
print("\n" + "="*50)
print("EVALUATING THE MODEL")
print("="*50)

test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Make predictions on test set
predictions = model.predict(x_test_flat)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Calculate accuracy manually
correct = (predicted_classes == y_test.flatten()).sum()
total = len(y_test)
print(f"\nManual accuracy check: {correct}/{total} ({correct/total*100:.2f}%)")

# Visualize training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot loss
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("\nSaved training history to 'training_history.png'")
plt.show()

# Test on some sample images
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    row = i // 5
    col = i % 5
    
    # Get prediction
    img = x_test[i].reshape(1, -1)
    pred = model.predict(img, verbose=0)[0][0]
    pred_class = 'Dog' if pred > 0.5 else 'Cat'
    confidence = pred if pred > 0.5 else 1 - pred
    
    # Actual label
    actual = 'Dog' if y_test[i] == 1 else 'Cat'
    correct = '✓' if (pred > 0.5) == (y_test[i] == 1) else '✗'
    
    axes[row, col].imshow(x_test[i])
    axes[row, col].set_title(f'{pred_class} ({confidence:.2f})\nActual: {actual} {correct}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
print("Saved sample predictions to 'sample_predictions.png'")
plt.show()
```

**Expected outcome:** You should achieve test accuracy around 60-75%. The visualizations show how the model learned and some example predictions.

**Checkpoint:** Verify that:
- Test accuracy is calculated
- Training history plots show learning progress
- Sample predictions are displayed

---

### Step 6: Exporting Model Weights

**Objective:** Save the trained model weights for later use.

**What to do:**
1. Save the entire model
2. Save only the weights
3. Verify the files are created

**Code template:**
```python
# Export model weights
print("\n" + "="*50)
print("EXPORTING MODEL WEIGHTS")
print("="*50)

# Create directory for saved models
import os
os.makedirs('saved_models', exist_ok=True)

# Method 1: Save entire model (architecture + weights + optimizer state)
model_path = 'saved_models/dog_cat_classifier_full.h5'
model.save(model_path)
print(f"✓ Saved full model to: {model_path}")
print(f"  File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

# Method 2: Save only weights
weights_path = 'saved_models/dog_cat_classifier_weights.h5'
model.save_weights(weights_path)
print(f"✓ Saved weights only to: {weights_path}")
print(f"  File size: {os.path.getsize(weights_path) / 1024 / 1024:.2f} MB")

# Method 3: Save in SavedModel format (TensorFlow's recommended format)
savedmodel_path = 'saved_models/dog_cat_classifier_savedmodel'
model.save(savedmodel_path)
print(f"✓ Saved in SavedModel format to: {savedmodel_path}/")
print(f"  Directory size: {sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                for dirpath, dirnames, filenames in os.walk(savedmodel_path)
                                for filename in filenames) / 1024 / 1024:.2f} MB")

# Verify we can load the model
print("\nVerifying saved model...")
loaded_model = keras.models.load_model(model_path)
test_loss_loaded, test_acc_loaded = loaded_model.evaluate(x_test_flat, y_test, verbose=0)
print(f"Loaded model test accuracy: {test_acc_loaded:.4f}")
print(f"Original model test accuracy: {test_accuracy:.4f}")
print(f"Match: {'✓' if abs(test_acc_loaded - test_accuracy) < 0.001 else '✗'}")

print("\n✓ Model weights exported successfully!")
```

**Expected outcome:** You should have saved model files in the `saved_models` directory, and be able to load them back.

**Checkpoint:** Verify that:
- Model files are created
- You can load the model back
- Loaded model gives same accuracy

---

### Step 7: Experimentation - Parameters You Can Change

**Objective:** Experiment with different hyperparameters to see how they affect model performance.

**What to do:** Try changing these parameters and observe the results. Document your findings!

**Parameters to Experiment With:**

1. **Number of Hidden Layers**
   - Try: 1 layer, 2 layers, 3 layers, 4 layers
   - Observe: How does depth affect accuracy and training time?

2. **Number of Neurons per Layer**
   - Try: 32, 64, 128, 256, 512 neurons
   - Observe: How does width affect performance?

3. **Learning Rate** (in optimizer)
   - Try: 0.001, 0.01, 0.1, 0.0001
   - Observe: How does learning rate affect convergence?

4. **Batch Size**
   - Try: 16, 32, 64, 128
   - Observe: How does batch size affect training speed and stability?

5. **Number of Epochs**
   - Try: 5, 10, 20, 50
   - Observe: When does the model stop improving? Does it overfit?

6. **Activation Functions**
   - Try: 'relu', 'tanh', 'sigmoid' in hidden layers
   - Observe: Which works best for this problem?

7. **Optimizers**
   - Try: 'adam', 'sgd', 'rmsprop'
   - Observe: Which optimizer converges faster?


**Expected outcome:** You should discover which parameters work best for this problem and understand how they affect model performance.

**Checkpoint:** Try at least 3 different parameter combinations and document your findings.

---

## Part 2: Teaching Models to Understand - Audio Classification with Transformers

### Background

In Part 1, you built a model to "see" images. Now, let's explore how models can "understand" audio! We'll use pre-trained transformer models that have already learned to recognize different types of audio, including speech intents. This part will be more hands off.

**What are Transformers?**
Transformers are a type of neural network architecture that has revolutionized AI. They can process sequences of data (like words in a sentence or audio samples) and understand context and relationships.

**Pre-trained Models:**
Instead of training from scratch, we'll use models that have already been trained on massive datasets. This is called "transfer learning" - using knowledge from one task to solve another.

---

### Setting Up Audio Classification

**Objective:** Install required libraries and understand the audio classification pipeline.

**What to do:**
1. Install required packages

**Required packages:**
```bash
pip install datasets transformers torch torchaudio
```

**Code template:**
You can use the [`transformer_audio.ipynb`](https://github.com/ai-consulting-bootcamp/ac-bootcamp-public/blob/main/notebooks/week-1/submodule-4-speech-recognition/transformer_audio.ipynb) (the notebook from class), as example to start.

### Step 1: Setting Up Audio Classification

1. Search for an Audio Dataset on Hugging Face. You can find any example you found interesting.
2. Load the audio dataset
3. Understand the data structure

### Step 2: Using a Pre-trained Audio Classifier

**Objective:** Use a pre-trained model to classify audio intents.

**What to do:**
1. Load a pre-trained audio classification model. You can choose any model that you find interesting.
2. Classify a sample audio
3. Understand what the model is doing

**Expected outcome:** You should see the model making predictions about what intent the audio contains (like "find_phone", "change_language", etc.).

**Checkpoint:** Verify that:
- The model loads successfully
- Predictions are made
- You can see the confidence scores

---

### Step 3: Exploring Multiple Samples

**Objective:** Test the model on multiple audio samples to understand its capabilities.

**What to do:**
1. Test on several different audio samples
2. Compare predictions with actual labels
3. Observe model confidence

**Expected outcome:** You should see the model's performance across multiple samples and an overall accuracy score.

**Checkpoint:** Verify that you can test multiple samples and calculate accuracy.


**Application:** What are some use cases for a pretrained model?
---

### Step 4: Understanding How Models Learn

**Objective:** Reflect on how models "see" and "understand" different types of data.

**What to do:** Answer these questions to deepen your understanding.

**Questions to Explore:**

1. **Teaching Models to See (Images):**
   - How did your neural network learn to distinguish dogs from cats?
   - What patterns do you think it learned? (edges, shapes, colors, textures?)
   - Why did flattening the image work? What information might be lost?

2. **Teaching Models to Understand (Audio):**
   - How is audio different from images as input to a neural network?
   - What do you think the pre-trained model learned during its training?
   - Why do we use pre-trained models instead of training from scratch?

3. **Transfer Learning:**
   - The audio model was trained on one dataset but works on another. Why?
   - What knowledge "transferred" from the original training?
   - How is this similar to how humans learn?

4. **Model Architecture:**
   - Both models use neural networks, but they're structured differently. Why?
   - What makes transformers good for audio/text sequences?
   - How do convolutional layers (used in image models) differ from transformer layers?


**Expected outcome:** You should have a deeper understanding of how models process different types of data.

**Checkpoint:** Write thoughtful answers to the reflection questions.

---

## Bonus Challenge: Full CIFAR-10 Classification

If you want an extra challenge, try building a classifier for all 10 CIFAR-10 classes!

**Task:** Modify your code to classify all 10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

**Changes needed:**
1. Don't filter the dataset - use all classes
2. Change output layer to 10 neurons (one per class)
3. Use 'softmax' activation instead of 'sigmoid'
4. Use 'categorical_crossentropy' loss instead of 'binary_crossentropy'
5. Convert labels to one-hot encoding

**Code hints:**
```python
# Load full CIFAR-10 (no filtering)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# One-hot encode labels
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

# Build model with 10 output neurons
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(3072,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile with categorical crossentropy
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # For multi-class
    metrics=['accuracy']
)

# Train and evaluate...
```

**Expected outcome:** You should be able to classify all 10 classes, though accuracy might be lower (~40-50%) since it's a harder problem.

---

## Submission Guidelines

### What to Submit

**Part 1 (Dog/Cat Classifier):**
- [ ] Your complete `dog_cat_classifier.py` file
- [ ] Saved model weights (upload the `saved_models` folder or specific weight files)
- [ ] Screenshots or output showing:
  - Training progress
  - Final test accuracy
  - Training history plots
  - Sample predictions
- [ ] Documentation of your experiments:
  - Which parameters did you try?
  - What were the results?
  - What did you learn?

**Part 2 (Audio Classification):**
- [ ] Your complete `audio_classification.py` file
- [ ] Screenshots or output showing:
  - Model predictions on sample audio
  - Accuracy calculation
- [ ] Your reflection document answering the questions about how models learn

### How to Submit

**Upload your work:**
1. **For code files:** Upload your Python scripts
2. **For model files:** Upload your saved model weights (can be large files)
3. **For screenshots:** Upload PNG or JPG images
4. **For reflections:** Upload text or PDF files

**Important notes:**
- Model weight files can be large (several MB). Make sure they upload successfully
- Document your experimentation process
- Include your reflections on how models learn
- Show your thought process in code comments

**Due date:** Check with your instructor for the specific due date.

---

## Troubleshooting

**Common issues and solutions:**

**Issue 1: "Out of memory" error during training**
- **Solution:** Reduce batch size (try 16 or 8), or reduce the number of neurons in layers.

**Issue 2: "Model accuracy is very low (<50%)"**
- **Solution:** 
  - Try training for more epochs
  - Increase model capacity (more neurons/layers)
  - Check if data is normalized correctly
  - Verify labels are correct

**Issue 3: "Training is very slow"**
- **Solution:** 
  - Reduce batch size
  - Train for fewer epochs initially
  - Use a smaller model for testing
  - Check if GPU is being used

**Issue 4: "Audio model download fails"**
- **Solution:** 
  - Check internet connection
  - The model will download automatically on first use
  - It may take several minutes
  - Ensure you have enough disk space

**Issue 5: "Audio classification gives wrong results"**
- **Solution:** 
  - This is normal - models aren't perfect
  - Try different audio samples
  - Check if audio sampling rate matches (16kHz)
  - Some audio may be unclear or contain noise

**Issue 6: "Can't load saved model"**
- **Solution:** 
  - Make sure you're using the same TensorFlow version
  - Check file paths are correct
  - Try loading the full model instead of just weights

---

## Additional Resources

**Reference materials:**
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Neural Network Basics](https://www.tensorflow.org/tutorials/keras/classification)

**Tools and software:**
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

**Further reading:**
- [Understanding Neural Networks](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
- [Transfer Learning Explained](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Audio Processing with ML](https://huggingface.co/docs/transformers/tasks/audio_classification)

---

## To-Do List

**Part 1 (Dog/Cat Classifier):**
- [ ] Complete Step 1: Setting Up the Environment
- [ ] Complete Step 2: Loading and Preparing the Data
- [ ] Complete Step 3: Building Your First Neural Network
- [ ] Complete Step 4: Training the Model
- [ ] Complete Step 5: Evaluating the Model
- [ ] Complete Step 6: Exporting Model Weights
- [ ] Complete Step 7: Experimentation with Parameters
- [ ] Document your experiments and findings

**Part 2 (Audio Classification):**
- [ ] Complete Step 1: Setting Up Audio Classification
- [ ] Complete Step 2: Using a Pre-trained Audio Classifier
- [ ] Complete Step 3: Exploring Multiple Samples
- [ ] Complete Step 4: Understanding How Models Learn
- [ ] Write your reflection document

**Bonus:**
- [ ] (Optional) Try the full CIFAR-10 classification challenge

---

## Learning Reflection

After completing this lab, reflect on what you've learned:

1. **Neural Network Basics:** What are the key components of a neural network? How do they work together?
2. **Training Process:** What happens during training? How does the model learn?
3. **Hyperparameters:** Which parameters had the biggest impact on performance? Why?
4. **Pre-trained Models:** What are the advantages of using pre-trained models? When would you train from scratch?
5. **Different Data Types:** How do models process images vs. audio? What's similar? What's different?
6. **Model Understanding:** How do you think models "see" images and "understand" audio? What patterns do they learn?

**Key takeaways:**
- Neural networks learn by adjusting weights through backpropagation
- Hyperparameters significantly affect model performance - experimentation is key
- Pre-trained models leverage transfer learning to solve new problems efficiently
- Different data types (images, audio) require different model architectures
- Understanding how models work helps you build better AI systems
- Exporting model weights allows you to reuse trained models

These concepts are fundamental to building more advanced AI systems and understanding how modern AI works.

---

*Congratulations on building your first neural network! You've taken an important step in understanding how AI models learn to perceive and understand the world. Keep experimenting and exploring!*
