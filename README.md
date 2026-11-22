# Build Simple Neural Network

**Track:** Module 1 — ML & neural networks · **When:** Week 1, Day 4 · **Status:** Optional / Extra

This repository contains everything you need for this lab.

## Files

- [`instructions.md`](./instructions.md) — the lab instructions
- [`rubric.md`](./rubric.md) — how your submission is graded; this is what the AI reviewer checks your PR against

## How to complete this lab

1. **Fork** this repository.
2. Do the work described in `instructions.md`, committing to your fork.
3. Open a **pull request** back into this repository.
4. You'll receive **AI feedback** on your PR based on `rubric.md`. Address any blocking feedback and push updates to the same PR.

## Submission hygiene

- Keep this repository scoped to this lab only — no unrelated projects or personal files.
- Use clear, descriptive filenames.
- Remove secrets, API keys, and tokens before committing.


## Hyperparameter experiments

The baseline model used hidden layers of 128 and 64 neurons, a learning rate of 0.001, and a batch size of 32.

| Configuration | Hidden layers | Learning rate | Batch size | Validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 128, 64 | 0.001 | 32 | 60.50% | 60.90% |
| Smaller network | 64, 32 | 0.001 | 32 | 58.40% | 63.05% |
| Larger network | 256, 128 | 0.001 | 32 | 60.00% | 59.85% |
| Lower learning rate | 128, 64 | 0.0005 | 32 | 58.75% | 60.55% |

The smaller network achieved the highest test accuracy among the three experiments, but its validation accuracy was lower. The baseline had the highest validation accuracy, so it was retained as the preferred configuration. Results varied slightly between runs.

## Audio Classification

The audio classifier used the pre-trained `anton-l/xtreme_s_xlsr_300m_minds14` model and the English-Australian (`en-AU`) subset of the MInDS-14 dataset. The model predicted the intent of each audio sample.

On a reproducible random sample of 20 audio examples, the model correctly classified 19, for a sample accuracy of 95%. This is an exploratory result, not an independent test score: the available subset uses a `train` split, and the selected model was fine-tuned on MInDS-14. The model's confidence score describes how strongly it favors a prediction; it does not guarantee that the prediction is correct.

## Learning Reflection

### 1. Teaching Models to See Images

The dog-versus-cat network learned by making predictions, comparing them with the correct labels, and adjusting its weights to reduce its errors. Through repeated examples, it could learn visual patterns such as edges, shapes, colors, and textures.

Flattening changed each 32 × 32 × 3 image into a row of 3,072 pixel values, which made it possible to pass the data into dense layers. The pixel values and their positions remain in that row, but the model is not directly told which pixels are neighbors. This makes it harder to learn spatial patterns than with a convolutional network.

### 2. Teaching Models to Understand Audio

An image is a two-dimensional grid of color values. Audio is a sequence of sound measurements that changes over time. For speech, the timing and order of sounds matter: the model must use acoustic patterns that unfold through the recording.

The pre-trained audio model learned representations of speech and intent from many labeled examples. Using a pre-trained model let us make predictions without collecting a large audio dataset and training a new model from scratch.

### 3. Transfer Learning

A pre-trained model can work on new audio because it has learned reusable acoustic features, such as patterns in speech sounds and their timing. Fine-tuning adapts those features to predict intent labels.

This is somewhat like how people apply language and listening skills learned over time when they encounter a new speaker or situation. A model does this through learned numerical representations rather than human understanding.

### 4. Model Architecture

Both systems use neural networks, but they work with different input structures. The image classifier uses dense layers on flattened pixel values. The audio classifier uses a transformer-based architecture designed to process audio features as a sequence.

Transformers use attention to relate information from different parts of a sequence. Convolutional layers instead apply filters to nearby regions, which works well for local image patterns such as edges and textures. The architectures reflect the different structures of images and audio.