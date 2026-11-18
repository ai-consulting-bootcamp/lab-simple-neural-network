# Build Simple Neural Network

## Part 1: Dog/Cat Classifier

Built a neural network using the CIFAR-10 dataset, filtered to cats and dogs.

- Training images: 10,000
- Test images: 2,000
- Final test accuracy: 62.85%
- Model format: `.keras`
- Weights: `.weights.h5`

Experiments tested a smaller network, a wider network, and a lower learning rate. The original model performed best.

## Part 2: Audio Classification

Used the SUPERB keyword-spotting dataset and a pre-trained Wav2Vec2 audio classifier. The model was tested on 20 audio samples and achieved 100% accuracy on that sample.

See `dog_cat_classifier.ipynb` for the complete code, outputs, experiments, predictions, and reflection.
