from datasets import Audio, load_dataset
from transformers import pipeline
from io import BytesIO
import librosa
import random

# Load one English-Australian subset of the MINDS-14 audio dataset
print("Loading the MINDS-14 dataset...")
dataset = load_dataset("PolyAI/minds14", name="en-AU", split="train")

# Resample audio to 16 kHz and convert it to mono
dataset = dataset.cast_column(
    "audio",
    Audio(decode=False),
)

print("Number of audio samples:", len(dataset))
print("Available columns:", dataset.column_names)

# Load a pre-trained classifier fine-tuned for MINDS-14
print("\nLoading the pre-trained audio classifier...")
classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)

# Get one sample and read its audio without TorchCodec
example = dataset[0]
audio_reference = example["audio"]

# Prefer the audio bytes embedded in the dataset
if audio_reference.get("bytes") is not None:
    audio_source = BytesIO(audio_reference["bytes"])
else:
    audio_source = audio_reference["path"]

# Load as mono audio and resample to 16 kHz
audio_array, sample_rate = librosa.load(
    audio_source,
    sr=16_000,
    mono=True,
)

audio_input = {
    "array": audio_array,
    "sampling_rate": sample_rate,
}

true_intent = dataset.features["intent_class"].int2str(
    example["intent_class"]
)

# Predict the intent
predictions = classifier(audio_input, top_k=5)

print("\nActual intent:", true_intent)
print("Top predictions:")

for prediction in predictions:
    print(
        f"  {prediction['label']}: "
        f"{prediction['score']:.2%} confidence"
    )

# Evaluate the model on 20 samples
num_samples = min(20, len(dataset))
correct_predictions = 0

print(f"\nEvaluating {num_samples} audio samples:")
sample_indices = random.Random(42).sample(
    range(len(dataset)),
    num_samples,
)

for sample_number, index in enumerate(sample_indices, start=1):
    sample = dataset[index]
    audio_reference = sample["audio"]

    # Read embedded audio bytes when available
    if audio_reference.get("bytes") is not None:
        audio_source = BytesIO(audio_reference["bytes"])
    else:
        audio_source = audio_reference["path"]

    audio_array, sample_rate = librosa.load(
        audio_source,
        sr=16_000,
        mono=True,
    )

    audio_input = {
        "array": audio_array,
        "sampling_rate": sample_rate,
    }

    actual_intent = dataset.features["intent_class"].int2str(
        sample["intent_class"]
    )
    prediction = classifier(audio_input, top_k=1)[0]
    predicted_intent = prediction["label"]
    confidence = prediction["score"]

    is_correct = predicted_intent == actual_intent
    correct_predictions += int(is_correct)
    result = "Correct" if is_correct else "Incorrect"

    print(
        f"{sample_number:02d}. Actual: {actual_intent} | "
        f"Predicted: {predicted_intent} | "
        f"Confidence: {confidence:.2%} | {result}"
    )

accuracy = correct_predictions / num_samples
print(f"\nAccuracy on these {num_samples} samples: {accuracy:.2%}")