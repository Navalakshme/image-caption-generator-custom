import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# === Load model and tokenizer ===
print("📂 Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("caption_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

vocab_size = metadata["vocab_size"]
max_length = metadata["max_length"]

print("🧠 Building and loading model...")
model = tf.keras.models.load_model("model.keras")

# === Extract features from the image ===
def extract_features(filename):
    model = InceptionV3(weights="imagenet")
    model = tf.keras.Model(model.input, model.layers[-2].output)

    image = load_img(filename, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = model.predict(image, verbose=0)
    return feature

# === Map index to word ===
index_word = {v: k for k, v in tokenizer.word_index.items()}

# === Generate caption with beam search ===
def generate_caption_beam_search(model, tokenizer, photo, max_length, beam_index=3):
    start = [tokenizer.word_index['startseq']]
    sequences = [[start, 0.0]]  # (sequence, score)

    while len(sequences[0][0]) < max_length:
        all_candidates = []
        for seq, score in sequences:
            padded = pad_sequences([seq], maxlen=max_length)
            y_pred = model.predict([photo, padded], verbose=0)[0]

            # Get top predictions
            top_indexes = np.argsort(y_pred)[-beam_index:]

            for idx in top_indexes:
                next_seq = seq + [idx]
                next_score = score + np.log(y_pred[idx] + 1e-10)
                all_candidates.append([next_seq, next_score])

        # Order by score and select top beam_index
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_index]

    # Best sequence
    final_seq = sequences[0][0]
    final_caption = [index_word[i] for i in final_seq if i in index_word]
    final_caption = [word for word in final_caption if word not in ('startseq', 'endseq')]

    return ' '.join(final_caption)

# === Predict Caption ===
img_path = "test2.webp"  # Change to your test image
print("🖼️ Generating caption...")
photo = extract_features(img_path)
caption = generate_caption_beam_search(model, tokenizer, photo, max_length, beam_index=5)
print("🧾 Caption:", caption)
