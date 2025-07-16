import os
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image

# ✅ Paths (relative)
IMAGE_FOLDER = "archive/Images"
CAPTION_FILE = "archive/captions.txt"


# ✅ Step 1: Load and clean captions
def load_captions(file_path):
    df = pd.read_csv(file_path)
    descriptions = {}

    for index, row in df.iterrows():
        img_id = row['image']
        caption = row['caption']

        # Clean the caption
        caption = caption.lower()
        caption = caption.translate(str.maketrans('', '', string.punctuation))
        caption = caption.split()
        caption = [word for word in caption if len(word) > 1 and word.isalpha()]
        caption = ' '.join(caption)

        if img_id not in descriptions:
            descriptions[img_id] = []
        descriptions[img_id].append(caption)

    return descriptions


# ✅ Step 2: Load InceptionV3 model
def load_cnn_model():
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return model


# ✅ Step 3: Preprocess a single image
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# ✅ Step 4: Extract features from all images
def extract_features(directory, model):
    features = {}
    count = 0
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        try:
            img = preprocess_img(img_path)
            feature = model.predict(img, verbose=0)
            features[img_name] = feature
            count += 1
            if count % 500 == 0:
                print(f"✅ Processed {count} images...")
        except Exception as e:
            print(f"❌ Failed on {img_name}: {e}")
    return features


# ✅ Main Code
if __name__ == "__main__":
    # Load and clean captions
    print("📄 Loading and cleaning captions...")
    captions = load_captions(CAPTION_FILE)
    print(f"✅ Captions loaded for {len(captions)} images.")

    # Load model
    print("📦 Loading InceptionV3 model...")
    cnn_model = load_cnn_model()

    # Extract features
    print("🧠 Extracting features from images...")
    features = extract_features(IMAGE_FOLDER, cnn_model)

    # Save features
    with open("image_features.pkl", "wb") as f:
        pickle.dump(features, f)

    print("✅ Image features extracted and saved to 'image_features.pkl'.")
