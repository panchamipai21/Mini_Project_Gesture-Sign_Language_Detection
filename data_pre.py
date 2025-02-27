import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to the dataset directory
DATA_DIR = './data'

# Containers for data and labels
data = []
labels = []

# Process each folder and image in the dataset directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Skip non-directories

    print(f"Processing directory: {dir_}")
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        if not os.path.isfile(img_full_path):
            continue  # Skip non-file entries

        data_aux = []
        img = cv2.imread(img_full_path)

        if img is None:
            print(f"Failed to load image: {img_full_path}")
            continue

        # Convert the image to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks in the image
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

        # Ensure the correct number of landmarks are collected
        if len(data_aux) == 42:  # 21 landmarks * 2 (x, y)
            data.append(data_aux)
            labels.append(dir_)
        else:
            print(f"Skipping image due to insufficient landmarks: {img_full_path}")

# Save the processed dataset to a pickle file
output_file = 'data.pickle'
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data saved to {output_file}. Total samples: {len(data)}")
