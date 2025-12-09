import os
import cv2
import pickle
import mediapipe as mp

DATA_DIR = './data'
OUTPUT_FILE = './data.pickle'

# Correct Mediapipe usage
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

features = []
labels = []
total_images = 0
images_with_landmarks = 0

for class_label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, class_label)
    if not os.path.isdir(class_dir):
        continue

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        total_images += 1

        if img is None:
            print("Could not read image:", img_path)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            images_with_landmarks += 1
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x)
                    data_aux.append(lm.y)

                features.append(data_aux)
                labels.append(class_label)

print("Total images:", total_images)
print("Images with landmarks:", images_with_landmarks)
print("Total samples:", len(features))

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': features, 'labels': labels}, f)

print("Dataset saved to:", OUTPUT_FILE)
