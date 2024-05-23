import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Define acceptable image extensions
image_extensions = ['.jpg', '.jpeg', '.png']

# Iterate through each item in the DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Ensure it's a directory
        print(f'Processing directory: {dir_path}')
        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            # Check if the file has a valid image extension
            if os.path.isfile(img_full_path) and os.path.splitext(img_full_path)[1].lower() in image_extensions:
                print(f'Processing file: {img_full_path}')
                try:
                    img = cv2.imread(img_full_path)
                    if img is None:
                        raise Exception(f"Error: Could not read image {img_full_path}")

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    results = hands.process(img_rgb)
                    if results.multi_hand_landmarks:
                        data_aux = []
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Draw landmarks on the image (optional)
                            # mp_drawing.draw_landmarks(
                            #     img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            #     mp_drawing_styles.get_default_hand_landmarks_style(),
                            #     mp_drawing_styles.get_default_hand_connections_style())

                            # Extract and normalize landmark coordinates
                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x / img.shape[1]  # Normalize by width
                                y = hand_landmarks.landmark[i].y / img.shape[0]  # Normalize by height
                                data_aux.append(x)
                                data_aux.append(y)

                        data.append(data_aux)
                        labels.append(dir_)

                        # Display the image with landmarks using matplotlib (optional)
                        # plt.figure(figsize=(10, 10))
                        # plt.imshow(img_rgb)
                        # plt.title(f'Image with Hand Landmarks: {img_full_path}')
                        # plt.axis('off')
                        # plt.show()
                except Exception as e:
                    print(f"Error processing {img_full_path}: {e}")

# Save the data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)