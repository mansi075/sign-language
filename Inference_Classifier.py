import os
import pickle
import mediapipe as mp
import cv2
from sklearn.ensemble import RandomForestClassifier


# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Load the trained Random Forest Classifier model
model_path = 'hand_gesture_classifier.p'  # Replace with your model path
with open(model_path, 'rb') as f:
    model = pickle.load(f)['model']

# Function to preprocess landmark data (ensure it matches training process)
def preprocess_data(landmarks):
  x_ = []
  y_ = []
  for i in range(len(landmarks.landmark)):
      x = landmarks.landmark[i].x
      y = landmarks.landmark[i].y
      x_.append(x)
      y_.append(y)
  data_aux = []
  # Adjust calculations here to ensure expected number of features are extracted
  # (consider using relative positions, angles, etc. based on your model training)
  for i in range(len(landmarks.landmark) // 2):  # Assuming 21 landmarks (adjust if needed)
      x1 = x_[i*2]
      y1 = y_[i*2]
      x2 = x_[i*2 + 1]
      y2 = y_[i*2 + 1]
      relative_x = x2 - x1
      relative_y = y2 - y1
      data_aux.append(relative_x)
      data_aux.append(relative_y)
  # ... Add calculations for remaining features as needed ...
  return data_aux

# Function to display prediction result
def display_prediction(text):
  cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
              1, (0, 255, 0), 2, cv2.LINE_AA)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a more natural user experience
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(img_rgb)

    # Draw landmarks on the frame (optional)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                     mp_drawing_styles.get_default_hand_landmarks_style(),
                                     mp_drawing_styles.get_default_hand_connections_style())

            # Preprocess landmark data
            data = preprocess_data(hand_landmarks)

            # Make prediction using the trained model
            prediction = model.predict([data])

    # Display prediction text (if a hand is detected)
    if results.multi_hand_landmarks:
        display_prediction(f"Prediction: {prediction[0]}")
    else:
        display_prediction("No hand detected")

    cv2.imshow('MediaPipe Hand Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
