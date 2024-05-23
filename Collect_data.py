import os
import cv2

def find_valid_camera_index():
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    return None

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

camera_index = find_valid_camera_index()
if camera_index is None:
    print("Error: No valid camera found")
    exit()

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Error: Cannot open video capture source at index {camera_index}")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Error: Failed to capture frame")
            continue

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Error: Failed to capture frame")
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        filename = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(filename, frame)
        print(f'Saved {filename}')

        counter += 1

cap.release()
cv2.destroyAllWindows()
