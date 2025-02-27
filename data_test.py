import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Label dictionary for predictions
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'HUG'
}

def process_landmarks(landmarks, width, height):
    """Process hand landmarks to extract normalized data."""
    data_aux = []
    x_coords, y_coords = [], []

    # Collect x and y coordinates
    for lm in landmarks:
        x_coords.append(lm.x)
        y_coords.append(lm.y)

    # Normalize coordinates
    for lm in landmarks:
        data_aux.append(lm.x - min(x_coords))
        data_aux.append(lm.y - min(y_coords))

    # Bounding box dimensions
    x1 = int(min(x_coords) * width) - 10
    y1 = int(min(y_coords) * height) - 10
    x2 = int(max(x_coords) * width) + 10
    y2 = int(max(y_coords) * height) + 10

    return data_aux, (x1, y1, x2, y2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame. Exiting...")
        break

    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Process and predict
            landmarks = hand_landmarks.landmark
            data_aux, bbox = process_landmarks(landmarks, width, height)

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Draw bounding box and label
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            except Exception as e:
                print(f"Error during prediction: {e}")

    cv2.imshow('Hand Gesture Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
