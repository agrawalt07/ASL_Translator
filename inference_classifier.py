import cv2
import mediapipe as mp
import pickle
import numpy as np

def find_working_camera_index():
    for index in range(10):  # Adjust the range if you have more cameras
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()  # Release the camera
            return index
    return -1  # Return -1 if no working camera is found

cap = cv2.VideoCapture(find_working_camera_index())

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']

while True:

    data_aux = []

    labels_dict = {0: 'A', 1: 'B', 2: 'L'}
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_aux.append(x)
                    data_aux.append(y)
            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            print(predicted_character)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()


