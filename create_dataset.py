import mediapipe as mp
import pickle
import cv2
import os
import matplotlib.pyplot as plt


DATA_DIR = './data'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []

labels = []

for dir_ in os.listdir(DATA_DIR):
    for file in os.listdir(os.path.join(DATA_DIR, dir_)):
        image = cv2.imread(os.path.join(DATA_DIR, dir_, file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data_aux = []

        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_aux.append(x)
                    data_aux.append(y)
            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data':data, 'labels':labels}, f)
f.close()