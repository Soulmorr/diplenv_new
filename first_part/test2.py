import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import cv2
from tkinter import messagebox
import sys
DATA_DIR = '.\\first_part\\data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
alpha = cv2.imread('.\\first_part\\amer_sign2.png', cv2.IMREAD_ANYCOLOR)
number_of_classes = 24
dataset_size = 100
cv2.imshow('alphabet',alpha)
cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))
    # if cv2.waitKey(1) == 27:
    #         done=True
    #         break
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q"', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        # if cv2.waitKey(1) == 27:
        #     done=True
        #     break
        if cv2.waitKey(25) == ord('q'):
            break
      
        
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
     
        counter += 1
    #     if cv2.waitKey(1) == 27:
    #         done=True
    #         break
    # if done:
    #     break
cap.release()
cv2.destroyAllWindows()
messagebox.showinfo("INFO","Зчитування завершено!")


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = '.\\first_part\\data'
print("proces starteed")
data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
print("proces ended")
messagebox.showinfo("INFO","Створення датасету завершено!")




print("proces starteed")
data_dict = pickle.load(open('.\\first_part\\data.pickle', 'rb'))

##data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
max_length = max(len(lst) for lst in data_dict['data'])
data = [lst + [0] * (max_length - len(lst)) for lst in data_dict['data']]

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

a = open('model.p', 'wb')
pickle.dump({'model': model}, a)
a.close()
print("proces ended")

messagebox.showinfo("INFO","Створення моделі завершено!")