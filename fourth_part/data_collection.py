import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
import keyboard
from my_functions import *
# Define the actions (signs) that will be recorded and stored in the dataset
actions = np.array(['a', 'd', 'n', 'y', 'name', 'my', 'hello'])

# Define the number of sequences and frames to be recorded for each action
sequences = 30
frames = 10

# Set the path where the dataset will be stored
PATH = os.path.join('./fourth_part/data')

# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    os.makedirs(os.path.join(PATH, action, str(sequence)), exist_ok=True)

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    for action in actions:
        print(f"Prepare to record data for action: {action}.")
        print("Press 'Space' to start recording all sequences for this action.")
        
        # Wait for the space key to start recording all sequences for the current action
        while True:
            ret, image = cap.read()
            if not ret:
                print("Camera error. Exiting...")
                exit()

            # Display instructions on the screen
            cv2.putText(image, f'Action: "{action}" - Ready to record.',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, 'Press "Space" to start recording.',
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Camera', image)
            if keyboard.is_pressed(' '):  # Start recording when space is pressed
                break
            if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:  # Exit if the window is closed
                cap.release()
                cv2.destroyAllWindows()
                exit()
            cv2.waitKey(1)

        # Loop through each sequence and frame for the action
        for sequence in range(sequences):
            for frame in range(frames):
                frame_path = os.path.join(PATH, action, str(sequence), f"{frame}.npy")
                
                # Skip recording if the keypoints file already exists
                if os.path.exists(frame_path):
                    print(f"Data for {action}, sequence {sequence}, frame {frame} already exists. Skipping...")
                    continue

                ret, image = cap.read()
                if not ret:
                    print("Camera error. Exiting...")
                    exit()

                # Process the image and extract hand landmarks using MediaPipe Holistic
                results = image_process(image, holistic)
                draw_landmarks(image, results)

                # Display the recording progress on the screen
                cv2.putText(image, f'Action: "{action}" | Sequence: {sequence + 1}/{sequences} | Frame: {frame + 1}/{frames}',
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                cv2.waitKey(1)

                # Extract the landmarks and save them to a file
                keypoints = keypoint_extraction(results)
                np.save(frame_path, keypoints)

                # Exit if the camera window is closed
                if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
                break
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")
