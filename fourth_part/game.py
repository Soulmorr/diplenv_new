# Import necessary libraries
import numpy as np
import os
import string
import mediapipe as mp
import tensorflow as tf
import cv2
from my_functions import *
import keyboard
from keras.src.saving import load_model
import language_tool_python
import random
import time
import asyncio
# Set the path to the data directory
PATH = os.path.join('./fourth_part/data')
display_text = False
display_start_time = 0
display_duration = 2
# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))
won_display_start_time = 0
won_display_duration = 5  # Duration in seconds
show_won_message = False
# Load the trained model
restart_timer_start_time = 0
restart_timer_duration = 60  # Time to think before restarting in seconds
waiting_for_restart = False
hint_timer_start_time = 0
hint_timer_duration = 5  # Time to think before restarting in seconds
hint_show = False
model = load_model('./fourth_part/my_model.h5')

# Create an instance of the grammar correction tool
tool = language_tool_python.LanguageToolPublicAPI('en-UK')

# Initialize the lists
sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []
score = 0
# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()
is_won =0
# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Run the loop while the camera is open
    while cap.isOpened():
        # Read a frame from the camera
        _, image = cap.read()
        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results = image_process(image, holistic)
        # Draw the sign landmarks on the image using draw_landmarks function from my_functions.py
        draw_landmarks(image, results)
        # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
        keypoints.append(keypoint_extraction(results))
        
        # Check if 10 frames have been accumulated
        if len(keypoints) == 10:
            # Convert keypoints list to a numpy array
            keypoints = np.array(keypoints)
            # Make a prediction on the keypoints using the loaded model
            prediction = model.predict(keypoints[np.newaxis, :, :])
            # Clear the keypoints list for the next set of frames
            keypoints = []
            
            if (sentence == []):
            # Check if the maximum prediction value is above 0.9
                
                hint_timer_start_time = time.time() 
                sentence = random.choice(actions)
            if np.amax(prediction) > 0.9:
                    # Check if the predicted sign is different from the previously predicted sign
                   #sentence = random.choice(actions)
                    
                if sentence == actions[np.argmax(prediction)] and score <5:
                    display_text = True
                    display_start_time = time.time()
                    # cv2.putText(image, "Correct! +1 point", (300, 200),
                    #     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    #time.sleep(3)
                    score=score +1
                    sentence = random.choice(actions)
                    hint_show=False
                    hint_timer_start_time= 0
                    hint_timer_start_time = time.time()
                    if score == 5 and show_won_message == False:
                        display_text=False
                        show_won_message = True
                        won_display_start_time = time.time()
                        waiting_for_restart = True
                        restart_timer_start_time = time.time() 
        
        cv2.putText(image, (f'Score: {score}'), (470, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X_coord = (image.shape[1] - textsize[0]) // 2
        cv2.putText(image,''.join(sentence), (text_X_coord, 470),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
        if display_text:
            # Check if the duration has passed
            if time.time() - display_start_time < display_duration:
                cv2.putText(image, "Correct! +1 point", (150, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Stop displaying the text after the duration
                display_text = False        
        # If the player has won, display "You won"
        if show_won_message:
            if time.time() - won_display_start_time < won_display_duration:
                cv2.putText(image, "You won!!!!", (230, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                show_won_message = False
        if waiting_for_restart:
            elapsed_time = time.time() - restart_timer_start_time
            remaining_time = int(restart_timer_duration - elapsed_time)
            if remaining_time > 0 and not keyboard.is_pressed('enter'):
                cv2.putText(image, f"Restarting in {remaining_time} seconds...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"or press enter to restart", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # If 60 seconds have passed, reset the game
                score = 0
                sentence = []
                waiting_for_restart = False  # Stop the countdown
                continue  # Restart the game loop
        if keyboard==keyboard.is_pressed('enter'):
                score = 0
                sentence = []
                waiting_for_restart = False  # Stop the countdown
                continue  # Restart
        if  hint_show==False:
                elapsed_time2 = time.time() - hint_timer_start_time
                remaining_time2 = int(hint_timer_duration - elapsed_time2)
                hint_show=True           
        if remaining_time2>0:
            cv2.putText(image, "For hint press CTRL", (300, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if (keyboard.is_pressed('CTRL')):
                    alpha = cv2.imread('.\\first_part\\amer_sign2.png', cv2.IMREAD_ANYCOLOR)
                    cv2.imshow('alphabet',alpha)
               
                
                    
                           
       
        # Show the image on the display
        cv2.imshow('Camera', image)

        cv2.waitKey(1)
        
        
        
        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Shut off the server
    tool.close()
