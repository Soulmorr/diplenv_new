from nltk import word_tokenize
import useless_words
from nltk.stem import PorterStemmer
import time
from shutil import copyfile
from difflib import SequenceMatcher
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from ffpyplayer.player import MediaPlayer
from PIL import ImageTk, Image
import cv2
import os
# CONSTANTS

SIGN_PATH = ".\\seccond_part\\Signs"
DOWNLOAD_WAIT = 20
SIMILIARITY_RATIO = 0.9
# Get words
username = os.getlogin()
window = tk.Tk()
window.title("ASL Video Generator")


def download_word_sign(word):
    browser = webdriver.Firefox()
    browser.get("https://www.aslpro.cc/cgi-bin/aslpro/aslpro.cgi")
    first_letter = word[0]
    letters = browser.find_elements("xpath",'//a[@class="sideNavBarUnselectedText"]')
    for letter in letters:
        if first_letter == str(letter.text).strip().lower():
            letter.click()
            time.sleep(2)
            break

    # Show drop down menu ( Spinner )
    spinner = browser.find_elements("xpath","//option")
    best_score = -1.
    closest_word_item = None
    for item in spinner:
        item_text = item.text
        # if stem == str(item_text).lower()[:len(stem)]:
        s = similar(word, str(item_text).lower())
        if s > best_score:
            best_score = s
            closest_word_item = item
            print(word, " ", str(item_text).lower())
            print("Score: " + str(s))
    if best_score < SIMILIARITY_RATIO:
        print(word + " not found in dictionary")
        return
    real_name = str(closest_word_item.text).lower()

    print("Downloading " + real_name + "...")
    closest_word_item.click()
    time.sleep(DOWNLOAD_WAIT)
    # src=browser.find_element('xpath','//video/source').get_attribute("src")
    # r = requests.get(src, allow_redirects=True)
    # open(in_path, 'wb').write(r.content) 
    in_path = "C:\\Users\\"+ username +"\\Downloads\\" + real_name + ".mp4"
    out_path = SIGN_PATH + "\\" + real_name + ".mp4"
    convert_file_format(in_path, out_path)
    browser.close()
    return real_name

def convert_file_format(in_path, out_path):
    # Converts .swf filw to .mp4 file and saves new file at out_path
    from ffmpy import FFmpeg

    ff = FFmpeg(
    inputs = {in_path: None},
    outputs = {out_path: None})
    ff.run()

def get_words_in_database():
    import os
    vids = os.listdir(SIGN_PATH)
    vid_names = [v[:-4] for v in vids]
    return vid_names

def process_text(text):
    # Split sentence into words
    words = word_tokenize(text)
    # Remove all meaningless words
    usefull_words = [str(w).lower() for w in words if w.lower() not in set(useless_words.words())]

    # TODO: Add stemming to words and change search accordingly. Ex: 'talking' will yield 'talk'.
    # from nltk.stem import PorterStemmer
    # ps = PorterStemmer()
    # usefull_stems = [ps.stem(word) for word in usefull_words]
    # print("Stems: " + str(usefull_stems))

    # TODO: Create Sytnax such that the words will be in ASL order as opposed to PSE.

    return usefull_words


def merge_signs(words):
    # Write a text file containing all the paths to each video
    with open("vidlist.txt", 'w') as f:
        for w in words:
            f.write("file '" + SIGN_PATH + "\\" + w + ".mp4'\n")
    command = "ffmpeg -f concat -safe 0 -i vidlist.txt -c copy output.mp4 -y"
    import shlex
    # Splits the command into pieces in order to feed the command line
    args = shlex.split(command)
    import subprocess
    process = subprocess.Popen(args)
    process.wait() # Block code until process is complete
    copyfile("output.mp4",SIGN_PATH + "\\Output\\out.mp4") # copyfile(src, dst)
    # remove the temporary file (it used to ask me if it should override previous file).
    import os
    os.remove("output.mp4")

def in_database(w):
    db_list = get_words_in_database()
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    s = ps.stem(w)
    for word in db_list:
        if s == word[:len(s)]:
            return True
    return False


def similar(a, b):
    # Returns a decimal representing the similiarity between the two strings.
    return SequenceMatcher(None, a, b).ratio()

def find_in_db(w):
    best_score = -1.
    best_vid_name = None
    for v in get_words_in_database():
        s = similar(w, v)
        if best_score < s:
            best_score =  s
            best_vid_name = v
    if best_score > SIMILIARITY_RATIO:
        return best_vid_name
# Get text
# text = str(input("Enter the text you would like to translate to pse \n"))
text = "hello"
print("Text: " + text)
# Process text
words = process_text(text)
# Download words that have not been downloaded in previous sessions.
real_words = []
for w in words:
    real_name = find_in_db(w)
    if real_name:
        print(w + " is already in db as " + real_name)
        real_words.append(real_name)
    else:
        real_words.append(download_word_sign(w))
words = real_words
# Concatenate videos and save output video to folder
merge_signs(words)

# Play the video
from os import startfile
#startfile(SIGN_PATH + "\\Output\\out.mp4")

def process_text_and_generate_video():
    # Get text from input field
    text = text_entry.get("1.0", tk.END).strip()
    
    if not text:
        messagebox.showerror("Error", "Please enter some text.")
        return
    
    # Process text
    words = process_text(text)
    
    # Download words that have not been downloaded in previous sessions
    real_words = []
    for w in words:
        real_name = find_in_db(w)
        if real_name:
            print(w + " is already in the database as " + real_name)
            real_words.append(real_name)
        else:
            real_words.append(download_word_sign(w))
    
    words = real_words
    
    # Concatenate videos and save output video to folder
    merge_signs(words)
    
    # Show success message
    messagebox.showinfo("Success", "Video generated successfully.")

def open_output_video():
    try:
        output_path = SIGN_PATH + "\\Output\\out.mp4"
        cap = cv2.VideoCapture(output_path)
        
        # Function to play the video frames
        def play_video():
            ret, frame = cap.read()
            if ret:
                cv2.imshow("ASL Video", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit video playback
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                window.after(30, play_video)
            else:
                cap.release()
                cv2.destroyAllWindows()
        
        # Start playing the video
        play_video()
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open the output video: {str(e)}")

# Function to exit the application
def exit_application():
    window.destroy()
# Create text input field
text_label = tk.Label(window, text="Enter the text:")
text_label.pack()
text_entry = tk.Text(window, height=5)
text_entry.pack()

# Create generate button
generate_button = tk.Button(window, text="Generate Video", command=process_text_and_generate_video)
generate_button.pack()

# Create open button
open_button = tk.Button(window, text="Open Output Video", command=open_output_video)
open_button.pack()

exit_button = tk.Button(window, text="Exit", command=exit_application)
exit_button.pack()
# Start the GUI event loop
window.mainloop()