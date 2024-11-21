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

SIGN_PATH = ".\\seccond_part\\Signs"
DOWNLOAD_WAIT = 20
SIMILIARITY_RATIO = 0.9
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
    spinner = browser.find_elements("xpath","//option")
    best_score = -1.
    closest_word_item = None
    for item in spinner:
        item_text = item.text
        s = similar(word, str(item_text).lower())
        if s > best_score:
            best_score = s
            closest_word_item = item
            print(word, " ", str(item_text).lower())
            print("Score: " + str(s))
    if best_score < SIMILIARITY_RATIO:
        print(word + " not found in dictionary")
        messagebox.showerror("Помилка", "Такого слова немає в словнику відео спробуйте інше.")
        return
    real_name = str(closest_word_item.text).lower()

    print("Downloading " + real_name + "...")
    closest_word_item.click()
    time.sleep(DOWNLOAD_WAIT)
    in_path = "C:\\Users\\"+ username +"\\Downloads\\" + real_name + ".mp4"
    out_path = SIGN_PATH + "\\" + real_name + ".mp4"
    convert_file_format(in_path, out_path)
    browser.close()
    return real_name

def convert_file_format(in_path, out_path):
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

    words = word_tokenize(text)

    usefull_words = [str(w).lower() for w in words if w.lower() not in set(useless_words.words())]
    return usefull_words


def merge_signs(words):

    with open("vidlist.txt", 'w') as f:
        for w in words:
            f.write("file '" + SIGN_PATH + "\\" + w + ".mp4'\n")
    command = "ffmpeg -f concat -safe 0 -i vidlist.txt -c copy output.mp4 -y"
    import shlex

    args = shlex.split(command)
    import subprocess
    process = subprocess.Popen(args)
    process.wait() 
    copyfile("output.mp4",SIGN_PATH + "\\Output\\out.mp4") 
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



def process_text_and_generate_video():

    text = text_entry.get("1.0", tk.END).strip()
    
    if not text:
        messagebox.showerror("Помилка", "Будь ласка введіть текст.")
        return

    words = process_text(text)

    real_words = []
    for w in words:
        real_name = find_in_db(w)
        if real_name:
            print(w + " is already in the database as " + real_name)
            real_words.append(real_name)
        else:
            real_words.append(download_word_sign(w))
    
    words = real_words

    merge_signs(words)

    messagebox.showinfo("Вдало", "Відео було згенеровано.")

def open_output_video():
    try:
        output_path = SIGN_PATH + "\\Output\\out.mp4"
        cap = cv2.VideoCapture(output_path)
        
        def play_video():
            ret, frame = cap.read()
            if ret:
                cv2.imshow("ASL Video", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                window.after(30, play_video)
            else:
                cap.release()
                cv2.destroyAllWindows()
        

        play_video()
    
    except Exception as e:
        messagebox.showerror("Помилка", f"Невдалось відкрити відео: {str(e)}")


def exit_application():
    window.destroy()

text_label = tk.Label(window, text="Введіть речення:")
text_label.pack(pady=5)
text_entry = tk.Text(window, height=5)
text_entry.pack()

generate_button = tk.Button(window, text="Згенерувати відео", command=process_text_and_generate_video)
generate_button.pack(pady=5)

open_button = tk.Button(window, text="Відкрити останнє відео", command=open_output_video)
open_button.pack(pady=5)

exit_button = tk.Button(window, text="Вихід", command=exit_application)
exit_button.pack(pady=5)

window.mainloop()