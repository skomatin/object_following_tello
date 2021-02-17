# Time keeping
import time

# Import tello
from tellolib import Tello

# Import Opencv
import cv2

import threading

from model import *

# Load the DB credentials
# import os

# At most, send a command every COMMAND_DELAY seconds
COMMAND_DELAY = 1
heart_beat_freq = 10

IMAGE_PROCESSING_DELAY = 0.1    #Delay between processing of each frame (seconds)

# commands_db = ["battery?", "takeoff", "cw 20", "ccw 20", "streamon", "left 20", "ccw 90", "cw 20", "ccw 20", "cw 20", "ccw 20",
#                "cw 20", "ccw 20", "streamoff", "right 20", "ccw 90", "cw 20", "ccw 20", "cw 20", "land"]

# commands_db = ["battery?", "takeoff", "streamon", "up 30", "ccw 90", "forward 50", "left 30", "cw 20", "ccw 20",
#                "back 20", "right 30", "down 30", "streamoff", "land"]

commands_db = ["battery?", "takeoff", "streamon", "wait 1", "up 10", "up 50", "wait 3", "ccw 90", "up 50", "wait 3", "ccw 90", "wait 3", "ccw 90", "wait 3",
               "ccw 90", "wait 3", "streamoff", "land"]

last_frame = None           #Latest frame the drone received
is_streaming = False        #Keeps track of whether streamon command was called

def video_thread():
    """
    Thread to constantly update the latest frame variable
    """
    global last_frame
    # Creating stream capture object
    cap = cv2.VideoCapture('udp://' + drone.tello_ip + ':11111')

    while(True):
      _, last_frame = cap.read()
    cap.release()

def process_image():
    """
    Thread to process the latest frame from the drone
    """
    global last_frame, is_streaming
    i=0

    imgproc = ImgProc()
    while(True):
        if last_frame is not None and is_streaming:
            time.sleep(0.1)

            print("Processing frame ", i)
            imgproc.detect_object(last_frame, i)
            print("Processing complete ", i)
            i+=1

""" Main Entrypoint """
if __name__ == "__main__":

    # Create Tello connection
    drone = Tello()

    video_thread = threading.Thread(target=video_thread)
    video_thread.daemon = True
    video_thread.start()

    img_proc_thread = threading.Thread(target=process_image)
    img_proc_thread.daemon = True
    img_proc_thread.start()

    # Enter into the main event loop
    before = time.time()
    time_last_command_sent = time.time()
    i = 0
    while i < len(commands_db):
        now = time.time()
        if now - before > COMMAND_DELAY:
            before = now
            time_last_command_sent = now

            if commands_db[i] == "streamon":
                is_streaming = True
            elif commands_db[i] == "streamoff":
                is_streaming = False
            elif "wait" in commands_db[i]:
                delay = int(commands_db[i].split(" ")[1])
                print('Waiting {} seconds...'.format(delay))
                time.sleep(delay)
                i += 1
                continue

            drone.send_command(commands_db[i])
            i += 1
