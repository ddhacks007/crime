
# coding: utf-8

# # Convert video into  frames 

import cv2
import os
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def getFrame(sec,root_name,type_of_video,video_number):
    curr_dir= root_name+type_of_video+'/'+video_number+" "+str(sec)+".jpg"
    global vidcap
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        ensure_dir(root_name+type_of_video+'/')
        cv2.imwrite(curr_dir, image)     # save frame as JPG file
    return hasFrames
root_dir='/Users/deepak/crime_detection/crime_frames/'
data_set_loc='/Users/deepak/crime_detection/Dataset/'
sec=0
for x in [x for x in sorted(os.listdir(data_set_loc)) if x!='.DS_Store']:
    for y in [y for y in sorted(os.listdir(data_set_loc+x)) if y!='.DS_Store']:
        vidcap = cv2.VideoCapture(data_set_loc+x+'/'+y)
        frameRate = 1 #it will capture image in each 0.5 second
        success = getFrame(sec,root_dir,x,y)
        while success:
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec,root_dir,x,y)
        sec=0
print("all the videos are converted to frames successfully")