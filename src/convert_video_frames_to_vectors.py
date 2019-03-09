#importing libraries
import cv2
import scipy
import os
import scipy.misc
import time
import matplotlib.pyplot as mat
import numpy as np
import pickle
import re
from sklearn.externals import joblib


start=time.time()
cur_dir='/Users/deepak/crime_detection/crime_frames/'
image_to_vector=[]
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(data, key=alphanum_key)
for x in sorted([x for x in sorted(os.listdir(cur_dir)) if x!='.DS_Store']):
    curr_vector=[]
    for y in  sorted_aphanumeric([x for x in os.listdir(cur_dir+x) if x!='.DS_Store']):
            print (x,y)
            curr_vector.append(scipy.misc.imresize(cv2.imread(cur_dir+'/'+x+'/'+y),[230,230]))
    image_to_vector.append(np.array(curr_vector))        
image_to_vector=np.array(image_to_vector)
#converting_the_entire_frame_values
for i in range(len(sorted(os.listdir(cur_dir)))-1):
    joblib.dump(image_to_vector[i],'crime_images_to_vectors'+sorted(os.listdir(cur_dir))[i+1]+'.txt')

print ('Time taken to run this code',time.time()-start)
print ('all the frames have been converted to vectors successfully !')