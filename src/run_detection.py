import scipy.misc
import numpy as np
import requests
import json
import os 
import scipy
import cv2
import re
from PIL import Image
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(data, key=alphanum_key)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

headers = {'content-type': 'application/json'}
images = [img for img in sorted_aphanumeric(os.listdir('/Users/deepakraju/crime_detection/crime_frames/assault'))if img.endswith(".jpg")]
i=0
while(cv2.waitKey(500) != ord('q')):
    full_image = Image.open('/Users/deepakraju/crime_detection/crime_frames/assault/'+images[i])
    full_image = full_image.resize((230,230))
    body = np.array(full_image)
    r = requests.post('http://127.0.0.1:5000/get_me_the_results', data=json.dumps(body,cls=NumpyEncoder), headers=headers)
    print('The probability that crime has taken place is ',(r.json()['results']))
    full_image = np.array(full_image)
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    i=i+1
cv2.destroyAllWindows()
