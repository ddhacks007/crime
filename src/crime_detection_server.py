from flask import jsonify, make_response, request, current_app
import numpy as np
from flask import Flask
import keras
from keras.models import model_from_json
app = Flask(__name__)

@app.route('/get_me_the_results', methods = ['POST'])
def get_me_the_results():
    keras.backend.clear_session()
    json_file = open('/Users/deepakraju/crime_detection/src/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("/Users/deepakraju/crime_detection/src/model.h5")
    k = request.get_json()
    k = np.array(k)
    k = (k/255.0).reshape(1,230,230,3)
    print(k.shape)
    return jsonify({"results":str(loaded_model.predict(k).flatten()[0])})
if __name__ == "__main__":
    app.run()
