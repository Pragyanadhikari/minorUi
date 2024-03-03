from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import load_model
#from sklearn.metrics import f1_score

from keras import backend as K


# Define the custom metric function
def f1_score_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

MODEL_PATH = 'models/cnnbest.h5'
model = load_model(MODEL_PATH, custom_objects={'f1_score': f1_score_metric})
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Handle file upload and prediction for POST requests
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        
        # Apply labels to the predictions
        labels = ['FreshApple',
                  'FreshBanana',
                  'FreshBellpepper',
                  'FreshCapsicum',
                  'FreshCarrot',
                  'FreshCucumber',
                  'FreshGuava',
                  'FreshLime',
                  'FreshOrange',
                  'FreshPotato',
                  'FreshTomato',
                  'RottenApple',
                  'RottenBanana',
                  'RottenBellpepper',
                  'RottenCapsicum',
                  'RottenCarrot',
                  'RottenCucumber',
                  'RottenGuava',
                  'RottenLime',
                  'RottenOrange',
                  'RottenPotato',
                  'RottenTomato']
        
        # Get the index of the highest probability
        max_index = np.argmax(preds)
        
        # Get the corresponding label
        predicted_label = labels[max_index]
        
        # Return the predicted label
        return predicted_label
    
    else:
        # Handle GET requests (e.g., render a form for file upload)
        return render_template('upload_form.html')




if __name__ == '__main__':
    app.run(debug=True)
