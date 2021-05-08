from flask import Flask, render_template,jsonify, flash, request, url_for
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from numpy import array
from tensorflow.keras.models import load_model

import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing import image
import os

app = Flask(__name__)

neg = ('angry', 'disgust', 'fear', 'sad',)
pos = ('happy','surprise')
objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def init():
    global graph
    graph = tf.get_default_graph()

def sentiment_analysis(text_sent):
    model = tf.keras.models.load_model('sentiment_analysis.h5')
    text = text_sent
    strip_special_char = re.compile("[^a-zA-z0-9\s]+")
    text = text.lower().replace("<br />"," ")
    text = re.sub(strip_special_char,"",text.lower())
    # loading tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    seq = tokenizer.texts_to_sequences([text])
    text = pad_sequences(seq)
    pred = model.predict(text)
    if(pred[0][0]>pred[0][1]):
        sentiment = 'Negative'
        probability = pred[0][0]
    if(pred[0][0]<pred[0][1]):
        sentiment = 'Positive'
        probability = pred[0][1]
    return sentiment, probability

def emotion_recognition(filename):
    
    model = tf.keras.models.load_model('facial_expression.h5')
    img = image.load_img(filename, color_mode = "grayscale", target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    
    x = np.array(x, 'float32')
    x = x.reshape(1,48,48,1)
    custom = model.predict(x) #prediction
    m=0.000000000000000000001
    a=custom[0]
    for i in range(0,len(a)):
        if a[i]>m:
            m=a[i]
            ind=i
    return ind,m

# Function to load and prepare the image in right shape
def read_image(filename):
    # Load the image
    img = image.load_img(filename, color_mode = "grayscale", target_size=(48, 48))
    # Convert the image to array
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img /=255
    # Reshape the image into a sample of 1 channel
    img = img.reshape(1, 48, 48, 1)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0   
    return img

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET','POST'])
def home():

    return render_template("home.html", pageType = "image")

@app.route('/text', methods=['POST','GET'])
def sa_prediction():
    if request.method == 'POST':
        text = request.form['text']
        sentiment, probability = sentiment_analysis(text)
    return render_template('home.html', text = text, sentiment = sentiment, probability = probability, pageType = "text" )

@app.route("/image", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename) #file path for storing image
            file.save(file_path)
            ind, m = emotion_recognition(file_path)
            exp = objects[ind]
                
    return render_template('home.html', expression = exp, per = m, user_image = file_path, pageType = "image")
        
@app.route('/depression', methods=['GET','POST'])  #single image
def depression():
    if request.method == 'POST':
        text = request.form['text']
        sentiment, probability = sentiment_analysis(text)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename) #file path for storing image
            file.save(file_path)
            ind, m = emotion_recognition(file_path)
    
            if objects[ind] in neg:
                f_result = 0
            else: 
                f_result =1
            if sentiment == 'Negative':
                s_result = 0
            else: 
                s_result =1

            if f_result==0 and s_result==0:
                depression = 'High'
            elif f_result==1 and s_result==1:
                depression = 'Low'
            elif f_result==1 and s_result==0:
                depression = 'Mild'
            elif f_result==0 and s_result==1:
                depression = 'Average'
    
    return render_template('home.html', status = depression, user_image_dep = file_path, pageType = "depression")

if __name__ == "__main__":
    app.run(debug=True)