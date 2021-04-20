from flask import Flask, request, jsonify, request
from skimage import io
import tensorflow as tf
import keras
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import re
import os


import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

neg = ('angry', 'disgust', 'fear', 'sad',)
pos = ('happy','surprise')
objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
def emotion_recognition(image_sent):
    file = image_sent
    file.save(file.filename)
    model = tf.keras.models.load_model('facial_expression.h5')
    img = image.load_img(file.filename, color_mode = "grayscale", target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    custom = model.predict(x) #prediction
    x = np.array(x, 'float32')
    x = x.reshape([48, 48])
    os.remove(file.filename)
    m=0.000000000000000000001
    a=custom[0]
    for i in range(0,len(a)):
        if a[i]>m:
            m=a[i]
            ind=i
    return ind,m

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

@app.route('/')
def main():
    return "Hi"

@app.route("/image", methods=["POST"])
def facial_expression():
   
    ind, m = emotion_recognition(request.files['image'])
    return jsonify({'exprssion':str(objects[ind]), 'precentage':str(m)})


@app.route('/text',methods=["GET"])
def sentiment_analsis():

    sentiment, probability = sentiment_analysis(request.args.get('stext'))    
    return jsonify({
        'sentiment':str(sentiment),
        'prob':str(probability)
    })

@app.route('/depression', methods=["POST"])  #single image
def depression():

    ind, m = emotion_recognition(request.files['image'])
    sentiment, probability = sentiment_analysis(request.args.get('stext'))
    print(str(objects[ind]) + '\n'+ str(sentiment))
    if objects[ind] in neg:
        f_result = 0
    else: 
        f_result =1
    if sentiment == 'Negative':
        s_result = 0
    else: 
        s_result =1
    
    print(''+ str(f_result) +'\n' + str(s_result))
    if f_result==0 and s_result==0:
        depression = 'High'
    elif f_result==1 and s_result==1:
        depression = 'Low'
    elif f_result==1 and s_result==0:
        depression = 'Mild'
    elif f_result==0 and s_result==1:
        depression = 'Average'
    
    return jsonify({'depression level':depression})

if __name__ == "__main__":
    app.run(debug=True)