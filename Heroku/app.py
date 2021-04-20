from flask import Flask, request, jsonify, request
from skimage import io
import tensorflow as tf
import keras
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import re


import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
   

@app.route('/')
def main():
    return "Hi"

@app.route("/image", methods=["POST"])
def emotion_recognition():
   
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    file = request.files['image']
    file.save(file.filename)
    model = tf.keras.models.load_model('facial_expression.h5')
    img = image.load_img(file.filename, color_mode = "grayscale", target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255

    custom = model.predict(x)

    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    x = np.array(x, 'float32')
    x = x.reshape([48, 48])

    m=0.000000000000000000001
    a=custom[0]
    for i in range(0,len(a)):
        if a[i]>m:
            m=a[i]
            ind=i
    return jsonify({'exprssion':str(objects[ind]), 'precentage':str(m)})


@app.route('/text',methods=["GET"])
def sentiment_analsis():

    model = tf.keras.models.load_model('sentiment_analysis.h5')
    text = request.args.get('stext')
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
    return jsonify({
        'sentiment':str(sentiment),
        'prob':str(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)