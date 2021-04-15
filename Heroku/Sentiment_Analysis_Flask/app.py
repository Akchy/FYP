from flask import Flask, render_template, flash, request, url_for
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from numpy import array
from tensorflow.keras.models import load_model

import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

def init():
    global model
    model = load_model('sentiment_analysis.h5')

@app.route('/', methods=['GET','POST'])
def home():

    return render_template("home.html")

@app.route('/sentiment_analysis', methods=['POST','GET'])
def sa_prediction():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = ''
        max_length = 500
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
    return render_template('home.html', text =text, sentiment = sentiment, probability = probability)


if __name__ == "__main__":
    init()
    app.run()