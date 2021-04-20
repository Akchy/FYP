# How to Run

Open a Virtual Env.

We need to install the dependencies for the same.
```
pip install -r requirements.txt
```

Then run flask
```
FLASK_APP=app.py
flask run
```

This app is hosted on heroku - depression-level.herokuapp.com

We have three API's - Analyse Facial Expressions only, Analyse Textual Sentiment and Depression Analysis

## For Facial Expression:
    - endpoint: /image
    - usage: curl -X POST -F \"image=@filename.extension\" https://depression-level.herokuapp.com/image
## For Sentiment Analysis:
    - endpoint: /text
    - usage: curl https://depression-level.herokuapp.com/text?stext="prection_text"

## For Depression Analysis:
    - endpoint: /depression
    - usage: curl -X POST -F \"image=@filename.extension\" https://depression-level.herokuapp.com/depression?stext="prediction_text"
    