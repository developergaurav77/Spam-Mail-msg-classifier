from flask import Flask, jsonify, render_template, request,url_for
import joblib
import os
import numpy as np
from werkzeug.utils import redirect

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import string

# 1. preprocess
def transform(text):
    text = text.lower() #transform the given data into lower case
    text = nltk.word_tokenize(text) #split the data(sentences) into words
    
    y = []
    for i in text:
        if i.isalnum(): #removing special characters
            y.append(i)
    text = y.copy()
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:#removing stop words and punctuation
            y.append(i)
            
    text = y.copy()
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)
# 2. vectorizer
# 3. predict
# 4. display

app = Flask(__name__)


tfidf = joblib.load('tfidf.pkl')
model = joblib.load('mnb_model.sav')

@app.route('/',methods=['POST','GET'])
def result():

    if request.method == 'POST':

    

        msgg= str(request.form['msg'])
        # msgg=[msgg]
        print(msgg)
        transformed_text = transform(msgg)
        print(transformed_text)
        

        # X= np.array([[msg]])

        cv = tfidf.transform([transformed_text])
        print(cv)



        pred=model.predict(cv)[0]
        print(pred)
        if pred == 1:
            return render_template('spam.html')
        else:
            return render_template('ham.html')

        # return redirect(url_for("predict",result=pred))
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run()
