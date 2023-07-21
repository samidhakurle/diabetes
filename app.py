import pandas as pd
from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__)
model=pickle.load(open('Diabetes_Classifier.pkl','rb'))

@app.route("/")
def homepage():
    return render_template('homepage.html')


@app.route("/predict",methods=['POST','GET'])
def predict():
    pregnancies=float(request.form['pregnancies'])
    glucose=float(request.form['glucose'])
    bp=float(request.form['bp'])
    skin_thickness=float(request.form['skin_thickness'])
    insulin=float(request.form['insulin'])
    bmi=float(request.form['bmi'])
    DBF=float(request.form['DBF'])
    AGE=float(request.form['AGE'])

    data=[np.array([pregnancies,glucose,bp,skin_thickness,insulin,bmi,DBF,AGE])]
    result=model.predict(data)
    print(result)
    
    msg="Prediction: "+str(result[0])
    
    return render_template('homepage.html',prediction_value=msg)


if __name__=='__main__':
    app.run(debug=True) 
