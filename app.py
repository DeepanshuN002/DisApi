from flask import Flask,render_template,redirect,url_for,request,jsonify
from flask_cors import CORS, cross_origin
import json
import numpy as np
import pandas as pd
import requests
import pickle
app=Flask(__name__)

CORS(app)

model=pickle.load(open('DiseaseModel.pkl','rb'))
@app.route('/')
def home():
    return "Deepanshu"

@app.route('/dis',methods=['POST'])
def dis():
    input = request.json["input_array"]
    inp=list(input.split(','))
    #input_array = np.array(input_array).reshape(1, -1)
    l=[0]*92
    for k in range(len(inp)-1):
        i=inp[k]
        j=int(i)
        l[j]=1
    tst=[]
    tst.append(l)
    prediction = model.predict(tst)
    dislist=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']
    pred=dislist[prediction[0]]
    return pred
    #return jsonify({"prediction": pred})


if __name__=='__main__':
    app.run(debug=True)
