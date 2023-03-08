from flask import Flask,render_template,redirect,url_for,request,jsonify
import json
import numpy as np
import pandas as pd
import requests
import pickle
app=Flask(__name__)

model=pickle.load(open('DiseaseModel.pkl','rb'))
@app.route('/')
def home():
    return "Deepanshu"

@app.route('/dis',methods=['POST'])
def dis():
    data=request.form.get("data")        
    l=[0]*92
    for i in data:
        l[i]=1
    l=l.transpose() 
    x=model.predict(l)
    dis=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
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
    return dis[x-1]
    


if __name__=='__main__':
    app.run(debug=True)