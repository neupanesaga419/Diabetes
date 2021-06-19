from flask import Flask,render_template,url_for,request
import numpy as np
import pickle

model= pickle.load(open('diabetes.py','rb'))
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    no_of_pregnant_time = request.form['no_of_pregnant_time']
    gulcose_level = request.form['gulcose_level']
    blood_pressure = request.form['blood_pressure']
    skin_thickness = request.form['skin_thickness']
    insulin_level = request.form['insulin_level']
    body_mass_index = request.form['body_mass_index']
    diabetic_pedigree_index = request.form['diabetic_pedigree_index']
    age = request.form['age']
    arr = np.array([[no_of_pregnant_time,gulcose_level,blood_pressure,skin_thickness,insulin_level,body_mass_index,diabetic_pedigree_index,age]])
    predict= model.predict(arr)
    print(predict)
    return render_template("predict.html",data=predict)

if __name__ =="__main__":
    app.run(debug=True)
